from __future__ import annotations

import random
import re
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

from .federation import Usage
from .model import Answer, Atom, Cell, Program, Proposal, Query, Value, Write
from .table import Table, norm
from . import programs

ROLES = ("imputer", "normalizer", "deriver", "matcher")


@dataclass(frozen=True)
class Rule:
    name: str
    target: str
    sources: tuple[str, ...] = ()
    program: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    when: str = "always"
    pattern: str = ""
    scope: str = "row"
    group_key: tuple[str, ...] = ()
    rows: tuple[str, ...] = ()
    score: float = 0.8
    role: str = "normalizer"
    invented: bool = False

    @property
    def replayable(self) -> bool:
        return self.program is not None

    def fires(self, row: dict[str, Value]) -> bool:
        if any(row.get(c) is None for c in self.sources if c != self.target):
            return False
        probe = row.get(self.target) if self.when in ("null", "not_null", "numeric") else \
            " ".join(str(row.get(c)) for c in (self.sources or (self.target,)) if row.get(c) is not None)
        if self.when == "null":
            return probe is None
        if self.when == "not_null":
            return probe is not None
        if self.when == "numeric":
            try:
                float(str(probe))
            except (TypeError, ValueError):
                return False
            return True
        if self.when == "regex":
            return bool(re.search(self.pattern, str(probe), re.IGNORECASE))
        return True


def parse_rules(specs: Sequence[dict[str, Any]]) -> list[Rule]:
    out = []
    for spec in specs:
        out.append(Rule(
            name=spec["name"],
            target=spec["target"],
            sources=tuple(spec.get("sources", ())),
            program=spec.get("program"),
            options=dict(spec.get("options", {})),
            when=spec.get("when", "always"),
            pattern=spec.get("pattern", ""),
            scope=spec.get("scope", "row"),
            group_key=tuple(spec.get("group_key", ())),
            rows=tuple(str(r) for r in spec.get("rows", ())),
            score=float(spec.get("score", 0.8)),
            role=spec.get("role", "normalizer"),
            invented=bool(spec.get("invented", False)),
        ))
    return out


class TemplateAgent:
    def __init__(self, name: str, rules: Sequence[Rule], seed: int = 0,
                 max_proposals: int | None = None, turns: int = 3):
        self.name = name
        self.rules = list(rules)
        self.seed = seed
        self.max_proposals = max_proposals
        self.turns = turns
        self._usage = Usage()
        self._rng = random.Random(f"{name}:{seed}")

    @property
    def usage(self) -> Usage:
        return self._usage

    def propose(self, view: Table, client: str) -> list[Proposal]:
        out: list[Proposal] = []
        for rule in self.rules:
            out.extend(self._fire(rule, view, client))
        out.sort(key=lambda p: p.id)
        if self.max_proposals is not None:
            out = out[: self.max_proposals]

        self._usage.calls += len(self.rules)
        self._usage.input_tokens += self.turns * 4_870 * len(self.rules)
        self._usage.output_tokens += self.turns * 163 * len(self.rules)
        return out

    def _fire(self, rule: Rule, view: Table, client: str) -> list[Proposal]:
        if rule.scope == "group":
            return self._fire_group(rule, view, client)
        if rule.scope == "row_in_group":
            return self._fire_in_group(rule, view, client)
        out = []
        for rid, row in view.rows():
            if rule.rows and rid not in rule.rows:
                continue
            if not rule.fires(row):
                continue
            proposal = self._build(rule, view, client, [rid], Cell(rid, rule.target))
            if proposal is not None:
                out.append(proposal)
        return out

    def _buckets(self, rule: Rule, view: Table) -> dict[tuple[Value, ...], list[str]]:
        out: dict[tuple[Value, ...], list[str]] = {}
        for rid, row in view.rows():
            key = tuple(row.get(c) for c in rule.group_key)
            if not any(k is None for k in key):
                out.setdefault(key, []).append(rid)
        return out

    def _fire_in_group(self, rule: Rule, view: Table, client: str) -> list[Proposal]:
        buckets = self._buckets(rule, view)
        out = []
        for key, members in sorted(buckets.items(), key=lambda kv: str(kv[0])):
            if len(members) < 2:
                continue
            for rid in members:
                if rule.rows and rid not in rule.rows:
                    continue
                if not rule.fires(view.row(rid)):
                    continue
                proposal = self._build(rule, view, client, members, Cell(rid, rule.target))
                if proposal is not None:
                    out.append(proposal)
        return out

    def _fire_group(self, rule: Rule, view: Table, client: str) -> list[Proposal]:
        buckets = self._buckets(rule, view)
        out = []
        for key, members in sorted(buckets.items(), key=lambda kv: str(kv[0])):
            targets = [Cell(r, rule.target) for r in members
                       if (not rule.rows or r in rule.rows) and rule.fires(view.row(r))]
            if not targets or len(members) < 2:
                continue
            proposal = self._build(rule, view, client, members, *targets)
            if proposal is not None:
                out.append(proposal)
        return out

    def _build(self, rule: Rule, view: Table, client: str, rows: Sequence[str],
               *targets: Cell) -> Proposal | None:
        columns = tuple(rule.sources or (rule.target,))
        pid = f"{client}:{rule.name}:{targets[0].row}"
        query = Query(name=f"{pid}/q0", atoms=(Atom(columns=columns, rows=tuple(rows)),))
        evidence = view.evidence([query])
        spec = Program(kind=rule.program or "constant", targets=tuple(targets),
                       options=dict(rule.options), replayable=rule.replayable)
        if rule.program is not None:
            if not programs.holds(spec, evidence.answers, self.private):
                return None
            writes = programs.run(spec, evidence.answers, self.private)
        else:
            writes = self.judge(rule, view, targets, evidence.answers)
        if not writes:
            return None
        writes = [w for w in writes if norm(w.value) != view.value(w.cell)]
        if not writes:
            return None
        options = dict(spec.options)
        if rule.program is None:
            options["values"] = [w.value for w in writes]
        spec = replace(spec, targets=tuple(w.cell for w in writes), options=options)
        return Proposal(
            id=pid, client=client, score=self.confidence(rule, writes),
            writes=tuple(writes), queries=(query,), evidence=evidence,
            program=spec, lineage=pid, group=f"{rule.name}:{rows[0]}",
            meta={"rule": rule.name, "role": rule.role},
        )

    def regenerate(self, proposal: Proposal, view: Table, answers: tuple[Answer, ...]) -> Proposal | None:
        rule = next((r for r in self.rules if r.name == proposal.meta.get("rule")), None)
        if rule is None:
            return None
        self._usage.calls += 1
        self._usage.input_tokens += 3_550
        self._usage.output_tokens += 290
        targets = tuple(w.cell for w in proposal.writes)
        if rule.program is not None:
            spec = replace(Program(rule.program, targets, dict(rule.options)), replayable=True)
            if not programs.holds(spec, answers, self.private):
                return None
            writes = programs.run(spec, answers, self.private)
        else:
            writes = self.judge(rule, view, targets, answers)
        if not writes:
            return None
        evidence = view.evidence(proposal.queries)
        return replace(proposal, writes=tuple(writes), evidence=evidence,
                       score=self.confidence(rule, writes))

    @property
    def private(self) -> dict[str, Any]:
        return {}

    def confidence(self, rule: Rule, writes: Sequence[Write]) -> float:
        return rule.score

    def judge(self, rule: Rule, view: Table, targets: Sequence[Cell],
              answers: tuple[Answer, ...]) -> list[Write] | None:
        raise NotImplementedError


class RuleAgent(TemplateAgent):
    def __init__(self, name: str, rules: Sequence[Rule], **kw: Any):
        super().__init__(name, [r for r in rules
                                if r.program is not None and not r.invented], **kw)

    def judge(self, rule, view, targets, answers):
        return None


class OfflineAgent(TemplateAgent):
    def __init__(self, name: str, rules: Sequence[Rule], reference: Table | None = None,
                 accuracy: float = 0.75, private: dict[str, Any] | None = None,
                 jitter: float = 0.12, **kw: Any):
        super().__init__(name, rules, **kw)
        self.reference = reference
        self.accuracy = accuracy
        self.jitter = jitter
        self._private = private or {}

    @property
    def private(self) -> dict[str, Any]:
        return self._private

    def confidence(self, rule: Rule, writes: Sequence[Write]) -> float:
        if not self.jitter:
            return rule.score
        return round(min(0.99, max(0.05, rule.score + self._rng.uniform(-self.jitter, self.jitter))), 3)

    def judge(self, rule, view, targets, answers):
        out: list[Write] = []
        for cell in targets:
            truth = self.reference.value(cell) if self.reference and cell.row in self.reference else None
            if truth is not None and self._rng.random() < self.accuracy:
                out.append(Write(cell, truth))
            else:
                out.append(Write(cell, self._plausible(view, cell)))
        return [w for w in out if w.value is not None]

    def _plausible(self, view: Table, cell: Cell) -> Value:
        pool = [v for _, row in view.rows() if (v := row.get(cell.column)) is not None]
        return self._rng.choice(pool) if pool else None


class LLMAgent(TemplateAgent):
    _last_score: float | None = None

    def __init__(self, name: str, rules: Sequence[Rule], client: Any,
                 private: dict[str, Any] | None = None, **kw: Any):
        super().__init__(name, rules, **kw)
        self.llm = client
        self._private = private or {}

    @property
    def private(self) -> dict[str, Any]:
        return self._private

    def confidence(self, rule: Rule, writes: Sequence[Write]) -> float:
        return float(self._last_score if self._last_score is not None else rule.score)

    def judge(self, rule, view, targets, answers):
        reply = self.llm.repair(
            rule=rule.name, columns=[c.column for c in targets],
            rows=[c.row for c in targets], answers=answers,
            instruction=rule.options.get("instruction", ""),
        )
        self._usage = self._usage.add(reply.usage)
        self._last_score = reply.confidence
        if not reply.values:
            return None
        return [Write(cell, value) for cell, value in zip(targets, reply.values) if value is not None]


def build_agent(kind: str, name: str, rules: Sequence[Rule], **kw: Any) -> TemplateAgent:
    if kind == "offline":
        return OfflineAgent(name, rules, **kw)
    if kind == "rules":
        return RuleAgent(name, rules, **{k: v for k, v in kw.items()
                                         if k in ("seed", "max_proposals", "turns")})
    if kind == "llm":
        return LLMAgent(name, rules, **kw)
    raise ValueError(f"unknown agent kind {kind!r}")
