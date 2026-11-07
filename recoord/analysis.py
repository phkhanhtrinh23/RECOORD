from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Sequence

from .constraints import (DenialConstraint, adopts, affected_rows, changed_cells,
                          worsens)
from .coordinator import utility
from .federation import Fleet
from .hypergraph import Hypergraph, has_cycle
from .model import Cell, Proposal, Write
from .table import Table


@dataclass
class ComponentReport:
    root: str
    size: int
    enumerated: bool
    p: int | None = None
    subset_closed: bool | None = None
    no_cancellation: bool | None = None
    optimum: float | None = None
    optimum_open: float | None = None
    greedy: float | None = None
    epsilon: int = 0
    bound: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class ValidityChecker:
    def __init__(self, base: Table, constraints: Sequence[DenialConstraint],
                 fleet: Fleet, graph: Hypergraph):
        self.base = base
        self.constraints = list(constraints)
        self.fleet = fleet
        self.graph = graph

    def valid(self, members: Sequence[Proposal]) -> bool:
        index = {m.lineage: m for m in members}

        for a, b in combinations(members, 2):
            shared = a.write_cells & b.write_cells
            if shared and (a.replayable or b.replayable
                           or any(a.patch[c] != b.patch[c] for c in shared)):
                return False

        if has_cycle(list(index), {m: self.graph.invalidation.get(m, set()) & set(index) for m in index}):
            return False
        settled = self._instance(members)
        if settled is None:
            return False
        table, index = settled
        members = list(index.values())
        changed = [c for c in changed_cells(self.base, table)
                   if any(c.column in dc.columns for dc in self.constraints)]
        rows = affected_rows(table, self.constraints, changed)
        for member in members:
            without = self._without(table, member, index)
            if not without.matches(member.evidence, member.queries):
                return False
            if any(c.row in rows for c in member.write_cells) \
                    and worsens(without, table, self.constraints, member.write_cells):
                return False
        protected = self._protected(table, members, index)
        return not adopts(table, self.constraints, protected, rows)

    def _instance(self, members: Sequence[Proposal]) -> tuple[Table, dict[str, Proposal]] | None:
        index = {m.lineage: m for m in members}
        order = self.graph.order(list(index))
        if order is None:
            return None
        table = self.base.copy()
        for lineage in order:
            member = index[lineage]
            if member.replayable:
                answers = self._without(table, member, index).evidence(member.queries).answers
                writes = self.fleet.replay(member, answers)
                if writes is None or {w.cell for w in writes} != set(member.write_cells):
                    return None
                index[lineage] = member.with_patch(writes, self._without(table, member, index).evidence(member.queries))
                table.apply(writes)
            else:
                table.apply(member.writes)
        return table, index

    def _without(self, table: Table, member: Proposal, index: dict[str, Proposal]) -> Table:
        others = {c for k, m in index.items() if k != member.lineage for c in m.write_cells}
        resets = [Write(c, self.base.value(c)) for c in member.write_cells - others]
        if not resets:
            return table
        clone = table.copy()
        clone.apply(resets)
        return clone

    def _protected(self, table: Table, members: Sequence[Proposal], index) -> frozenset[Cell]:
        cells = {c for m in members for c in m.write_cells}
        for member in members:
            without = self._without(table, member, index)
            for query in member.queries:
                cells |= without.provenance(query)
        return frozenset(cells)


def analyse_components(base: Table, pool: Sequence[Proposal], constraints: Sequence[DenialConstraint],
                       fleet: Fleet, graph: Hypergraph, greedy_members: Sequence[Proposal],
                       limit: int = 14, use_ilp: bool = True) -> list[ComponentReport]:
    checker = ValidityChecker(base, constraints, fleet, graph)
    by_lineage = {p.lineage: p for p in pool}
    taken = {m.lineage for m in greedy_members}
    out: list[ComponentReport] = []
    for root, lineages in sorted(graph.components().items()):
        members = [by_lineage[l] for l in lineages if l in by_lineage]
        report = ComponentReport(root=root, size=len(members), enumerated=len(members) <= limit)
        if not report.enumerated or not members:
            out.append(report)
            continue
        valid = _enumerate_valid(checker, members)
        alone = {m.lineage for m in members if frozenset([m.lineage]) in valid}
        report.p = _interference_parameter(valid, [m.lineage for m in members])
        report.subset_closed = _subset_closed(valid)
        report.no_cancellation = _no_cancellation(valid, alone)
        report.epsilon = int(any(
            a.write_cells & b.write_cells and not a.replayable and not b.replayable
            and all(a.patch[c] == b.patch[c] for c in a.write_cells & b.write_cells)
            for a, b in combinations(members, 2)))
        scores = {m.lineage: m for m in members}
        report.optimum = max((utility([scores[l] for l in s]) for s in valid), default=0.0)
        report.optimum_open = max(
            (utility([scores[l] for l in s]) for s in valid if set(s) <= alone), default=0.0)
        report.greedy = utility([scores[l] for l in lineages if l in taken])
        if report.p:
            report.bound = report.optimum_open / (report.p + report.epsilon)
        if use_ilp:
            exact = _ilp_optimum(valid, scores)
            if exact is not None:
                report.optimum = exact
        out.append(report)
    return out


def _enumerate_valid(checker: ValidityChecker, members: Sequence[Proposal]) -> set[frozenset[str]]:
    lineages = [m.lineage for m in members]
    index = {m.lineage: m for m in members}
    valid: set[frozenset[str]] = {frozenset()}
    frontier: list[frozenset[str]] = [frozenset()]
    while frontier:
        nxt: list[frozenset[str]] = []
        for base_set in frontier:
            for lineage in lineages:
                if lineage in base_set:
                    continue
                candidate = base_set | {lineage}
                if candidate in valid:
                    continue
                if checker.valid([index[l] for l in sorted(candidate)]):
                    valid.add(candidate)
                    nxt.append(candidate)
        frontier = nxt
    return valid


def _subset_closed(valid: set[frozenset[str]]) -> bool:
    for s in valid:
        for size in range(len(s)):
            if any(frozenset(sub) not in valid for sub in combinations(sorted(s), size)):
                return False
    return True


def _interference_parameter(valid: set[frozenset[str]], lineages: Sequence[str]) -> int:
    downward = _downward_closure(valid)
    best = 1
    for e in lineages:
        if frozenset([e]) not in downward:
            continue
        for b in downward:
            if e in b:
                continue
            minimal = [h for h in _minimal_invalid(downward, lineages) if e in h and (h - {e}) <= b]
            best = max(best, len(minimal))
    return best


def _downward_closure(valid: set[frozenset[str]]) -> set[frozenset[str]]:
    out: set[frozenset[str]] = set()
    for s in valid:
        for size in range(len(s) + 1):
            out.update(frozenset(sub) for sub in combinations(sorted(s), size))
    return out


def _minimal_invalid(downward: set[frozenset[str]], lineages: Sequence[str]) -> list[frozenset[str]]:
    out = []
    for size in range(1, min(4, len(lineages)) + 1):
        for combo in combinations(sorted(lineages), size):
            s = frozenset(combo)
            if s in downward:
                continue
            if all(frozenset(sub) in downward for sub in combinations(combo, size - 1)):
                out.append(s)
    return out


def _no_cancellation(valid: set[frozenset[str]], alone: set[str]) -> bool:
    for t in valid:
        for e in alone:
            if e in t:
                continue
            if any(e in s and t <= s for s in valid) and (t | {e}) not in valid:
                return False
    return True


def _ilp_optimum(valid: set[frozenset[str]], scores: dict[str, Proposal]) -> float | None:
    try:
        import highspy
    except ImportError:
        return None

    sets = sorted(valid, key=lambda s: sorted(s))
    if not sets:
        return 0.0
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", 60.0)
    values = [utility([scores[l] for l in sorted(s)]) for s in sets]
    for value in values:
        h.addVar(0.0, 1.0)
        h.changeColIntegrality(h.getNumCol() - 1, highspy.HighsVarType.kInteger)
    h.changeColsCost(len(values), list(range(len(values))), [-v for v in values])
    h.addRow(1.0, 1.0, len(values), list(range(len(values))), [1.0] * len(values))
    h.run()
    return -h.getObjectiveValue()
