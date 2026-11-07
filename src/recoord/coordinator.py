from __future__ import annotations

import heapq
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Sequence

from .constraints import (DenialConstraint, adopts, affected_rows, counts_before,
                          keys_before, violations, worsens_from)
from .federation import CallLog, Fleet
from .hypergraph import Hypergraph, has_cycle
from .model import Cell, Proposal, Value, Write
from .rrr import Action, current, delta
from .table import Table


@dataclass
class CoordinatorConfig:
    threshold: float = 0.5
    regeneration_cap: int = 2
    use_invalidation: bool = True
    use_replay: bool = True
    use_dc_test: bool = True
    use_null_patch: bool = True
    use_retests: bool = True
    score_order: bool = True
    always_regenerate: bool = False
    recalibrate: dict[str, Any] | None = None


@dataclass
class Report:
    selected: list[Proposal]
    table: Table
    null_patch: list[Write]
    stats: dict[str, Any] = field(default_factory=dict)
    log: CallLog = field(default_factory=CallLog)
    hypergraph: Hypergraph | None = None

    @property
    def records(self) -> list[dict[str, Any]]:
        return self.table.as_records()


class Coordinator:
    def __init__(self, base: Table, constraints: Sequence[DenialConstraint],
                 fleet: Fleet, config: CoordinatorConfig | None = None):
        self.base = base
        self.constraints = list(constraints)
        self.fleet = fleet
        self.config = config or CoordinatorConfig()
        self.graph = Hypergraph()
        self.table = base.copy()
        self.selected: dict[str, Proposal] = {}
        self.deferred: set[str] = set()
        self.retest: set[str] = set()
        self.generations: dict[str, int] = {}
        self.stats: dict[str, Any] = {
            "retained": 0, "replayed": 0, "regenerated": 0, "regeneration_needed": 0,
            "write_rejected": 0, "stale_rejected": 0, "dc_rejected": 0, "cycle_rejected": 0,
            "candidate_edges": 0, "rounds": 0, "replanned": 0, "pool": 0, "frozen": 0,
        }
        self._writers: dict[Cell, set[str]] = {}
        self._best: dict[Cell, float] = {}
        self._read: set[Cell] = set()
        self._pending: Proposal | None = None
        self._rejected_at: dict[str, int] = {}
        self._component_admissions: dict[str, int] = {}
        self._admissions = 0

    def merge(self, pool: Iterable[Proposal], shards: int = 1) -> Report:
        arrivals = self._admit_to_pool(pool)
        for group in self._shard(arrivals, shards):
            self._rounds(group)
        return self.finish()

    def _shard(self, arrivals: Sequence[Proposal], shards: int) -> list[list[Proposal]]:
        if shards <= 1 or not arrivals:
            return [list(arrivals)]
        by_component: dict[str, list[Proposal]] = {}
        for proposal in arrivals:
            by_component.setdefault(self.graph.find(proposal.lineage), []).append(proposal)
        buckets: list[list[Proposal]] = [[] for _ in range(shards)]

        for root in sorted(by_component, key=lambda r: (-len(by_component[r]), r)):
            min(buckets, key=lambda b: (len(b), buckets.index(b))).extend(by_component[root])
        return [b for b in buckets if b]

    def _rounds(self, arrivals: Sequence[Proposal]) -> None:
        queue = _Queue(self.config.score_order)
        for proposal in arrivals:
            queue.push(proposal, self._gain(proposal))
        while True:
            self.stats["rounds"] += 1
            grown = self._selection_loop(queue)
            refills = self._regenerate_round()
            if grown and self.config.use_retests:
                refills += self._requeue_retests()
            if not refills:
                break
            for proposal in refills:
                queue.push(proposal, self._gain(proposal))

    def add_constraints(self, new: Sequence[DenialConstraint]) -> Report:
        self.constraints.extend(new)
        for proposal in self.graph.proposals.values():
            self.graph._link_by_key(proposal, self.base, list(new))
        touched = {
            self.graph.find(lineage) for lineage, proposal in self.graph.proposals.items()
            if any(self.graph.reachable_keys(proposal, self.base, dc) for dc in new)
        }
        retracted = [m for m in self.selected.values() if self.graph.find(m.lineage) in touched]
        self.stats["retracted"] = len(retracted)
        self.stats["recomputed_components"] = len(touched)
        for member in retracted:
            del self.selected[member.lineage]
        self._rebuild_table()
        for lineage in list(self.retest):
            if self.graph.find(lineage) in touched:
                self.retest.discard(lineage)
        self._rounds([self.graph.proposals[m.lineage] for m in retracted])
        return self.finish()

    def _rebuild_table(self) -> None:
        self.table = self.base.copy()
        self._writers, self._best, self._read = {}, {}, set()
        order = self.graph.order(list(self.selected)) or sorted(self.selected)
        for lineage in order:
            member = self.selected[lineage]
            self.table.apply(member.writes)
            self._index(member)

    def finish(self) -> Report:
        protected = self.protected(self.selected.values())
        patch = self._null_patch(protected) if self.config.use_null_patch else []
        table = self.table.copy()
        table.apply(patch)
        members = list(self.selected.values())
        stats = dict(self.stats)
        stats.update({
            "selected": len(members),
            "null_cells": len(patch),
            "violations": len(violations(table, self.constraints)),
            "stale": self.stale(table, members),
            "utility": utility(members),
            "components": len(self.graph.components()),
            "largest_component": max((len(v) for v in self.graph.components().values()), default=0),
        })
        delta = [Write(cell, table.value(cell)) for cell in sorted(
            {c for m in members for c in m.write_cells} | {w.cell for w in patch})]
        self.fleet.broadcast(delta)
        return Report(members, table, list(patch), stats, self.fleet.log, self.graph)

    def _admit_to_pool(self, pool: Iterable[Proposal]) -> list[Proposal]:
        arrivals: list[Proposal] = []
        for proposal in pool:
            if proposal.score < self.config.threshold:
                continue
            proposal = self._recalibrate(proposal)
            if not any(self.base.value(w.cell) != _norm(w.value) for w in proposal.writes):
                continue
            if proposal.lineage in self.selected:
                continue
            self.graph.add(proposal, self.base, self.constraints)
            arrivals.append(proposal)

        cycles = self.graph.replay_cycles()
        self.stats["frozen"] += len(cycles)
        for lineage in cycles:
            proposal = self.graph.proposals[lineage]
            if proposal.program is not None:
                frozen = replace(proposal, program=replace(proposal.program, replayable=False))
                self.graph.proposals[lineage] = frozen
                arrivals = [frozen if p.lineage == lineage else p for p in arrivals]
        self.stats["pool"] += len(arrivals)
        self.stats["candidate_edges"] = sum(len(v) for v in self.graph.invalidation.values())
        return arrivals

    def _recalibrate(self, proposal: Proposal) -> Proposal:
        maps = self.config.recalibrate or {}
        fn = maps.get(proposal.client) or maps.get("*")
        return proposal if fn is None else replace(proposal, score=float(fn(proposal.score)))

    def _gain(self, proposal: Proposal) -> float:
        if not self.config.score_order:
            return 1.0
        return sum(max(0.0, proposal.score - self._best.get(cell, 0.0))
                   for cell in proposal.write_cells)

    def _selection_loop(self, queue: "_Queue") -> bool:
        grown = False
        while True:
            item = queue.pop(self._gain)
            if item is None:
                return grown
            proposal = item
            was_retest = proposal.lineage in self.retest
            self.retest.discard(proposal.lineage)
            if was_retest and self.config.use_retests and self._component_unchanged(proposal):
                self.retest.add(proposal.lineage)
                continue

            if self.config.use_invalidation and not current(self.table, proposal):
                answers = self.table.evidence(proposal.queries).answers
                replayed = self._replay(proposal, self.table, answers)
                if replayed is None:
                    self.stats["regeneration_needed"] += 1
                    self.deferred.add(proposal.lineage)
                    continue
                proposal = replayed
            else:
                self.stats[Action.RETAIN.value + "ed"] += 1

            if self._write_conflict(proposal):
                self.stats["write_rejected"] += 1
                continue

            trial = self._settle(proposal)
            if trial is None:
                self._pending = None
                continue
            table, members, touched = trial
            ok = not self.config.use_dc_test or self._dc_test(table, members, touched)
            self._pending = None
            if not ok:
                self.stats["dc_rejected"] += 1
                self.retest.add(proposal.lineage)
                self._rejected_at[proposal.lineage] = self._admitted_in(proposal)
                continue
            self.table = table
            self.selected = {m.lineage: m for m in members}
            self.graph.proposals.update(self.selected)
            self._index(proposal)
            self._admissions += 1
            root = self.graph.find(proposal.lineage)
            self._component_admissions[root] = self._component_admissions.get(root, 0) + 1
            grown = True

    def _replay(self, proposal: Proposal, table: Table, answers) -> Proposal | None:
        if not self.config.use_replay or self.config.always_regenerate:
            return None
        if not proposal.replayable:
            return None
        writes = self.fleet.replay(proposal, answers,
                                   delta(proposal.evidence.answers, answers))
        if writes is None or {w.cell for w in writes} != set(proposal.write_cells):
            return None
        self.stats[Action.REPLAY.value + "ed"] += 1
        return proposal.with_patch(writes, table.evidence(proposal.queries))

    def _write_conflict(self, proposal: Proposal) -> bool:
        patch = proposal.patch
        for cell in proposal.write_cells:
            for lineage in self._writers.get(cell, ()):
                member = self.selected.get(lineage)
                if member is None:
                    continue
                if proposal.replayable or member.replayable:
                    return True
                if _norm(member.patch[cell]) != _norm(patch[cell]):
                    return True
        return False

    def _admitted_in(self, proposal: Proposal) -> int:
        return self._component_admissions.get(self.graph.find(proposal.lineage), 0)

    def _component_unchanged(self, proposal: Proposal) -> bool:
        return self._rejected_at.get(proposal.lineage, -1) == self._admitted_in(proposal)

    def _index(self, member: Proposal) -> None:
        for cell in member.write_cells:
            self._writers.setdefault(cell, set()).add(member.lineage)
            self._best[cell] = max(self._best.get(cell, 0.0), member.score)
        self._read |= member.write_cells
        for query in member.queries:
            static = query.static_cells()
            if static is not None:
                self._read |= {Cell(r, c) for r, c in static}

    def _settle(self, candidate: Proposal) -> tuple[Table, list[Proposal], set[Cell]] | None:
        members = {m.lineage: m for m in self.selected.values()}
        members[candidate.lineage] = candidate
        self._pending = candidate

        local = self._component_members(candidate, members)
        if has_cycle(list(local), {m: self.graph.invalidation.get(m, set()) & set(local)
                                   for m in local}):
            self.stats["cycle_rejected"] += 1
            return None
        table = self.table.copy()
        table.apply(candidate.writes)
        frontier = set(candidate.write_cells)
        done: set[str] = set()
        while self.config.use_invalidation:
            order = self.graph.order(list(local))
            if order is None:
                self.stats["cycle_rejected"] += 1
                return None
            readers = self.graph.candidate_readers(frontier, local)
            stale = [
                lineage for lineage in order
                if lineage in readers and lineage not in done and lineage != candidate.lineage
                and self._is_stale(table, members[lineage], members)
            ]
            if not stale:
                break
            lineage = stale[0]
            member = members[lineage]
            with self._excluding(table, member, members) as without:
                evidence = without.evidence(member.queries)
            answers = evidence.answers
            if not (self.config.use_replay and member.replayable):
                self.stats["stale_rejected"] += 1
                return None
            writes = self.fleet.replay(member, answers,
                                       delta(member.evidence.answers, answers))
            if writes is None or {w.cell for w in writes} != set(member.write_cells):
                self.stats["stale_rejected"] += 1
                return None
            members[lineage] = member.with_patch(writes, evidence)
            table.apply(writes)
            frontier |= set(member.write_cells)
            done.add(lineage)
            self.stats["replayed"] += 1
        return table, list(members.values()), frontier

    def _resets(self, member: Proposal, members: dict[str, Proposal]) -> list[Cell]:
        pending = self._pending
        out = []
        for cell in member.write_cells:
            shared = any(o != member.lineage and o in members
                         for o in self._writers.get(cell, ()))
            if not shared and pending is not None and pending.lineage != member.lineage:
                shared = cell in pending.write_cells
            if not shared:
                out.append(cell)
        return out

    @contextmanager
    def _excluding(self, table: Table, member: Proposal, members: dict[str, Proposal]):
        cells = self._resets(member, members)
        if not cells:
            yield table
            return
        saved = [Write(c, table.value(c)) for c in cells]
        table.apply(Write(c, self.base.value(c)) for c in cells)
        try:
            yield table
        finally:
            table.apply(saved)

    def _without(self, table: Table, member: Proposal, members: dict[str, Proposal]) -> Table:
        cells = self._resets(member, members)
        if not cells:
            return table
        clone = table.copy()
        clone.apply(Write(c, self.base.value(c)) for c in cells)
        return clone

    def _dc_test(self, trial: Table, members: Sequence[Proposal], touched: Iterable[Cell]) -> bool:
        columns = {c for dc in self.constraints for c in dc.columns}
        changed = [c for c in touched
                   if c.column in columns and self.table.value(c) != trial.value(c)]
        if not changed:
            return True
        rows = affected_rows(trial, self.constraints, changed)
        index = {m.lineage: m for m in members}
        local = [m for m in members if any(c.row in rows for c in m.write_cells)]
        for member in local:
            if self._worsens(trial, member, index):
                return False
        return not adopts(trial, self.constraints, self._protected_near(local, rows, trial, index), rows)

    def _component_members(self, candidate: Proposal, members: dict[str, Proposal]) -> dict[str, Proposal]:
        root = self.graph.find(candidate.lineage)
        return {k: v for k, v in members.items() if self.graph.find(k) == root}

    def _is_stale(self, table: Table, member: Proposal, members: dict[str, Proposal]) -> bool:
        with self._excluding(table, member, members) as without:
            return not current(without, member)

    def _worsens(self, trial: Table, member: Proposal, index: dict[str, Proposal]) -> bool:
        with self._excluding(trial, member, index) as before:
            counts = counts_before(before, self.constraints, member.write_cells)
            keys = keys_before(before, self.constraints, member.write_cells)
        return worsens_from(counts, keys, trial, self.constraints, member.write_cells)

    def _protected_near(self, local: Sequence[Proposal], rows: set[str], trial: Table,
                        index: dict[str, Proposal]) -> frozenset[Cell]:
        cells = set(self._read)
        for member in local:
            cells |= member.write_cells
            dynamic = []
            for query in member.queries:
                static = query.static_cells()
                if static is None:
                    dynamic.append(query)
                else:
                    cells |= {Cell(r, c) for r, c in static}
            if dynamic:
                with self._excluding(trial, member, index) as without:
                    for query in dynamic:
                        cells |= without.provenance(query)
        return frozenset(cells)

    def protected(self, members: Iterable[Proposal], table: Table | None = None) -> frozenset[Cell]:
        members = list(members)
        table = table if table is not None else self.table
        index = {m.lineage: m for m in members}
        cells: set[Cell] = {c for m in members for c in m.write_cells}
        for member in members:
            dynamic = [q for q in member.queries if q.static_cells() is None]
            for query in member.queries:
                static = query.static_cells()
                if static is not None:
                    cells |= {Cell(r, c) for r, c in static}
            if dynamic:
                without = self._without(table, member, index)
                for query in dynamic:
                    cells |= without.provenance(query)
        return frozenset(cells)

    def _null_patch(self, protected: frozenset[Cell]) -> list[Write]:
        remaining = violations(self.table, self.constraints)
        chosen: list[Cell] = []
        while remaining:
            counts: dict[Cell, int] = {}
            for violation in remaining:
                for cell in violation.value_cells - protected:
                    counts[cell] = counts.get(cell, 0) + 1
            if not counts:
                raise RuntimeError("a residual violation has only protected value cells; "
                               "the constraint test must run before the NULL patch")
            cell = min(counts, key=lambda c: (-counts[c], c.row, c.column))
            chosen.append(cell)
            remaining = [v for v in remaining if cell not in v.value_cells]
        trial = self.table.copy()
        trial.apply(Write(c, None) for c in chosen)
        for cell in list(reversed(chosen)):
            trial.apply([Write(cell, self.table.value(cell))])
            if violations(trial, self.constraints, [cell.row]):
                trial.apply([Write(cell, None)])
            else:
                chosen.remove(cell)
        return [Write(cell, None) for cell in chosen]

    def _regenerate_round(self) -> list[Proposal]:
        batch, self.deferred = sorted(self.deferred), set()
        out: list[Proposal] = []
        for lineage in batch:
            proposal = self.graph.proposals[lineage]
            if self.generations.get(lineage, 0) >= self.config.regeneration_cap:
                continue
            answers = self._without(self.table, proposal, self.selected).evidence(proposal.queries).answers
            version = self.fleet.regenerate(proposal, self.table, answers)
            self.generations[lineage] = self.generations.get(lineage, 0) + 1
            if version is None:
                continue
            self.stats["regenerated"] += 1
            if version.read_columns != proposal.read_columns or version.write_cells != proposal.write_cells:
                self.stats["replanned"] += 1
            self.graph.add(version, self.base, self.constraints)
            out.append(version)
        return out

    def _requeue_retests(self) -> list[Proposal]:
        return [self.graph.proposals[lineage] for lineage in sorted(self.retest)]

    def stale(self, final: Table, members: Sequence[Proposal]) -> int:
        index = {m.lineage: m for m in members}
        count = 0
        for member in members:
            without = self._without(final, member, index)
            if not without.matches(member.evidence, member.queries):
                count += 1
        return count


def utility(members: Iterable[Proposal]) -> float:
    best: dict[Cell, float] = {}
    for member in members:
        for cell in member.write_cells:
            best[cell] = max(best.get(cell, 0.0), member.score)
    return sum(best.values())


def _norm(value: Value) -> Value:
    return None if value is None or value == "" else str(value)


class _Queue:
    def __init__(self, by_score: bool = True):
        self.by_score = by_score
        self._heap: list[tuple[float, str, int, Proposal]] = []
        self._tick = 0

    def push(self, proposal: Proposal, gain: float) -> None:
        self._tick += 1
        key = -gain if self.by_score else 0.0
        heapq.heappush(self._heap, (key, proposal.tiebreak if self.by_score else f"{self._tick:012d}",
                                    self._tick, proposal))

    def pop(self, gain_of) -> Proposal | None:
        while self._heap:
            key, tie, tick, proposal = heapq.heappop(self._heap)
            if not self.by_score:
                return proposal
            fresh = gain_of(proposal)
            if fresh <= 0:
                continue
            if self._heap and -fresh > self._heap[0][0] + 1e-12:
                heapq.heappush(self._heap, (-fresh, tie, tick, proposal))
                continue
            return proposal
        return None
