"""Algorithm 1: evidence-consistent merging of repair proposals.

The coordinator treats every proposal as a materialised view over its evidence
queries.  It indexes candidate interference once (`hypergraph`), admits
proposals greedily by confidence-weighted gain under three exact tests, keeps
admitted repairs current by retain/replay/regenerate, and finally removes the
residual constraint violations with one NULL patch.

It reads a proposal's patch, query text, score and replayable flag.  Programs
run only through `Fleet`, i.e. as calls to the client that owns them, which is
what makes a federated run reproduce a central one.
"""

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
    threshold: float = 0.5           # tau: ignore proposals below it
    regeneration_cap: int = 2        # gamma
    use_invalidation: bool = True    # currency check (E_inv + delta queries)
    use_replay: bool = True          # otherwise a stale repair is regenerated
    use_dc_test: bool = True         # admission test (iii)
    use_null_patch: bool = True
    use_retests: bool = True         # re-queue proposals rejected only by (iii)
    score_order: bool = True         # otherwise arrival order
    always_regenerate: bool = False  # OCC-RV: regenerate every candidate reader
    recalibrate: dict[str, Any] | None = None   # per-client monotone score maps


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
        self.selected: dict[str, Proposal] = {}        # lineage -> admitted version
        self.deferred: set[str] = set()                # G
        self.retest: set[str] = set()                  # U
        self.generations: dict[str, int] = {}
        self.stats: dict[str, Any] = {
            "retained": 0, "replayed": 0, "regenerated": 0, "regeneration_needed": 0,
            "write_rejected": 0, "stale_rejected": 0, "dc_rejected": 0, "cycle_rejected": 0,
            "candidate_edges": 0, "rounds": 0, "replanned": 0, "pool": 0, "frozen": 0,
        }
        self._writers: dict[Cell, set[str]] = {}    # cell -> lineages writing it
        self._best: dict[Cell, float] = {}          # cell -> best admitted score
        self._read: set[Cell] = set()               # cells a member's query reads
        self._pending: Proposal | None = None       # the candidate of the current trial
        self._rejected_at: dict[str, int] = {}      # lineage -> admissions in its component
        self._component_admissions: dict[str, int] = {}
        self._admissions = 0

    # == entry points =====================================================
    def merge(self, pool: Iterable[Proposal]) -> Report:
        """One batch.  Calling it again resumes from the current S (a wave)."""
        arrivals = self._admit_to_pool(pool)
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
        return self.finish()

    def finish(self) -> Report:
        """Remove the residual violations and publish the merged delta."""
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

    # == pool and scores ===================================================
    def _admit_to_pool(self, pool: Iterable[Proposal]) -> list[Proposal]:
        """Keep proposals above tau that change a cell, then index them."""
        arrivals: list[Proposal] = []
        for proposal in pool:
            if proposal.score < self.config.threshold:
                continue      # tau applies to the client's own score
            proposal = self._recalibrate(proposal)
            if not any(self.base.value(w.cell) != _norm(w.value) for w in proposal.writes):
                continue
            if proposal.lineage in self.selected:
                continue
            self.graph.add(proposal, self.base, self.constraints)
            arrivals.append(proposal)
        # A6: a replayable lineage on a cycle of the replay graph is frozen.
        # Two program rules writing the same attribute are the usual cause.
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
        """f(S + r) - f(S); submodular, so an older gain is an upper bound."""
        if not self.config.score_order:
            return 1.0
        return sum(max(0.0, proposal.score - self._best.get(cell, 0.0))
                   for cell in proposal.write_cells)

    # == selection loop ====================================================
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
                self.retest.add(proposal.lineage)   # Lemma 4.2: nothing can have changed
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
        """Test (i): a shared cell with a different value, or either replayable."""
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
        """No admission in this component since *this* proposal's last rejection."""
        return self._rejected_at.get(proposal.lineage, -1) == self._admitted_in(proposal)

    # == Settle (test (ii)) ================================================
    def _index(self, member: Proposal) -> None:
        """Add one admitted member to the write and read indexes.

        A replay keeps a member's write cells and score, so only a new
        admission changes them.
        """
        for cell in member.write_cells:
            self._writers.setdefault(cell, set()).add(member.lineage)
            self._best[cell] = max(self._best.get(cell, 0.0), member.score)
        self._read |= member.write_cells
        for query in member.queries:
            static = query.static_cells()
            if static is not None:
                self._read |= {Cell(r, c) for r, c in static}

    def _settle(self, candidate: Proposal) -> tuple[Table, list[Proposal], set[Cell]] | None:
        """Trial on a copy: replay every member whose answers the patch changes."""
        members = {m.lineage: m for m in self.selected.values()}
        members[candidate.lineage] = candidate
        self._pending = candidate
        # E_inv(S) is acyclic by the invariant, so a new cycle has to run
        # through the candidate's own component (Lemma 4.2).
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
        """The cells of `member` that no other member of the set writes.

        The write index answers this without scanning the selection.
        """
        pending = self._pending
        out = []
        for cell in member.write_cells:
            shared = any(o != member.lineage and o in members
                         for o in self._writers.get(cell, ()))
            if not shared and pending is not None and pending.lineage != member.lineage:
                shared = cell in pending.write_cells   # the candidate is not indexed yet
            if not shared:
                out.append(cell)
        return out

    @contextmanager
    def _excluding(self, table: Table, member: Proposal, members: dict[str, Proposal]):
        """`D^{-r}_T` in place: reset the cells only `member` writes, then restore.

        The instance is read-only inside the block, so mutating and restoring
        saves copying the whole relation on every test.
        """
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
        """A copy of `D^{-r}_T`, for callers that keep it beyond one block."""
        cells = self._resets(member, members)
        if not cells:
            return table
        clone = table.copy()
        clone.apply(Write(c, self.base.value(c)) for c in cells)
        return clone

    # == test (iii) ========================================================
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
        """Test (iii) for one member, measuring `D^{-r}` in place first."""
        with self._excluding(trial, member, index) as before:
            counts = counts_before(before, self.constraints, member.write_cells)
            keys = keys_before(before, self.constraints, member.write_cells)
        return worsens_from(counts, keys, trial, self.constraints, member.write_cells)

    def _protected_near(self, local: Sequence[Proposal], rows: set[str], trial: Table,
                        index: dict[str, Proposal]) -> frozenset[Cell]:
        """Protection for the cells this trial could put in a violation.

        `self._read` holds the written cells and the row-pinned provenance of
        the members admitted so far; the candidate is not in it yet, and a
        member with a value-dependent query has to be evaluated.  Both are in
        `local`, which covers every member writing into an affected group, so
        the result is exactly Prot on the cells that matter here.
        """
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

    # == protected cells and the NULL patch ================================
    def protected(self, members: Iterable[Proposal], table: Table | None = None) -> frozenset[Cell]:
        """Written cells plus the provenance of each member's evidence."""
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
        """A greedy hitting set over the unprotected value cells, reverse-deleted."""
        remaining = violations(self.table, self.constraints)
        chosen: list[Cell] = []
        while remaining:
            counts: dict[Cell, int] = {}
            for violation in remaining:
                for cell in violation.value_cells - protected:
                    counts[cell] = counts.get(cell, 0) + 1
            if not counts:
                raise RuntimeError("a residual violation has no unprotected value cell; test (iii) was skipped")
            cell = min(counts, key=lambda c: (-counts[c], c.row, c.column))
            chosen.append(cell)
            remaining = [v for v in remaining if cell not in v.value_cells]
        trial = self.table.copy()
        trial.apply(Write(c, None) for c in chosen)
        for cell in list(reversed(chosen)):
            trial.apply([Write(cell, self.table.value(cell))])
            if violations(trial, self.constraints, [cell.row]):
                trial.apply([Write(cell, None)])      # it was needed after all
            else:
                chosen.remove(cell)
        return [Write(cell, None) for cell in chosen]

    # == regeneration ======================================================
    def _regenerate_round(self) -> list[Proposal]:
        """One batch of agent calls for the deferred lineages below the cap."""
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
            self.graph.add(version, self.base, self.constraints)   # a re-plan extends H
            out.append(version)
        return out

    def _requeue_retests(self) -> list[Proposal]:
        return [self.graph.proposals[lineage] for lineage in sorted(self.retest)]

    # == measurement =======================================================
    def stale(self, final: Table, members: Sequence[Proposal]) -> int:
        """Members whose evidence answers on `D*` without their own patch moved."""
        index = {m.lineage: m for m in members}
        count = 0
        for member in members:
            without = self._without(final, member, index)
            if not without.matches(member.evidence, member.queries):
                count += 1
        return count


def utility(members: Iterable[Proposal]) -> float:
    """f(S): the best score per written cell (Eq. 5)."""
    best: dict[Cell, float] = {}
    for member in members:
        for cell in member.write_cells:
            best[cell] = max(best.get(cell, 0.0), member.score)
    return sum(best.values())


def _norm(value: Value) -> Value:
    return None if value is None or value == "" else str(value)


class _Queue:
    """A lazily updated max-heap on the gain, with a content-hash tiebreak (A4)."""

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
