"""Other ways to merge the same proposals, and centralised references.

Three families:

* *merge policies* that accept or reject the uploaded proposals in arrival
  order -- cell-wise registers, the usual concurrency-control validations, and
  transaction repair with value-based validation;
* *variants* of Algorithm 1 that differ only in what they do when a proposal
  may be stale (they are what Proposition 4.5 compares against);
* *central references* that schedule the agents themselves instead of merging
  what they uploaded.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Sequence

from .constraints import DenialConstraint, violations
from .coordinator import Coordinator, CoordinatorConfig, Report, utility
from .federation import Fleet
from .model import Cell, Proposal, Write
from .rrr import current
from .table import Table


def _report(name: str, base: Table, table: Table, members: Sequence[Proposal],
            constraints: Sequence[DenialConstraint], fleet: Fleet,
            extra: dict[str, Any] | None = None) -> Report:
    stats = {
        "method": name, "selected": len(members), "null_cells": 0,
        "violations": len(violations(table, constraints)),
        "stale": _stale(base, table, members),
        "utility": utility(members),
    }
    stats.update(extra or {})
    return Report(list(members), table, [], stats, fleet.log)


def _stale(base: Table, final: Table, members: Sequence[Proposal]) -> int:
    """A member is stale if its evidence moved on the output without its patch."""
    owners: dict[Cell, int] = {}
    for member in members:
        for cell in member.write_cells:
            owners[cell] = owners.get(cell, 0) + 1
    count = 0
    for member in members:
        probe = final.copy()
        probe.apply(Write(c, base.value(c)) for c in member.write_cells if owners[c] == 1)
        if not probe.matches(member.evidence, member.queries):
            count += 1
    return count


# == merge policies =========================================================

def last_writer_wins(base: Table, pool: Sequence[Proposal], constraints, fleet: Fleet, **_: Any) -> Report:
    """A last-writer-wins register per cell: blind to evidence (Prop. 4.7)."""
    table = base.copy()
    for proposal in pool:
        table.apply(proposal.writes)
    return _report("lww", base, table, pool, constraints, fleet)


def occ_write(base: Table, pool: Sequence[Proposal], constraints, fleet: Fleet, **_: Any) -> Report:
    """Write-set validation: rejects overlapping writes, ignores reads."""
    table, taken, written = base.copy(), [], set()
    for proposal in pool:
        if proposal.write_cells & written:
            continue
        table.apply(proposal.writes)
        written |= set(proposal.write_cells)
        taken.append(proposal)
    return _report("occ-write", base, table, taken, constraints, fleet)


def ssi(base: Table, pool: Sequence[Proposal], constraints, fleet: Fleet, **_: Any) -> Report:
    """Serialisable snapshot isolation: abort on a read-write dependency cycle."""
    table, taken, aborts = base.copy(), [], 0
    written: set[Cell] = set()
    for proposal in pool:
        reads = {Cell(r, c) for q in proposal.queries for a in q.atoms
                 for r in (a.rows or base.row_ids) for c in a.read_columns}
        conflict = bool(proposal.write_cells & written) or bool(reads & written)
        if conflict:
            aborts += 1
            continue
        table.apply(proposal.writes)
        written |= set(proposal.write_cells)
        taken.append(proposal)
    return _report("ssi", base, table, taken, constraints, fleet, {"aborts": aborts})


def det_order(base: Table, pool: Sequence[Proposal], constraints, fleet: Fleet, **_: Any) -> Report:
    """Deterministic ordering: a fixed sequence, no validation."""
    order = sorted(pool, key=lambda p: p.tiebreak)
    table = base.copy()
    for proposal in order:
        table.apply(proposal.writes)
    return _report("det-order", base, table, order, constraints, fleet)


def fixpoint_holistic(base: Table, pool: Sequence[Proposal], constraints, fleet: Fleet,
                      rounds: int = 5, **_: Any) -> Report:
    """Re-run every program to a fixpoint, then repair DCs by majority."""
    table = base.copy()
    members = [replace(p) for p in pool]
    for proposal in members:
        table.apply(proposal.writes)
    for _round in range(rounds):
        changed = False
        for i, proposal in enumerate(members):
            if not proposal.replayable:
                continue
            answers = table.evidence(proposal.queries).answers
            writes = fleet.replay(proposal, answers)
            if writes and {w.cell for w in writes} == set(proposal.write_cells) \
                    and any(table.value(w.cell) != w.value for w in writes):
                table.apply(writes)
                members[i] = proposal.with_patch(writes, table.evidence(proposal.queries))
                changed = True
        if not changed:
            break
    _holistic_repair(table, constraints)
    return _report("fixpoint-holistic", base, table, members, constraints, fleet)


def _holistic_repair(table: Table, constraints: Sequence[DenialConstraint], rounds: int = 3) -> None:
    from .constraints import groups, plurality
    for _ in range(rounds):
        if not violations(table, constraints):
            return
        for dc in constraints:
            if dc.kind != "fd" or dc.rhs is None:
                continue
            for members in groups(table, dc).values():
                value = plurality(table, dc, members[0])
                if value is None:
                    continue
                table.apply(Write(Cell(r, dc.rhs), value) for r in members
                            if table.row(r).get(dc.rhs) not in (None, value))


def trvv(base: Table, pool: Sequence[Proposal], constraints, fleet: Fleet,
         dc_check: bool = True, **_: Any) -> Report:
    """Transaction repair with value-based validation, in arrival order.

    Bayou's check-and-merge with a currency check and regeneration: a stale
    reader re-executes its program, or is re-prompted, or aborts.  No commit
    protocol defines a DC test, so with `dc_check` a merge that leaves a new
    violation aborts at commit.
    """
    table, taken = base.copy(), []
    aborts = replays = regens = 0
    for proposal in pool:
        if not current(table, proposal):
            answers = table.evidence(proposal.queries).answers
            writes = fleet.replay(proposal, answers) if proposal.replayable else None
            if writes and {w.cell for w in writes} == set(proposal.write_cells):
                proposal = proposal.with_patch(writes, table.evidence(proposal.queries))
                replays += 1
            else:
                version = fleet.regenerate(proposal, table, answers)
                if version is None:
                    aborts += 1
                    continue
                regens += 1
                proposal = version
        trial = table.copy()
        trial.apply(proposal.writes)
        if dc_check and len(violations(trial, constraints)) > len(violations(table, constraints)):
            aborts += 1
            continue
        table, _ = trial, taken.append(proposal)
    name = "trvv" if dc_check else "trvv-nodc"
    return _report(name, base, table, taken, constraints, fleet,
                   {"aborts": aborts, "replayed": replays, "regenerated": regens})


# == variants of Algorithm 1 ================================================

def recoord(base: Table, pool: Sequence[Proposal], constraints, fleet: Fleet,
            config: CoordinatorConfig | None = None, name: str = "recoord",
            waves: int = 1, **_: Any) -> Report:
    coordinator = Coordinator(base, constraints, fleet, config or CoordinatorConfig())
    if waves <= 1:
        report = coordinator.merge(pool)
    else:
        size = (len(pool) + waves - 1) // max(1, waves)
        for start in range(0, len(pool), max(1, size)):
            report = coordinator.merge(pool[start:start + size])
    report.stats["method"] = name
    report.stats["graph_components"] = len(coordinator.graph.components())
    return report


def _variant(name: str, **flags: Any) -> Callable[..., Report]:
    def run(base, pool, constraints, fleet, config: CoordinatorConfig | None = None, **kw):
        cfg = replace(config or CoordinatorConfig(), **flags)
        return recoord(base, pool, constraints, fleet, cfg, name=name, **kw)
    run.__name__ = name.replace("-", "_")
    return run


# OCC-RV re-derives every reader of a candidate edge; IVM-Regen confirms by
# delta query but regenerates instead of replaying; OCC-Replay re-executes
# every reader's program and re-prompts the others.
occ_rv = _variant("occ-rv", use_replay=False, always_regenerate=True)
ivm_regen = _variant("ivm-regen", use_replay=False)
occ_replay = _variant("occ-replay", use_retests=False)

# Ablations of Algorithm 1.
ablation_no_invalidation = _variant("no-invalidation", use_invalidation=False)
ablation_no_replay = _variant("no-replay", use_replay=False)
ablation_no_dc = _variant("no-dc-test", use_dc_test=False, use_null_patch=False)
ablation_no_null = _variant("no-null-patch", use_null_patch=False)
ablation_no_retest = _variant("no-retests", use_retests=False)
ablation_uniform = _variant("uniform-scores", score_order=False)


# == central references =====================================================

def single_agent(base: Table, pool: Sequence[Proposal], constraints, fleet: Fleet, **_: Any) -> Report:
    """One agent does everything: only the first client's proposals exist."""
    if not pool:
        return _report("single-agent", base, base.copy(), [], constraints, fleet)
    owner = sorted({p.client for p in pool})[0]
    report = last_writer_wins(base, [p for p in pool if p.client == owner], constraints, fleet)
    report.stats["method"] = "single-agent"
    return report


def serial(base: Table, pool: Sequence[Proposal], constraints, fleet: Fleet,
           rounds: int = 1, **_: Any) -> Report:
    """A fixed client order, once or to a fixpoint (at most `rounds` passes)."""
    clients = sorted({p.client for p in pool})
    table, taken = base.copy(), []
    for _pass in range(max(1, rounds)):
        changed = False
        for client in clients:
            for proposal in [p for p in pool if p.client == client]:
                if not current(table, proposal):
                    answers = table.evidence(proposal.queries).answers
                    writes = fleet.replay(proposal, answers) if proposal.replayable else None
                    if not writes:
                        continue
                    proposal = proposal.with_patch(writes, table.evidence(proposal.queries))
                if any(table.value(w.cell) != w.value for w in proposal.writes):
                    table.apply(proposal.writes)
                    taken.append(proposal)
                    changed = True
        if not changed:
            break
    return _report("serial" if rounds <= 1 else "serial-fixpoint", base, table, taken, constraints, fleet)


def debate(base: Table, pool: Sequence[Proposal], constraints, fleet: Fleet,
           rounds: int = 2, **_: Any) -> Report:
    """Per cell, the agents argue and the highest total confidence wins."""
    table, taken = base.copy(), []
    by_cell: dict[Cell, list[Proposal]] = {}
    for proposal in pool:
        for cell in proposal.write_cells:
            by_cell.setdefault(cell, []).append(proposal)
    for cell, candidates in sorted(by_cell.items()):
        tally: dict[Any, float] = {}
        for proposal in candidates:
            value = proposal.patch[cell]
            tally[value] = tally.get(value, 0.0) + proposal.score * rounds
        value = max(sorted(tally), key=lambda v: tally[v])
        table.apply([Write(cell, value)])
        taken.extend(p for p in candidates if p.patch[cell] == value)
    return _report("debate", base, table, list({p.id: p for p in taken}.values()), constraints, fleet)


def sequencing_planner(base: Table, pool: Sequence[Proposal], constraints, fleet: Fleet,
                       config: CoordinatorConfig | None = None, **_: Any) -> Report:
    """An orchestrator runs the clients one at a time, writers first.

    Each step's proposals are admitted by Algorithm 1's tests, so most
    invalidations never arise -- at the price of telling every client when to
    start, and of a planner that would need their private tables to run the
    agents itself.
    """
    cfg = config or CoordinatorConfig()
    coordinator = Coordinator(base, constraints, fleet, cfg)
    order = _plan(pool)
    for client in order:
        coordinator.merge([p for p in pool if p.client == client])
    report = coordinator.finish()
    report.stats["method"] = "sequencing-planner"
    report.stats["plan"] = order
    return report


def _plan(pool: Sequence[Proposal]) -> list[str]:
    """Writers before readers: a client is early if others read what it writes."""
    clients = sorted({p.client for p in pool})
    reads: dict[str, set[str]] = {c: set() for c in clients}
    writes: dict[str, set[str]] = {c: set() for c in clients}
    for proposal in pool:
        writes[proposal.client] |= {c.column for c in proposal.write_cells}
        reads[proposal.client] |= set(proposal.read_columns)
    score = {c: sum(len(writes[c] & reads[o]) for o in clients if o != c) for c in clients}
    return sorted(clients, key=lambda c: (-score[c], c))


MERGES: dict[str, Callable[..., Report]] = {
    "recoord": recoord,
    "lww": last_writer_wins,
    "occ-write": occ_write,
    "ssi": ssi,
    "det-order": det_order,
    "fixpoint-holistic": fixpoint_holistic,
    "trvv": trvv,
    "trvv-nodc": lambda *a, **k: trvv(*a, **{**k, "dc_check": False}),
    "occ-rv": occ_rv,
    "ivm-regen": ivm_regen,
    "occ-replay": occ_replay,
    "single-agent": single_agent,
    "serial": serial,
    "serial-fixpoint": lambda *a, **k: serial(*a, **{**k, "rounds": 5}),
    "debate": debate,
    "sequencing-planner": sequencing_planner,
    "no-invalidation": ablation_no_invalidation,
    "no-replay": ablation_no_replay,
    "no-dc-test": ablation_no_dc,
    "no-null-patch": ablation_no_null,
    "no-retests": ablation_no_retest,
    "uniform-scores": ablation_uniform,
}

ABLATIONS = ["recoord", "no-invalidation", "no-replay", "no-dc-test", "no-null-patch",
             "no-retests", "uniform-scores"]
POLICIES = ["recoord", "lww", "occ-write", "ssi", "det-order", "fixpoint-holistic",
            "trvv", "trvv-nodc", "occ-rv", "ivm-regen", "occ-replay"]
CENTRAL = ["single-agent", "serial", "serial-fixpoint", "debate", "sequencing-planner"]
