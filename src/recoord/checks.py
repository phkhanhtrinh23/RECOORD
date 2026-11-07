"""Self-checks for the claims that are checkable by enumeration.

1. *Guarantees*: on random small pools the output has no stale repair, no
   constraint violation, no write conflict, and the NULL patch touches no
   protected cell.
2. *Order independence*: with ties broken by a content hash, permuting the
   arrival order returns the same selection, output and number of replay calls.
3. *Patches are not enough*: two pools whose patches, scores and provenance
   cells agree, where one has no valid pair and the other is entirely valid --
   so a rule that sees no queries cannot do better than 1/n on the second.
"""

from __future__ import annotations

import random
from typing import Any

from .coordinator import Coordinator, CoordinatorConfig
from .datasets import load
from .experiments import RunConfig, collect, fresh_fleet
from .federation import Client, Fleet
from .model import Atom, Cell, Program, Proposal, Query, Write
from .table import Table


def run_checks(limit: int = 60, seeds: int = 3) -> dict[str, Any]:
    lines: list[str] = []
    failed = False
    for name, fn in (("guarantees", check_guarantees), ("order", check_order_independence),
                     ("evidence", check_evidence_needed)):
        try:
            detail = fn(limit=limit, seeds=seeds)
            ok = detail.pop("ok")
        except Exception as error:                        # noqa: BLE001 - report, do not crash
            ok, detail = False, {"error": repr(error)}
        failed = failed or not ok
        lines.append(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}")
    return {"failed": failed, "lines": lines}


def check_guarantees(limit: int = 60, seeds: int = 3, **_: Any) -> dict[str, Any]:
    """Theorem 4.2 on random pools of the synthetic catalog."""
    worst = {"stale": 0, "violations": 0, "conflicts": 0, "protected_nulls": 0}
    for seed in range(seeds):
        dataset = load("catalog", seed=seed, limit=limit)
        cfg = RunConfig(dataset="catalog", seed=seed, limit=limit)
        pool, log, _ = collect(dataset, cfg)
        rng = random.Random(seed)
        sample = rng.sample(pool, k=min(len(pool), 40))
        fleet = fresh_fleet(dataset, cfg, log)
        coordinator = Coordinator(dataset.dirty, dataset.constraints, fleet, CoordinatorConfig())
        report = coordinator.merge(sample)
        protected = coordinator.protected(report.selected)
        worst["stale"] = max(worst["stale"], report.stats["stale"])
        worst["violations"] = max(worst["violations"], report.stats["violations"])
        worst["conflicts"] = max(worst["conflicts"], _conflicts(report.selected))
        worst["protected_nulls"] = max(worst["protected_nulls"],
                                       sum(1 for w in report.null_patch if w.cell in protected))
    return {"ok": all(v == 0 for v in worst.values()), **worst}


def _conflicts(members) -> int:
    count = 0
    for i, a in enumerate(members):
        for b in members[i + 1:]:
            shared = a.write_cells & b.write_cells
            if shared and any(a.patch[c] != b.patch[c] for c in shared):
                count += 1
    return count


def check_order_independence(limit: int = 40, orders: int = 8, **_: Any) -> dict[str, Any]:
    """Section 6.6: the same output for every arrival order of the same pool."""
    dataset = load("catalog", seed=0, limit=limit)
    cfg = RunConfig(dataset="catalog", seed=0, limit=limit)
    pool, log, _ = collect(dataset, cfg)
    pool = pool[:14]
    rng = random.Random(0)
    outcomes = set()
    for _ in range(orders):
        shuffled = pool[:]
        rng.shuffle(shuffled)
        fleet = fresh_fleet(dataset, cfg, log)
        report = Coordinator(dataset.dirty, dataset.constraints, fleet,
                             CoordinatorConfig()).merge(shuffled)
        outcomes.add((tuple(sorted(p.lineage for p in report.selected)),
                      _fingerprint(report.table), fleet.log.replay_calls))
    return {"ok": len(outcomes) == 1, "orders": orders, "distinct_outcomes": len(outcomes)}


def _fingerprint(table: Table) -> str:
    return str(hash(tuple(sorted((rid, tuple(sorted(row.items(), key=str))) for rid, row in table.rows()
                                 ))))


def check_evidence_needed(n: int = 3, **_: Any) -> dict[str, Any]:
    """Proposition 4.7: two pools a patch-only rule cannot tell apart."""
    rows = [{"id": "u", "A": "1", "B": "1", "J": "0"}]
    rows += [{"id": f"t{j}", "A": str(j + 10), "B": "0", "J": str(j)} for j in range(1, n + 1)]
    base = Table(rows, "id", ["A", "B", "J"])

    def pool(selective: bool) -> list[Proposal]:
        out = []
        for j in range(1, n + 1):
            selections = (("B", "=", "1"), ("J", "=", "0")) if selective else (("B", "=", "1"), ("J", ">=", "0"))
            query = Query(name=f"r{j}/q0", atoms=(Atom(columns=("A",), selections=selections),))
            out.append(Proposal(
                id=f"r{j}", client="c", score=0.9,
                writes=(Write(Cell(f"t{j}", "B"), "1"),),
                queries=(query,), evidence=base.evidence([query]),
                program=Program("constant", (Cell(f"t{j}", "B"),), {"values": ["1"]}, replayable=False),
            ))
        return out

    results = {}
    for name, selective in (("R1_broad", False), ("R2_selective", True)):
        fleet = Fleet([Client("c", _NullAgent(), base.copy())])
        report = Coordinator(base, [], fleet, CoordinatorConfig()).merge(pool(selective))
        results[name] = len(report.selected)
    # The uploads agree: same patches, same scores, same provenance cell (u, A).
    same_patches = {w.cell for p in pool(True) for w in p.writes} == \
                   {w.cell for p in pool(False) for w in p.writes}
    ok = same_patches and results["R1_broad"] <= 1 and results["R2_selective"] == n
    return {"ok": ok, "selected_on_broad_queries": results["R1_broad"],
            "selected_on_selective_queries": results["R2_selective"], "n": n}


class _NullAgent:
    name = "c"

    def __init__(self) -> None:
        from .federation import Usage
        self._usage = Usage()

    @property
    def usage(self):
        return self._usage

    def propose(self, view, client):
        return []

    def regenerate(self, proposal, view, answers):
        return None
