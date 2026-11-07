from recoord.baselines import MERGES
from recoord.checks import check_evidence_needed, check_guarantees, check_order_independence
from recoord.coordinator import Coordinator, CoordinatorConfig
from recoord.constraints import violations
from recoord.experiments import RunConfig, collect, fresh_fleet


def run(catalog, **flags):
    cfg = RunConfig(dataset="catalog", limit=40)
    pool, log, _ = collect(catalog, cfg)
    fleet = fresh_fleet(catalog, cfg, log)
    coordinator = Coordinator(catalog.dirty, catalog.constraints, fleet,
                              CoordinatorConfig(**flags))
    return coordinator, coordinator.merge(pool), pool


def test_output_satisfies_every_constraint(catalog):
    _c, report, _pool = run(catalog)
    assert violations(report.table, catalog.constraints) == []


def test_no_selected_repair_is_stale(catalog):
    _c, report, _pool = run(catalog)
    assert report.stats["stale"] == 0


def test_the_null_patch_avoids_protected_cells(catalog):
    coordinator, report, _pool = run(catalog)
    protected = coordinator.protected(report.selected)
    assert all(w.cell not in protected for w in report.null_patch)


def test_selected_members_do_not_write_the_same_cell_differently(catalog):
    _c, report, _pool = run(catalog)
    for i, a in enumerate(report.selected):
        for b in report.selected[i + 1:]:
            for cell in a.write_cells & b.write_cells:
                assert a.patch[cell] == b.patch[cell]


def test_one_version_per_lineage(catalog):
    _c, report, _pool = run(catalog)
    lineages = [p.lineage for p in report.selected]
    assert len(lineages) == len(set(lineages))


def test_without_the_dc_test_violations_remain(catalog):
    _c, report, _pool = run(catalog, use_dc_test=False, use_null_patch=False)
    assert report.stats["violations"] > 0


def test_replay_saves_calls_at_the_same_quality(catalog):
    cfg = RunConfig(dataset="catalog", limit=40)
    pool, log, _ = collect(catalog, cfg)
    out = {}
    for name in ("recoord", "ivm-regen"):
        fleet = fresh_fleet(catalog, cfg, log)
        report = MERGES[name](catalog.dirty, pool, catalog.constraints, fleet,
                              config=CoordinatorConfig())
        out[name] = (report.stats["utility"], fleet.log.llm_calls)
    assert out["recoord"][1] < out["ivm-regen"][1]
    assert out["recoord"][0] >= out["ivm-regen"][0]


def test_arrival_order_does_not_change_the_output(catalog):
    assert check_order_independence(limit=40, orders=4)["ok"]


def test_random_pools_keep_the_guarantees():
    assert check_guarantees(limit=40, seeds=2)["ok"]


def test_patches_alone_cannot_tell_the_two_pools_apart():
    assert check_evidence_needed(n=3)["ok"]


def test_the_same_seed_gives_the_same_output():
    from recoord.experiments import RunConfig, run
    a = run(RunConfig(dataset="catalog", limit=80, methods=("recoord",)))
    b = run(RunConfig(dataset="catalog", limit=80, methods=("recoord",)))
    assert a["selected"] == b["selected"]
    assert a["output"] == b["output"]
    assert a["rows"][0]["f1"] == b["rows"][0]["f1"]


def test_sharding_returns_the_single_worker_output():
    from recoord.checks import check_sharding
    assert check_sharding(limit=40, shards=(1, 3))["ok"]


def test_a_late_constraint_leaves_no_violation():
    from recoord.constraints import violations
    from recoord.experiments import RunConfig, collect, fresh_fleet
    from recoord.datasets import load
    dataset = load("catalog", seed=0, limit=60)
    cfg = RunConfig(dataset="catalog", limit=60)
    pool, log, _ = collect(dataset, cfg)
    fleet = fresh_fleet(dataset, cfg, log)
    held, rest = dataset.constraints[0], dataset.constraints[1:]
    coordinator = Coordinator(dataset.dirty, rest, fleet, CoordinatorConfig())
    coordinator.merge(pool)
    report = coordinator.add_constraints([held])
    assert violations(report.table, dataset.constraints) == []
    assert report.stats["stale"] == 0


def test_a_joining_client_keeps_the_guarantees():
    from recoord.constraints import violations
    from recoord.experiments import RunConfig, collect, fresh_fleet
    from recoord.datasets import load
    dataset = load("catalog", seed=0, limit=60)
    cfg = RunConfig(dataset="catalog", limit=60)
    pool, log, _ = collect(dataset, cfg)
    early = [p for p in pool if p.client != "gtin-pool"]
    late = [p for p in pool if p.client == "gtin-pool"]
    fleet = fresh_fleet(dataset, cfg, log)
    coordinator = Coordinator(dataset.dirty, dataset.constraints, fleet, CoordinatorConfig())
    coordinator.merge(early)
    before = set(coordinator.selected)
    report = coordinator.merge(late)
    assert before <= set(coordinator.selected)
    assert violations(report.table, dataset.constraints) == []
    assert report.stats["stale"] == 0
