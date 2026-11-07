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
        out[name] = (report.stats["selected"], fleet.log.llm_calls)
    assert out["recoord"][0] == out["ivm-regen"][0]
    assert out["recoord"][1] < out["ivm-regen"][1]


def test_arrival_order_does_not_change_the_output(catalog):
    assert check_order_independence(limit=40, orders=4)["ok"]


def test_random_pools_keep_the_guarantees():
    assert check_guarantees(limit=40, seeds=2)["ok"]


def test_patches_alone_cannot_tell_the_two_pools_apart():
    assert check_evidence_needed(n=3)["ok"]
