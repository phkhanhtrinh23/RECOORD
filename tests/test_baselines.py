from recoord.baselines import CENTRAL, MERGES, POLICIES
from recoord.coordinator import CoordinatorConfig
from recoord.experiments import RunConfig, collect, fresh_fleet
from recoord.model import Cell


def results(dataset, names, limit=None):
    cfg = RunConfig(dataset=dataset.name, limit=limit)
    pool, log, _ = collect(dataset, cfg)
    out = {}
    for name in names:
        fleet = fresh_fleet(dataset, cfg, log)
        out[name] = MERGES[name](dataset.dirty, pool, dataset.constraints, fleet,
                                 config=CoordinatorConfig())
    return out


def test_every_policy_runs_and_reports(demo):
    for name, report in results(demo, POLICIES + CENTRAL).items():
        assert report.stats["method"]
        assert "violations" in report.stats and "stale" in report.stats


def test_last_writer_wins_keeps_the_stale_derivation(demo):
    report = results(demo, ["lww"])["lww"]
    assert report.table.value(Cell("P101", "total_volume_ml")) == "198"
    assert report.stats["stale"] >= 1


def test_only_recoord_and_its_variants_are_clean(catalog):
    reports = results(catalog, ["recoord", "occ-rv", "ivm-regen", "lww", "trvv-nodc"])
    for name in ("recoord", "occ-rv", "ivm-regen"):
        assert reports[name].stats["stale"] == 0
        assert reports[name].stats["violations"] == 0
    assert reports["lww"].stats["violations"] > 0


def test_the_planner_orders_writers_before_readers(demo):
    report = results(demo, ["sequencing-planner"])["sequencing-planner"]
    assert report.stats["plan"][0] == "A"
