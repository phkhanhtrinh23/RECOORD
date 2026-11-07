from dataclasses import replace

from recoord.coordinator import Coordinator, CoordinatorConfig, utility
from recoord.experiments import fresh_fleet
from recoord.model import Cell


def merge(demo, cfg, pool, log, **flags):
    fleet = fresh_fleet(demo, cfg, log)
    config = replace(CoordinatorConfig(), **flags)
    coordinator = Coordinator(demo.dirty, demo.constraints, fleet, config)
    return coordinator, coordinator.merge(pool), fleet


def test_the_pool_is_the_three_proposals_of_the_example(demo_run):
    _demo, _cfg, pool, _log = demo_run
    assert [p.client for p in pool] == ["A", "B", "C"]
    assert [p.score for p in pool] == [0.90, 0.80, 0.60]
    assert [p.replayable for p in pool] == [True, True, False]


def test_the_derivation_is_uploaded_stale(demo_run):
    _demo, _cfg, pool, _log = demo_run
    assert pool[1].patch[Cell("P101", "total_volume_ml")] == "198"


def test_recoord_retains_replays_and_rejects(demo_run):
    demo, cfg, pool, log = demo_run
    _c, report, fleet = merge(demo, cfg, pool, log)
    assert [p.lineage for p in report.selected] == [
        "A:volume-from-description:P101", "B:total-from-unit:P101"]
    assert report.table.value(Cell("P101", "unit_volume_ml")) == "330"
    assert report.table.value(Cell("P101", "total_volume_ml")) == "1980"
    assert report.stats["stale"] == 0 and report.stats["violations"] == 0
    assert report.stats["dc_rejected"] == 1
    assert round(utility(report.selected), 6) == 1.7


def test_the_replay_costs_no_llm_call(demo_run):
    demo, cfg, pool, log = demo_run
    _c, _report, fleet = merge(demo, cfg, pool, log)
    assert fleet.log.replay_calls == 1
    assert fleet.log.regeneration_calls == 0


def test_no_null_patch_is_needed(demo_run):
    demo, cfg, pool, log = demo_run
    _c, report, _fleet = merge(demo, cfg, pool, log)
    assert report.null_patch == []


def test_without_the_currency_check_the_stale_total_survives(demo_run):
    demo, cfg, pool, log = demo_run
    _c, report, _fleet = merge(demo, cfg, pool, log, use_invalidation=False)
    assert report.table.value(Cell("P101", "total_volume_ml")) != "1980"


def test_the_threshold_drops_low_confidence_proposals(demo_run):
    demo, cfg, pool, log = demo_run
    _c, report, _fleet = merge(demo, cfg, pool, log, threshold=0.85)
    assert [p.client for p in report.selected] == ["A"]


def test_a_wave_keeps_earlier_members(demo_run):
    demo, cfg, pool, log = demo_run
    fleet = fresh_fleet(demo, cfg, log)
    coordinator = Coordinator(demo.dirty, demo.constraints, fleet, CoordinatorConfig())
    coordinator.merge(pool[:1])
    first = set(coordinator.selected)
    report = coordinator.merge(pool[1:])
    assert first <= set(coordinator.selected)
    assert report.table.value(Cell("P101", "total_volume_ml")) == "1980"
