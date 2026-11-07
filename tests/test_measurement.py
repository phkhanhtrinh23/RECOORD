from recoord.calibration import Scored, auroc, ece, fit, isotonic
from recoord.metrics import error_cells, quality
from recoord.model import Cell, Write
from recoord.report import aggregate, compare, table
from recoord.stats import cluster_bootstrap, holm, paired_difference
from recoord.table import Table


def tables():
    clean = Table([{"id": "1", "a": "x"}, {"id": "2", "a": "y"}], "id")
    dirty = Table([{"id": "1", "a": "WRONG"}, {"id": "2", "a": "y"}], "id")
    return dirty, clean


def test_a_fixed_error_is_a_true_positive():
    dirty, clean = tables()
    fixed = dirty.copy()
    fixed.apply([Write(Cell("1", "a"), "x")])
    assert quality(dirty, fixed, clean).as_dict()["f1"] == 1.0


def test_a_null_counts_as_unrepaired_and_as_a_wrong_repair():
    dirty, clean = tables()
    nulled = dirty.copy()
    nulled.apply([Write(Cell("1", "a"), None), Write(Cell("2", "a"), None)])
    scores = quality(dirty, nulled, clean)
    assert scores.tp == 0 and scores.fn == 1 and scores.fp == 1


def test_error_cells_finds_the_injected_cell():
    dirty, clean = tables()
    assert error_cells(dirty, clean) == {Cell("1", "a")}


def test_auroc_and_ece_on_a_perfect_scorer():
    samples = [Scored("c", 0.9, True), Scored("c", 0.1, False)]
    assert auroc(samples) == 1.0
    assert ece(samples) < 0.2


def test_isotonic_is_monotone():
    samples = [Scored("c", s / 10, s > 5) for s in range(10)]
    fitted = isotonic(samples)
    values = [fitted(s / 10) for s in range(10)]
    assert values == sorted(values)


def test_fit_gives_a_map_per_client_and_a_fallback():
    samples = [Scored("a", 0.5, True)] * 12 + [Scored("b", 0.5, False)]
    maps = fit(samples)
    assert "a" in maps and "*" in maps and "b" not in maps


def test_bootstrap_interval_brackets_the_mean():
    interval = cluster_bootstrap([[1.0], [1.0], [1.0]], draws=200)
    assert interval.low <= interval.estimate <= interval.high


def test_paired_difference_is_signed():
    a = {"g1": [1.0], "g2": [1.0]}
    b = {"g1": [0.5], "g2": [0.5]}
    assert paired_difference(a, b, draws=200).estimate == 0.5


def test_holm_rejects_from_the_smallest_p():
    assert holm({"a": 0.001, "b": 0.5}) == {"a": True, "b": False}


def test_aggregate_and_table_round_trip():
    rows = [{"dataset": "d", "method": "m", "f1": 0.5}, {"dataset": "d", "method": "m", "f1": 0.7}]
    out = aggregate(rows)
    assert out[0]["runs"] == 2 and out[0]["f1"] == 0.6
    assert "method" in table(out)


def test_compare_reports_a_signed_interval():
    rows = [{"dataset": "d", "method": "recoord", "f1": 0.8},
            {"dataset": "d", "method": "lww", "f1": 0.5}]
    out = compare(rows, draws=200)
    assert out["per_dataset"]["d"]["estimate"] == 0.3
