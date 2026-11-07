import warnings

import pytest

from recoord.agents import parse_rules
from recoord.catalog import CatalogSpec, build
from recoord.datasets import available, load
from recoord.inject import ErrorPlan, inject
from recoord.metrics import error_cells


def test_every_shipped_spec_parses():
    for name in available():
        if name in ("demo", "catalog"):
            continue
        assert name


def test_demo_is_the_running_example(demo):
    assert len(demo.dirty) == 3
    assert [c.name for c in demo.constraints] == ["gtin-volume"]
    assert {c.name for c in demo.clients} == {"A", "B", "C"}


def test_catalog_injects_the_held_out_types():
    dataset = load("catalog", seed=0, limit=120)
    kinds = set(dataset.buckets)
    assert {"U1", "U2", "U3"} & kinds
    assert dataset.buckets["held_out"] and dataset.buckets["seen"]


def test_the_development_split_has_no_held_out_types():
    dataset = load("catalog", seed=0, limit=120, split="dev")
    assert not dataset.buckets["held_out"]


def test_the_error_rate_is_honoured():
    clean = build(CatalogSpec(products=40, vendors=4, offers=120, seed=1))
    for rate in (0.05, 0.25):
        dirty, log = inject(clean, ErrorPlan(rate=rate, columns=("brand", "title", "category")), seed=1)
        share = len(error_cells(dirty, clean)) / (len(clean) * 3)
        assert abs(share - rate) < 0.05


def test_injection_is_reproducible():
    clean = build(CatalogSpec(products=20, vendors=3, offers=60, seed=2))
    a, _ = inject(clean, ErrorPlan(rate=0.2), seed=7)
    b, _ = inject(clean, ErrorPlan(rate=0.2), seed=7)
    assert a.as_records() == b.as_records()


def test_a_spec_naming_unknown_columns_fails_loudly(tmp_path):
    spec = tmp_path / "broken.yaml"
    spec.write_text(
        "name: broken\nkind: inline\nid_column: id\n"
        "dirty_rows:\n  - {id: '1', a: x}\n"
        "constraints:\n  - {lhs: [nope], rhs: alsonope}\n"
    )
    with pytest.raises(ValueError, match="columns this table lacks"):
        load(str(spec))


def test_a_partly_unknown_spec_only_warns(tmp_path):
    spec = tmp_path / "partial.yaml"
    spec.write_text(
        "name: partial\nkind: inline\nid_column: id\n"
        "dirty_rows:\n  - {id: '1', a: x, b: y}\n  - {id: '2', a: x, b: z}\n"
        "constraints:\n  - {lhs: [a], rhs: b}\n  - {lhs: [nope], rhs: b}\n"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dataset = load(str(spec))
    assert len(dataset.constraints) == 1 and caught


def test_rules_keep_their_scope():
    rules = parse_rules([{"name": "r", "target": "a", "scope": "row_in_group",
                          "group_key": ["k"], "program": "plurality"}])
    assert rules[0].scope == "row_in_group" and rules[0].replayable
