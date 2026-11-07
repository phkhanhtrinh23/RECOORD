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
        dirty, _log = inject(clean, ErrorPlan(rate=rate), seed=1)
        share = len(error_cells(dirty, clean)) / (len(clean) * len(clean.columns))
        assert abs(share - rate) < 0.02


def test_the_error_rate_counts_every_column():
    clean = build(CatalogSpec(products=40, vendors=4, offers=120, seed=1))
    dirty, _log = inject(clean, ErrorPlan(rate=0.05, columns=("brand", "title", "category")), seed=1)
    share = len(error_cells(dirty, clean)) / (len(clean) * len(clean.columns))
    assert abs(share - 0.05) < 0.02


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


def test_each_spec_carries_its_constraint_set():
    expected = {"hospital": 9, "flights": 4, "beers": 6, "rayyan": 4, "tax": 5, "catalog": 8}
    import yaml
    from recoord.datasets import spec_path
    for name, count in expected.items():
        spec = yaml.safe_load(spec_path(name).read_text())
        assert len(spec["constraints"]) == count, name


def test_each_spec_names_its_client_roles():
    import yaml
    from recoord.datasets import spec_path
    for name in ("hospital", "flights", "beers", "rayyan", "tax"):
        spec = yaml.safe_load(spec_path(name).read_text())
        roles = {c.get("role") for c in spec["clients"]}
        assert roles == {"imputer", "normalizer", "matcher"}, name
    spec = yaml.safe_load(spec_path("catalog").read_text())
    assert {c.get("role") for c in spec["clients"]} == {
        "imputer", "normalizer", "matcher", "deriver"}


def test_the_catalog_splits_are_disjoint_and_sized():
    dev = load("catalog", seed=0, split="dev")
    test = load("catalog", seed=0, split="test")
    assert len(dev.dirty.row_ids) == 1_200
    assert len(test.dirty.row_ids) == 4_800
    assert not set(dev.dirty.row_ids) & set(test.dirty.row_ids)
    assert len({r["vendor"] for _, r in dev.clean.rows()}) == 15
    assert len({r["vendor"] for _, r in test.clean.rows()}) == 18


def test_the_catalog_corrupts_the_configured_share_of_cells():
    for split in ("dev", "test"):
        dataset = load("catalog", seed=0, split=split)
        cells = len(dataset.dirty.row_ids) * len(dataset.dirty.columns)
        assert dataset.notes["errors"] / cells == pytest.approx(0.12, abs=0.002)


def test_the_development_split_holds_back_the_unseen_types():
    dev = load("catalog", seed=0, split="dev")
    assert not {"U1", "U2", "U3"} & set(dev.buckets)


def test_a_csv_spec_with_a_development_share_splits_in_two():
    try:
        dev = load("beers", split="dev")
        test = load("beers", split="test")
    except FileNotFoundError:
        pytest.skip("beers is not downloaded")
    assert len(dev.dirty.row_ids) == 482
    assert len(test.dirty.row_ids) == 1_928
    assert not set(dev.dirty.row_ids) & set(test.dirty.row_ids)
