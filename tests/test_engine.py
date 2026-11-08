import pytest

from recoord.datasets import load
from recoord.experiments import RunConfig, collect, run

duckdb = pytest.importorskip("duckdb")

from recoord.duck import DuckEngine


def _dataset(name, limit):
    try:
        return load(name, seed=0, limit=limit)
    except FileNotFoundError:
        pytest.skip(f"{name} is not downloaded")


@pytest.mark.parametrize("name,limit", [("demo", None), ("catalog", 120), ("hospital", 120)])
def test_both_engines_answer_every_query_the_same(name, limit):
    dataset = _dataset(name, limit)
    pool, _log, _clients = collect(dataset, RunConfig(agent="offline", dataset=name, limit=limit))
    engine = DuckEngine(dataset.dirty)
    queries = {q.name: q for p in pool for q in p.queries}
    assert queries
    for query in queries.values():
        assert engine.evaluate(query) == (dataset.dirty.answer(query),
                                          dataset.dirty.provenance(query))


@pytest.mark.parametrize("name,limit", [("demo", None), ("catalog", 120)])
def test_both_engines_give_the_same_run(name, limit):
    a = run(RunConfig(agent="offline", dataset=name, limit=limit, engine="python"))
    b = run(RunConfig(agent="offline", dataset=name, limit=limit, engine="duckdb"))
    assert a["output"] == b["output"]
    assert a["selected"] == b["selected"]
    for x, y in zip(a["rows"], b["rows"]):
        assert {k: v for k, v in x.items() if k != "seconds"} == \
               {k: v for k, v in y.items() if k != "seconds"}


def test_the_engine_sees_writes():
    dataset = load("demo")
    table = dataset.dirty
    table.use_engine(DuckEngine)
    pool, _log, _clients = collect(dataset, RunConfig(agent="offline", dataset="demo"))
    query = pool[0].queries[0]
    before = table.answer(query)
    cell = next(iter(table.provenance(query)))
    from recoord.model import Write
    table.apply([Write(cell, "changed-value")])
    assert table.answer(query) != before
