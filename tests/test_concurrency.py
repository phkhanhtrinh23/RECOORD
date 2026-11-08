from recoord.experiments import RunConfig, run


def _signature(result):
    return (result["output"], result["selected"],
            [{k: v for k, v in row.items() if k != "seconds"} for row in result["rows"]])


def test_worker_count_does_not_change_the_output():
    base = _signature(run(RunConfig(agent="offline", dataset="catalog", limit=120, concurrency=1)))
    for workers in (2, 4, 16):
        assert _signature(run(RunConfig(agent="offline", dataset="catalog", limit=120,
                                        concurrency=workers))) == base


def test_repeated_runs_agree():
    cfg = RunConfig(agent="offline", dataset="catalog", limit=120, concurrency=16)
    assert _signature(run(cfg)) == _signature(run(cfg))
