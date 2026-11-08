import pytest

from recoord.datasets import load
from recoord.experiments import RunConfig, collect


@pytest.fixture(scope="session")
def demo():
    return load("demo")


@pytest.fixture
def demo_run(demo):
    cfg = RunConfig(agent="offline", dataset="demo")
    pool, log, _ = collect(demo, cfg)
    return demo, cfg, pool, log


@pytest.fixture
def catalog():
    return load("catalog", seed=0, limit=40)
