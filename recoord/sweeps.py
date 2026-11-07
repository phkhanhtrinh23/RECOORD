from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Sequence

from .baselines import ABLATIONS, CENTRAL, POLICIES
from .datasets import load
from .experiments import RunConfig, run

Rows = list[dict[str, Any]]
SWEEPS: dict[str, Callable[..., Rows]] = {}


def sweep(name: str) -> Callable:
    def wrap(fn: Callable[..., Rows]) -> Callable[..., Rows]:
        SWEEPS[name] = fn
        return fn
    return wrap


def _rows(cfg: RunConfig, label: str, value: Any, **extra: Any) -> Rows:
    result = run(replace(cfg, tag=f"{label}={value}"))
    for row in result["rows"]:
        row[label] = value
        row.update(extra)
    return result["rows"]


@sweep("seeds")
def seeds(cfg: RunConfig, runs: int = 5, **_: Any) -> Rows:
    out: Rows = []
    for seed in range(runs):
        out += _rows(replace(cfg, seed=seed), "seed_index", seed)
    return out


@sweep("clients")
def clients(cfg: RunConfig, values: Sequence[int] = (2, 3, 4, 8, 16), **_: Any) -> Rows:
    out: Rows = []
    for n in values:
        out += _rows(replace(cfg, clients=n, methods=("recoord", "occ-rv", "lww")), "clients", n)
    return out


@sweep("size")
def size(cfg: RunConfig, values: Sequence[int] = (250, 500, 1000, 2000), **_: Any) -> Rows:
    out: Rows = []
    for n in values:
        out += _rows(replace(cfg, limit=n, methods=("recoord",)), "rows", n)
    return out


@sweep("errors")
def errors(cfg: RunConfig, values: Sequence[float] = (0.05, 0.10, 0.20, 0.30), **_: Any) -> Rows:
    out: Rows = []
    for rate in values:
        out += _rows(replace(cfg, error_rate=rate, methods=("recoord", "occ-rv", "lww")), "error_rate", rate)
    return out


@sweep("budget")
def budget(cfg: RunConfig, values: Sequence[int] = (0, 1, 2, 4), **_: Any) -> Rows:
    out: Rows = []
    for gamma in values:
        out += _rows(replace(cfg, cap=gamma, methods=("recoord", "occ-replay")), "cap", gamma)
    return out


@sweep("failures")
def failures(cfg: RunConfig, values: Sequence[float] = (0.0, 0.2, 0.4), **_: Any) -> Rows:
    out: Rows = []
    for rate in values:
        for retries in (0, 1):
            out += _rows(replace(cfg, fail_rate=rate, retries=retries, methods=("recoord",)),
                         "fail_rate", rate, retries=retries)
    return out


@sweep("waves")
def waves(cfg: RunConfig, values: Sequence[int] = (1, 2, 5, 10), **_: Any) -> Rows:
    out: Rows = []
    for n in values:
        out += _rows(replace(cfg, waves=n, methods=("recoord", "lww")), "waves", n)
    return out


@sweep("departure")
def departure(cfg: RunConfig, **_: Any) -> Rows:
    dataset = load(cfg.dataset, root=cfg.root, seed=cfg.seed, limit=cfg.limit, split=cfg.split)
    out = _rows(replace(cfg, methods=("recoord",)), "departed", "none")
    for client in dataset.clients:
        out += _rows(replace(cfg, freeze=client.name, methods=("recoord",)), "departed", client.name)
    return out


@sweep("partition")
def partition(cfg: RunConfig, column: str | None = None, **_: Any) -> Rows:
    dataset = load(cfg.dataset, root=cfg.root, seed=cfg.seed, limit=cfg.limit, split=cfg.split)
    column = column or ("vendor" if "vendor" in dataset.dirty.columns else dataset.dirty.columns[0])
    out = _rows(replace(cfg, methods=("recoord",)), "layout", "shared-full-copy")
    out += _rows(replace(cfg, partition=column, methods=("recoord",)), "layout", f"partitioned:{column}")
    out += _rows(replace(cfg, row_share=0.7, methods=("recoord",)), "layout", "partial-copies:0.7")
    return out


@sweep("sharding")
def sharding(cfg: RunConfig, values: Sequence[int] = (1, 2, 4, 8), **_: Any) -> Rows:
    out: Rows = []
    for n in values:
        out += _rows(replace(cfg, shards=n, methods=("recoord",)), "shards", n)
    return out


@sweep("constraints")
def constraints(cfg: RunConfig, values: Sequence[int] = (0, 1, 2), **_: Any) -> Rows:
    out = _rows(replace(cfg, methods=("recoord",)), "late_constraint", "none")
    for index in values:
        out += _rows(replace(cfg, late_constraint=index, methods=("recoord",)),
                     "late_constraint", index)
    return out


@sweep("joining")
def joining(cfg: RunConfig, **_: Any) -> Rows:
    dataset = load(cfg.dataset, root=cfg.root, seed=cfg.seed, limit=cfg.limit, split=cfg.split)
    out = _rows(replace(cfg, methods=("recoord",)), "joined", "from-the-start")
    for client in dataset.clients:
        out += _rows(replace(cfg, joining=client.name, methods=("recoord",)),
                     "joined", client.name)
        out += _rows(replace(cfg, clients=max(1, len(dataset.clients) - 1),
                             methods=("recoord",)), "joined", "absent")
        break
    return out


@sweep("ablation")
def ablation(cfg: RunConfig, **_: Any) -> Rows:
    return _rows(replace(cfg, methods=tuple(ABLATIONS)), "study", "ablation")


@sweep("central")
def central(cfg: RunConfig, **_: Any) -> Rows:
    return _rows(replace(cfg, methods=("recoord", *CENTRAL)), "study", "central")


@sweep("policies")
def policies(cfg: RunConfig, **_: Any) -> Rows:
    return _rows(replace(cfg, methods=tuple(POLICIES)), "study", "policies")


@sweep("calibration")
def calibration(cfg: RunConfig, **_: Any) -> Rows:
    out = _rows(replace(cfg, recalibrate=False, methods=("recoord",)), "recalibrated", False)
    out += _rows(replace(cfg, recalibrate=True, methods=("recoord",)), "recalibrated", True)
    return out


@sweep("agents")
def agents(cfg: RunConfig, kinds: Sequence[str] = ("offline", "rules"), **_: Any) -> Rows:
    out: Rows = []
    for kind in kinds:
        out += _rows(replace(cfg, agent=kind, methods=("recoord",)), "agent_kind", kind)
    return out


@sweep("backbones")
def backbones(cfg: RunConfig, models: Sequence[str] = (
        "gpt-4o-mini", "gpt-4o", "claude-3.5-sonnet",
        "llama-3.1-70b", "qwen2.5-72b"), **_: Any) -> Rows:
    out: Rows = []
    for model in models:
        out += _rows(replace(cfg, agent="llm", model=model, methods=("recoord", "occ-rv")), "model", model)
    return out


def available() -> list[str]:
    return sorted(SWEEPS)
