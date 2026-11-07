from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from .agents import Rule, build_agent
from .analysis import analyse_components
from .baselines import MERGES, POLICIES
from .calibration import fit, label, report as calibration_report
from .coordinator import CoordinatorConfig
from .datasets import ClientSpec, Dataset, load
from .federation import CallLog, Client, Fleet
from .metrics import summarise
from .model import Proposal


@dataclass
class RunConfig:
    dataset: str = "catalog"
    root: str = "data"
    out: str = "results"
    methods: tuple[str, ...] = tuple(POLICIES)
    agent: str = "offline"
    model: str = "gpt-4o-mini"
    base_url: str | None = None
    seed: int = 0
    limit: int | None = None
    split: str = "test"
    error_rate: float | None = None
    threshold: float = 0.5
    cap: int = 2
    clients: int | None = None
    partition: str | None = None
    row_share: float = 1.0
    fail_rate: float = 0.0
    retries: int = 0
    concurrency: int = 16
    engine: str = "python"
    waves: int = 1
    shards: int = 1
    late_constraint: int | None = None
    joining: str | None = None
    freeze: str | None = None
    recalibrate: bool = False
    analyse: bool = False
    analyse_limit: int = 25
    tag: str = ""

    @property
    def coordinator(self) -> CoordinatorConfig:
        return CoordinatorConfig(threshold=self.threshold, regeneration_cap=self.cap)


def make_clients(dataset: Dataset, cfg: RunConfig) -> list[Client]:
    specs = _team(dataset.clients, cfg.clients)
    if cfg.partition:
        specs = _partitioned_specs(dataset, specs, cfg.partition)
    clients: list[Client] = []
    for i, spec in enumerate(specs):
        rules = dataset.rules_for(spec)
        rows = _rows_for(dataset, spec, cfg, i)
        agent = _agent(cfg, spec, dataset, rules, seed=cfg.seed * 100 + i)
        clients.append(Client(
            name=spec.name, agent=agent, table=dataset.dirty.copy(),
            private=dataset.private if spec.private else {},
            row_ids=rows, fail_rate=cfg.fail_rate, seed=cfg.seed * 100 + i,
        ))
    return clients


def _team(specs: Sequence[ClientSpec], size: int | None) -> list[ClientSpec]:
    specs = list(specs)
    if not size or not specs:
        return specs
    if size <= len(specs):
        return specs[:size]
    out = []
    for i in range(size):
        spec = specs[i % len(specs)]
        round_ = i // len(specs)
        out.append(spec if round_ == 0 else replace(spec, name=f"{spec.name}-{round_}"))
    return out


def _agent(cfg: RunConfig, spec: ClientSpec, dataset: Dataset, rules: Sequence[Rule], seed: int):
    if cfg.agent == "llm":
        from .llm import LLMClient
        return build_agent("llm", spec.name, rules, seed=seed,
                           client=LLMClient(model=cfg.model, base_url=cfg.base_url),
                           private=dataset.private if spec.private else {})
    if cfg.agent == "rules":
        return build_agent("rules", spec.name, rules, seed=seed)
    return build_agent("offline", spec.name, rules, seed=seed, reference=dataset.clean,
                       accuracy=spec.accuracy, jitter=spec.jitter,
                       private=dataset.private if spec.private else {})


def _rows_for(dataset: Dataset, spec: ClientSpec, cfg: RunConfig, index: int) -> tuple[str, ...] | None:
    share = min(spec.rows, cfg.row_share)
    if share >= 1.0:
        return None
    ids = dataset.dirty.row_ids
    step = max(1, int(1 / share))
    return tuple(ids[(index + i * step) % len(ids)] for i in range(int(len(ids) * share)))


def _partitioned_specs(dataset: Dataset, specs: Sequence[ClientSpec], column: str) -> list[ClientSpec]:
    values = sorted({row.get(column) for _, row in dataset.dirty.rows() if row.get(column)})
    rules = tuple(r.name for spec in specs for r in dataset.rules_for(spec))
    return [ClientSpec(name=f"{column}={v}", rules=rules, role="normalizer",
                       accuracy=sum(s.accuracy for s in specs) / max(1, len(specs)))
            for v in values]


def owned_rows(dataset: Dataset, column: str, value: Any) -> tuple[str, ...]:
    return tuple(rid for rid, row in dataset.dirty.rows() if row.get(column) == value)


def collect(dataset: Dataset, cfg: RunConfig) -> tuple[list[Proposal], CallLog, list[Client]]:
    clients = make_clients(dataset, cfg)
    if cfg.partition:
        for client in clients:
            value = client.name.split("=", 1)[1]
            client.row_ids = owned_rows(dataset, cfg.partition, value)
    fleet = Fleet(clients, retries=cfg.retries, concurrency=cfg.concurrency)
    pool = fleet.collect(dataset.dirty)
    pool.sort(key=lambda p: (p.client, p.id))
    return pool, fleet.log, clients


def fresh_fleet(dataset: Dataset, cfg: RunConfig, start: CallLog) -> Fleet:
    clients = make_clients(dataset, cfg)
    if cfg.partition:
        for client in clients:
            client.row_ids = owned_rows(dataset, cfg.partition, client.name.split("=", 1)[1])
    fleet = Fleet(clients, retries=cfg.retries, concurrency=cfg.concurrency)
    fleet.log = CallLog(**{k: v for k, v in start.__dict__.items()})
    if cfg.freeze:
        fleet.freeze(cfg.freeze)
    return fleet


def run(cfg: RunConfig, dataset: Dataset | None = None) -> dict[str, Any]:
    dataset = dataset or load(cfg.dataset, root=cfg.root, seed=cfg.seed, limit=cfg.limit,
                              split=cfg.split, error_rate=cfg.error_rate)
    if cfg.engine == "duckdb":
        from .duck import DuckEngine
        dataset.dirty.use_engine(DuckEngine)
    pool, initial, _clients = collect(dataset, cfg)
    late_pool: list[Proposal] = []
    if cfg.joining:
        late_pool = [p for p in pool if p.client == cfg.joining]
        pool = [p for p in pool if p.client != cfg.joining]
    maps = None
    if cfg.recalibrate and dataset.clean is not None:
        maps = fit(label(pool, dataset.dirty, dataset.clean))

    rows: list[dict[str, Any]] = []
    extras: dict[str, Any] = {}
    for method in cfg.methods:
        if method not in MERGES:
            raise ValueError(f"unknown method {method!r}; have {', '.join(sorted(MERGES))}")
        fleet = fresh_fleet(dataset, cfg, initial)
        config = replace(cfg.coordinator, recalibrate=maps)
        started = time.perf_counter()
        constraints, late = _split_constraints(dataset, cfg)
        report = MERGES[method](dataset.dirty, pool, constraints, fleet, config=config,
                                waves=cfg.waves, shards=cfg.shards, late_constraints=late,
                                joining=late_pool)
        seconds = time.perf_counter() - started
        row = summarise(method, dataset.name, report, dataset.dirty, dataset.clean,
                        dataset.constraints, seconds, dataset.buckets)
        row["seed"] = cfg.seed
        row["agent"] = cfg.agent
        row["tag"] = cfg.tag
        row["pool_size"] = len(pool)
        rows.append(row)
        if method == "recoord":
            extras["output"] = report.table.as_records()
            extras["selected"] = [p.id for p in report.selected]
            extras["null_patch"] = [[w.cell.row, w.cell.column] for w in report.null_patch]
            if dataset.clean is not None:
                samples = label(pool, report.table, dataset.clean)
                extras["calibration"] = {
                    "raw": calibration_report(samples),
                    "recalibrated": calibration_report(samples, maps or fit(samples)),
                }
            if cfg.analyse:
                extras["components"] = [
                    c.as_dict() for c in analyse_components(
                        dataset.dirty, pool, dataset.constraints, fleet,
                        report.hypergraph, report.selected, limit=cfg.analyse_limit)
                ]
    return {"dataset": dataset.name, "config": _config_dict(cfg), "rows": rows, **extras}


def _split_constraints(dataset: Dataset, cfg: RunConfig):
    if cfg.late_constraint is None or not dataset.constraints:
        return dataset.constraints, ()
    index = cfg.late_constraint % len(dataset.constraints)
    kept = [c for i, c in enumerate(dataset.constraints) if i != index]
    return kept, (dataset.constraints[index],)


def _config_dict(cfg: RunConfig) -> dict[str, Any]:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items()}


def write(result: dict[str, Any], out: str | Path, name: str | None = None) -> Path:
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    stem = name or f"{result['dataset']}"
    (folder / f"{stem}.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    write_csv(folder / f"{stem}.csv", result["rows"])
    return folder / f"{stem}.json"


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> Path:
    import csv as _csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = _columns(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})
    return path


def _columns(rows: Sequence[dict[str, Any]]) -> list[str]:
    front = ["dataset", "method", "f1", "precision", "recall", "stale", "violations",
             "null_cells", "selected", "llm_calls", "rederivation_calls", "replay_calls",
             "tokens", "traffic_bytes", "seconds"]
    seen = {k for row in rows for k in row}
    return [c for c in front if c in seen] + sorted(seen - set(front))
