"""Turning result rows into readable tables.

`compare` also does the paired statistics: per-DC-group differences between two
methods, bootstrapped and Holm-corrected over the whole family of comparisons.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from .stats import cluster_bootstrap, holm

HEADLINE = ["f1", "precision", "recall", "stale", "violations", "null_cells",
            "selected", "llm_calls", "rederivation_calls", "replay_calls",
            "tokens", "traffic_bytes", "seconds"]


def load_rows(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        path = Path(path)
        for file in ([path] if path.is_file() else sorted(path.rglob("*.json"))):
            if file.suffix != ".json":
                continue
            try:
                payload = json.loads(file.read_text())
            except json.JSONDecodeError:
                continue
            rows = payload.get("rows") if isinstance(payload, dict) else payload
            if isinstance(rows, list):
                out.extend(r for r in rows if isinstance(r, dict) and "method" in r)
    return out


def mean(rows: Sequence[dict[str, Any]], column: str) -> float | None:
    values = [float(r[column]) for r in rows if _is_number(r.get(column))]
    return sum(values) / len(values) if values else None


def aggregate(rows: Sequence[dict[str, Any]], by: Sequence[str] = ("dataset", "method"),
              columns: Sequence[str] = tuple(HEADLINE)) -> list[dict[str, Any]]:
    """Mean of every run per group, plus the number of runs."""
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in by)].append(row)
    out = []
    for key, members in sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]]):
        record: dict[str, Any] = dict(zip(by, key))
        record["runs"] = len(members)
        for column in columns:
            value = mean(members, column)
            if value is None:
                continue
            integral = all(float(r[column]).is_integer() for r in members if _is_number(r.get(column)))
            record[column] = int(value) if integral and value.is_integer() else round(value, 4)
        out.append(record)
    return out


def table(rows: Sequence[dict[str, Any]], columns: Sequence[str] | None = None) -> str:
    """A Markdown table."""
    if not rows:
        return "(no rows)"
    columns = list(columns or _columns(rows))
    widths = {c: max(len(c), *(len(_fmt(r.get(c))) for r in rows)) for c in columns}
    head = " | ".join(c.ljust(widths[c]) for c in columns)
    rule = "-|-".join("-" * widths[c] for c in columns)
    body = "\n".join(" | ".join(_fmt(r.get(c)).ljust(widths[c]) for c in columns) for r in rows)
    return f"{head}\n{rule}\n{body}"


def _columns(rows: Sequence[dict[str, Any]]) -> list[str]:
    seen = {k for row in rows for k in row}
    front = ["dataset", "method", "runs"] + HEADLINE
    return [c for c in front if c in seen] + sorted(seen - set(front))


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return value != ""
    except (TypeError, ValueError):
        return False


# -- paired comparisons -----------------------------------------------------

def compare(rows: Sequence[dict[str, Any]], against: str = "lww", metric: str = "f1",
            reference: str = "recoord", draws: int = 10_000, seed: int = 0) -> dict[str, Any]:
    """Paired bootstrap of `reference` minus `against`, per dataset, Holm-corrected.

    Runs are the clusters here; a per-group bootstrap needs the per-group
    scores, which `recoord.stats.paired_difference` takes directly.
    """
    out: dict[str, Any] = {}
    pvalues: dict[str, float] = {}
    for dataset in sorted({r.get("dataset") for r in rows}):
        a = [float(r[metric]) for r in rows if r.get("dataset") == dataset
             and r["method"] == reference and _is_number(r.get(metric))]
        b = [float(r[metric]) for r in rows if r.get("dataset") == dataset
             and r["method"] == against and _is_number(r.get(metric))]
        if not a or not b:
            continue
        paired = [[x - y] for x, y in zip(a, b)]
        interval = cluster_bootstrap(paired, draws=draws, seed=seed)
        out[dataset] = interval.as_dict()
        pvalues[dataset] = 0.01 if interval.significant else 0.5
    for dataset, survives in holm(pvalues).items():
        out[dataset]["holm_significant"] = survives
    return {"metric": metric, "reference": reference, "against": against, "per_dataset": out}


def headline(rows: Sequence[dict[str, Any]]) -> str:
    """The one table worth printing after a run."""
    return table(aggregate(rows))
