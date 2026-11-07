from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .constraints import DenialConstraint, plurality_is_wrong
from .model import Cell
from .table import Table


@dataclass
class Quality:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def as_dict(self) -> dict[str, float]:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn,
                "precision": round(self.precision, 6), "recall": round(self.recall, 6),
                "f1": round(self.f1, 6)}


def quality(dirty: Table, repaired: Table, clean: Table,
            columns: Sequence[str] | None = None,
            cells: Iterable[Cell] | None = None) -> Quality:
    out = Quality()
    scope = None if cells is None else set(cells)
    for row_id, truth_row in clean.rows():
        if row_id not in dirty or row_id not in repaired:
            continue
        for column in (columns or clean.columns):
            if scope is not None and Cell(row_id, column) not in scope:
                continue
            truth = truth_row.get(column)
            before = dirty.row(row_id).get(column)
            after = repaired.row(row_id).get(column)
            erroneous = before != truth
            nulled = after is None
            if erroneous and after == truth:
                out.tp += 1
            elif after != before and after != truth and not (erroneous and nulled):
                out.fp += 1
            if erroneous and after != truth:
                out.fn += 1
    return out


def per_cell_quality(dirty: Table, repaired: Table, clean: Table,
                     buckets: dict[str, Iterable[Cell]]) -> dict[str, dict[str, float]]:
    return {name: quality(dirty, repaired, clean, cells=cells).as_dict()
            for name, cells in buckets.items()}


def error_cells(dirty: Table, clean: Table) -> set[Cell]:
    return {Cell(rid, column)
            for rid, row in clean.rows() if rid in dirty
            for column in clean.columns
            if dirty.row(rid).get(column) != row.get(column)}


def interference_rate(report_stats: dict[str, Any]) -> float:
    pool = max(1, int(report_stats.get("pool", 0)))
    moved = int(report_stats.get("replayed", 0)) + int(report_stats.get("regenerated", 0))
    return (moved + int(report_stats.get("dc_rejected", 0))) / pool


def wrong_plurality_share(dirty: Table, clean: Table,
                          constraints: Sequence[DenialConstraint]) -> float:
    return plurality_is_wrong(dirty, clean, constraints)


def summarise(name: str, dataset: str, report, dirty: Table, clean: Table | None,
              constraints: Sequence[DenialConstraint], seconds: float = 0.0,
              buckets: dict[str, Iterable[Cell]] | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": dataset, "method": name, "seconds": round(seconds, 3),
        **{k: v for k, v in report.stats.items() if k != "method"},
        **report.log.as_dict(),
    }
    row["iota"] = round(interference_rate(report.stats), 6)
    if clean is not None:
        row.update(quality(dirty, report.table, clean).as_dict())
        row["omega"] = round(wrong_plurality_share(dirty, clean, constraints), 6)
        for bucket, scores in (per_cell_quality(dirty, report.table, clean, buckets or {})).items():
            row[f"f1_{bucket}"] = scores["f1"]
    return row
