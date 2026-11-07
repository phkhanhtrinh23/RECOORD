"""How informative the agents' confidence scores are, and how to fix them.

Utility is the expected number of correct cells only for calibrated scores
(Proposition 4.4), so a server that merges several clients' proposals wants a
monotone per-client recalibration.  We fit one on a labelled sample with
pool-adjacent violators (isotonic regression), which keeps the queue order
inside a client and therefore the guarantees.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from .model import Proposal
from .table import Table


@dataclass
class Scored:
    client: str
    score: float
    correct: bool


def label(proposals: Iterable[Proposal], repaired: Table, clean: Table) -> list[Scored]:
    """A proposal counts as correct if every cell it wrote holds the true value."""
    out = []
    for proposal in proposals:
        cells = [c for c in proposal.write_cells if c.row in clean]
        if not cells:
            continue
        correct = all(repaired.value(c) == clean.value(c) for c in cells)
        out.append(Scored(proposal.client, proposal.score, correct))
    return out


def auroc(samples: Sequence[Scored]) -> float:
    """Can the score tell a correct proposal from a wrong one?"""
    pos = [s.score for s in samples if s.correct]
    neg = [s.score for s in samples if not s.correct]
    if not pos or not neg:
        return 0.5
    ranked = sorted(pos + neg)
    ranks = {}
    i = 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1] == ranked[i]:
            j += 1
        ranks[ranked[i]] = (i + j) / 2 + 1
        i = j + 1
    total = sum(ranks[s] for s in pos)
    return (total - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def ece(samples: Sequence[Scored], bins: int = 10) -> float:
    """Expected calibration error over equal-width score bins."""
    if not samples:
        return 0.0
    buckets: list[list[Scored]] = [[] for _ in range(bins)]
    for s in samples:
        buckets[min(bins - 1, int(s.score * bins))].append(s)
    total = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        mean_score = sum(s.score for s in bucket) / len(bucket)
        accuracy = sum(1 for s in bucket if s.correct) / len(bucket)
        total += len(bucket) / len(samples) * abs(mean_score - accuracy)
    return total


def isotonic(samples: Sequence[Scored]) -> Callable[[float], float]:
    """A monotone score map by pool-adjacent violators."""
    points = sorted(((s.score, 1.0 if s.correct else 0.0) for s in samples), key=lambda p: p[0])
    if not points:
        return lambda x: x
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    weights = [1.0] * len(ys)
    i = 0
    while i < len(ys) - 1:
        if ys[i] <= ys[i + 1]:
            i += 1
            continue
        total = weights[i] + weights[i + 1]
        merged = (ys[i] * weights[i] + ys[i + 1] * weights[i + 1]) / total
        ys[i:i + 2] = [merged]
        weights[i:i + 2] = [total]
        xs[i:i + 2] = [xs[i]]
        i = max(0, i - 1)

    def apply(score: float) -> float:
        index = max(0, bisect_right(xs, score) - 1)
        fitted = min(1.0, max(0.0, ys[index]))
        # Isotonic steps are flat; the raw score breaks ties inside a step, so
        # the map stays strictly monotone and the queue order is preserved.
        return round(min(1.0, fitted + 1e-3 * score), 6)

    return apply


def fit(samples: Sequence[Scored]) -> dict[str, Callable[[float], float]]:
    """One monotone map per client, plus a pooled fallback."""
    by_client: dict[str, list[Scored]] = {}
    for s in samples:
        by_client.setdefault(s.client, []).append(s)
    maps = {client: isotonic(rows) for client, rows in by_client.items() if len(rows) >= 10}
    maps["*"] = isotonic(samples)
    return maps


def report(samples: Sequence[Scored], maps: dict[str, Callable[[float], float]] | None = None) -> dict[str, float]:
    mapped = samples if maps is None else [
        Scored(s.client, (maps.get(s.client) or maps["*"])(s.score), s.correct) for s in samples
    ]
    return {"n": len(samples), "auroc": round(auroc(mapped), 4), "ece": round(ece(mapped), 4),
            "mean_score": round(sum(s.score for s in mapped) / len(mapped), 4) if mapped else 0.0,
            "accuracy": round(sum(1 for s in mapped if s.correct) / len(mapped), 4) if mapped else 0.0}
