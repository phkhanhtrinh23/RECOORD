from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass
class Interval:
    estimate: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.estimate:+.3f} [{self.low:+.3f}, {self.high:+.3f}]"

    @property
    def significant(self) -> bool:
        return self.low > 0 or self.high < 0

    def as_dict(self) -> dict[str, float]:
        return {"estimate": round(self.estimate, 6), "low": round(self.low, 6),
                "high": round(self.high, 6)}


def cluster_bootstrap(clusters: Sequence[Sequence[float]], statistic: Callable[[Sequence[float]], float] | None = None,
                      draws: int = 10_000, alpha: float = 0.05, seed: int = 0) -> Interval:
    statistic = statistic or _mean
    flat = [v for cluster in clusters for v in cluster]
    if not flat:
        return Interval(0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(clusters)
    samples = []
    for _ in range(draws):
        picked = [clusters[rng.randrange(n)] for _ in range(n)]
        values = [v for cluster in picked for v in cluster]
        if values:
            samples.append(statistic(values))
    samples.sort()
    lo = samples[int((alpha / 2) * len(samples))]
    hi = samples[min(len(samples) - 1, int((1 - alpha / 2) * len(samples)))]
    return Interval(statistic(flat), lo, hi)


def paired_difference(a: dict[str, Sequence[float]], b: dict[str, Sequence[float]],
                      draws: int = 10_000, seed: int = 0) -> Interval:
    keys = sorted(set(a) & set(b))
    clusters = [[x - y for x, y in zip(a[k], b[k])] for k in keys]
    return cluster_bootstrap(clusters, draws=draws, seed=seed)


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    order = sorted(pvalues, key=lambda k: pvalues[k])
    m = len(order)
    out, rejected = {}, True
    for i, key in enumerate(order):
        rejected = rejected and pvalues[key] <= alpha / (m - i)
        out[key] = rejected
    return out


def bootstrap_pvalue(interval_samples: Sequence[float]) -> float:
    if not interval_samples:
        return 1.0
    below = sum(1 for v in interval_samples if v <= 0) / len(interval_samples)
    return min(1.0, 2 * min(below, 1 - below))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
