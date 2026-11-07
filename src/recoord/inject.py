from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .model import Cell, Value
from .table import Table

Mutator = Callable[[Value, random.Random, dict[str, Any]], Value]
TYPES: dict[str, Mutator] = {}
HELD_OUT = ("U1", "U2", "U3")


def error_type(name: str) -> Callable[[Mutator], Mutator]:
    def wrap(fn: Mutator) -> Mutator:
        TYPES[name] = fn
        return fn
    return wrap


@error_type("typo")
def _typo(value: Value, rng: random.Random, opts: dict) -> Value:
    text = str(value)
    if len(text) < 2:
        return text + "x"
    i = rng.randrange(len(text) - 1)
    return text[:i] + text[i + 1] + text[i] + text[i + 2:]


@error_type("missing")
def _missing(value: Value, rng: random.Random, opts: dict) -> Value:
    return None


@error_type("format")
def _format(value: Value, rng: random.Random, opts: dict) -> Value:
    text = str(value)
    return rng.choice([text.upper(), text.lower(), f" {text} ", text.replace(" ", "_")])


@error_type("swap")
def _swap(value: Value, rng: random.Random, opts: dict) -> Value:
    pool = [v for v in opts.get("column_values", []) if v is not None and v != value]
    return rng.choice(pool) if pool else value


@error_type("dependency")
def _dependency(value: Value, rng: random.Random, opts: dict) -> Value:
    pool = [v for v in opts.get("group_values", []) if v is not None and v != value]
    return rng.choice(pool) if pool else _typo(value, rng, opts)


@error_type("U1")
def _dropped_decimal_comma(value: Value, rng: random.Random, opts: dict) -> Value:
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return value
    return str(int(number / 10)) if number >= 10 else str(number / 10)


@error_type("U2")
def _unit_confusion(value: Value, rng: random.Random, opts: dict) -> Value:
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return value
    factor = rng.choice([0.001, 1000, 6, 1 / 6])
    out = number * factor
    return str(int(out)) if out == int(out) else f"{out:.3f}"


@error_type("U3")
def _other_locale(value: Value, rng: random.Random, opts: dict) -> Value:
    table = opts.get("locale", {})
    return table.get(str(value), str(value) + " (DE)")


@dataclass
class Injection:
    cell: Cell
    kind: str
    before: Value
    after: Value


@dataclass
class ErrorPlan:
    rate: float = 0.12
    types: tuple[str, ...] = ("typo", "missing", "format", "dependency", "swap")
    columns: tuple[str, ...] = ()
    group_key: tuple[str, ...] = ()
    weights: dict[str, float] = field(default_factory=dict)
    column_types: dict[str, tuple[str, ...]] = field(default_factory=dict)
    locale: dict[str, str] = field(default_factory=dict)


def inject(clean: Table, plan: ErrorPlan, seed: int = 0) -> tuple[Table, list[Injection]]:
    rng = random.Random(seed)
    dirty = clean.copy()
    columns = list(plan.columns or clean.columns)
    column_values = {c: [row.get(c) for _, row in clean.rows()] for c in columns}
    group_values: dict[tuple, dict[str, list[Value]]] = {}
    if plan.group_key:
        for rid, row in clean.rows():
            key = tuple(row.get(c) for c in plan.group_key)
            bucket = group_values.setdefault(key, {})
            for c in columns:
                bucket.setdefault(c, []).append(row.get(c))

    cells = [Cell(rid, c) for rid, _ in clean.rows() for c in columns
             if clean.row(rid).get(c) is not None]
    rng.shuffle(cells)
    budget = int(len(cells) * plan.rate)
    log: list[Injection] = []
    for cell in cells[:budget]:
        allowed = plan.column_types.get(cell.column, plan.types)
        allowed = tuple(t for t in allowed if t in plan.types) or plan.types
        kind = _pick(rng, allowed, plan.weights)
        before = clean.value(cell)
        key = tuple(clean.row(cell.row).get(c) for c in plan.group_key) if plan.group_key else ()
        opts = {
            "column_values": column_values[cell.column],
            "group_values": group_values.get(key, {}).get(cell.column, []),
            "locale": plan.locale,
        }
        after = TYPES[kind](before, rng, opts)
        if after == before:
            continue
        dirty.apply([_write(cell, after)])
        log.append(Injection(cell, kind, before, after))
    return dirty, log


def _pick(rng: random.Random, kinds: Sequence[str], weights: dict[str, float]) -> str:
    w = [weights.get(k, 1.0) for k in kinds]
    total = sum(w) or 1.0
    draw = rng.random() * total
    for kind, weight in zip(kinds, w):
        draw -= weight
        if draw <= 0:
            return kind
    return kinds[-1]


def _write(cell: Cell, value: Value):
    from .model import Write
    return Write(cell, value)


def buckets(log: Sequence[Injection]) -> dict[str, list[Cell]]:
    out: dict[str, list[Cell]] = {}
    for item in log:
        out.setdefault(item.kind, []).append(item.cell)
    out["held_out"] = [c for k, cells in out.items() if k in HELD_OUT for c in cells]
    out["seen"] = [c for k, cells in out.items() if k not in HELD_OUT and k not in ("seen", "held_out")
                   for c in cells]
    return out
