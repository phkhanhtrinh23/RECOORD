"""Values exchanged between clients and the coordinator.

A client's agent reads its copy of the shared relation through *evidence
queries* and emits a *proposal*: a patch, the queries whose answers it used, an
optional deterministic *program* that recomputes the patch from those answers,
and a confidence score.  The coordinator only ever sees what `Upload` carries.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Sequence

Value = Any
OPS = ("=", "!=", "<", "<=", ">", ">=")


@dataclass(frozen=True, order=True)
class Cell:
    row: str
    column: str

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return f"{self.row}.{self.column}"


@dataclass(frozen=True)
class Write:
    cell: Cell
    value: Value


@dataclass(frozen=True)
class Atom:
    """One relation access of an evidence query.

    `columns` are read (and projected), `selections` are local constant
    predicates, and `rows` optionally pins the access to known row ids.
    """

    columns: tuple[str, ...]
    selections: tuple[tuple[str, str, Value], ...] = ()
    rows: tuple[str, ...] | None = None
    relation: str = "data"

    @property
    def read_columns(self) -> frozenset[str]:
        return frozenset(self.columns) | {c for c, _, _ in self.selections}

    def satisfiable_on(self, values: dict[str, Value], free: Iterable[str]) -> bool:
        """Can the local selections hold for a tuple whose `free` cells vary?"""
        free = set(free)
        for column, op, const in self.selections:
            if column in free:
                continue
            if not compare(values.get(column), op, const):
                return False
        return True


@dataclass(frozen=True)
class Query:
    """A select--project--join query over the shared relation."""

    name: str
    atoms: tuple[Atom, ...]
    join_on: tuple[str, ...] = ()

    @property
    def read_columns(self) -> frozenset[str]:
        return frozenset().union(*(a.read_columns for a in self.atoms)) | set(self.join_on)

    def static_cells(self) -> frozenset[tuple[str, str]] | None:
        """The cells this query reads, when they do not depend on the values.

        True for a query pinned to known rows without constant predicates,
        which is what an agent working tuple by tuple issues.  It lets the
        coordinator take the provenance without evaluating anything.
        """
        cells: set[tuple[str, str]] = set()
        for atom in self.atoms:
            if atom.rows is None or atom.selections:
                return None
            columns = atom.read_columns | set(self.join_on)
            cells |= {(r, c) for r in atom.rows for c in columns}
        return frozenset(cells)

    @property
    def sql(self) -> str:
        """The text a client uploads.  Join attributes count as read."""
        parts = []
        for i, atom in enumerate(self.atoms):
            cols = ", ".join(f"t{i}.{c}" for c in sorted(atom.read_columns | set(self.join_on)))
            where = [f"t{i}.{c} {op} {const!r}" for c, op, const in atom.selections]
            if atom.rows is not None:
                where.append(f"t{i}.id IN ({', '.join(sorted(atom.rows))})")
            tail = f" WHERE {' AND '.join(where)}" if where else ""
            parts.append(f"SELECT {cols} FROM {atom.relation} t{i}{tail}")
        if len(parts) == 1:
            return parts[0]
        on = " AND ".join(f"t0.{c} = t1.{c}" for c in self.join_on)
        return f"{parts[0]} JOIN ({parts[1]}) ON {on}"


Answer = tuple[tuple[Value, ...], ...]


@dataclass(frozen=True)
class Evidence:
    """Query answers a client cached, compared as sorted bags (A3)."""

    answers: tuple[Answer, ...]
    provenance: frozenset[Cell] = frozenset()


@dataclass(frozen=True)
class Program:
    """A deterministic repair program, kept at its client.

    `kind` names a function in `recoord.programs`; `inputs` are the answer
    positions it consumes.  `replayable` is the only part the server learns.
    """

    kind: str
    targets: tuple[Cell, ...]
    options: dict[str, Any] = field(default_factory=dict)
    replayable: bool = True


@dataclass
class Proposal:
    """One version of one repair.  Versions of a repair share a `lineage`."""

    id: str
    client: str
    score: float
    writes: tuple[Write, ...]
    queries: tuple[Query, ...] = ()
    evidence: Evidence = Evidence(())
    program: Program | None = None
    lineage: str = ""
    generation: int = 0
    group: str = ""          # bundle key: one agent call over one DC group
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.lineage = self.lineage or self.id
        self.writes = tuple(self.writes)
        self.queries = tuple(self.queries)
        self._write_cells = frozenset(w.cell for w in self.writes)
        self._patch = {w.cell: w.value for w in self.writes}

    @property
    def replayable(self) -> bool:
        return self.program is not None and self.program.replayable

    @property
    def write_cells(self) -> frozenset[Cell]:
        return self._write_cells

    @property
    def read_columns(self) -> frozenset[str]:
        return frozenset().union(*(q.read_columns for q in self.queries)) if self.queries else frozenset()

    @property
    def patch(self) -> dict[Cell, Value]:
        return self._patch

    def with_patch(self, writes: Sequence[Write], evidence: Evidence) -> "Proposal":
        return replace(self, writes=tuple(writes), evidence=evidence)

    @property
    def tiebreak(self) -> str:
        """Content hash, so the order never depends on arrival (A4)."""
        blob = json.dumps(
            [self.lineage, self.generation, sorted((w.cell.row, w.cell.column, str(w.value)) for w in self.writes)],
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()


@dataclass(frozen=True)
class Violation:
    """A witness on which every predicate of a DC holds."""

    constraint: str
    rows: tuple[str, ...]
    value_cells: frozenset[Cell]
    cells: frozenset[Cell]


def compare(left: Value, op: str, right: Value) -> bool:
    """Three-valued comparison: a predicate on NULL is not true (A5)."""
    if left is None or right is None:
        return False
    try:
        a, b = float(left), float(right)
    except (TypeError, ValueError):
        a, b = str(left), str(right)
    if op == "=":
        return a == b
    if op == "!=":
        return a != b
    if op == "<":
        return a < b
    if op == "<=":
        return a <= b
    if op == ">":
        return a > b
    if op == ">=":
        return a >= b
    raise ValueError(f"unknown operator {op!r}")
