from __future__ import annotations

from typing import Any, Iterable, Iterator

from .model import Answer, Atom, Cell, Evidence, Query, Value, Write, compare


def norm(value: Value) -> Value:
    if value is None or value == "":
        return None
    return str(value)


class Table:
    __slots__ = ("columns", "id_column", "_rows", "_order", "_cache", "_cached_columns",
                 "_groups", "_by_column", "_engine", "_engine_stale")

    def __init__(self, rows: Iterable[dict[str, Any]], id_column: str = "id", columns: list[str] | None = None):
        self._rows: dict[str, dict[str, Value]] = {}
        self._order: list[str] = []
        self.id_column = id_column
        first = None
        for raw in rows:
            if id_column not in raw:
                raise ValueError(f"row is missing the id column {id_column!r}")
            rid = str(raw[id_column])
            if rid in self._rows:
                raise ValueError(f"duplicate row id {rid!r}")
            first = first or list(raw)
            self._rows[rid] = {k: norm(v) for k, v in raw.items() if k != id_column}
            self._order.append(rid)
        self.columns = list(columns or [c for c in (first or [id_column]) if c != id_column])
        self._engine: Any = None
        self._engine_stale = False
        self._cache: dict[str, tuple[Answer, ...]] = {}
        self._cached_columns: dict[str, frozenset[str]] = {}
        self._groups: dict[tuple[str, ...], dict[tuple[Value, ...], list[str]]] = {}
        self._by_column: dict[str, set[str]] = {}

    def __len__(self) -> int:
        return len(self._order)

    def __contains__(self, row_id: str) -> bool:
        return row_id in self._rows

    @property
    def row_ids(self) -> list[str]:
        return list(self._order)

    def row(self, row_id: str) -> dict[str, Value]:
        return self._rows[row_id]

    def rows(self) -> Iterator[tuple[str, dict[str, Value]]]:
        for rid in self._order:
            yield rid, self._rows[rid]

    def value(self, cell: Cell) -> Value:
        return self._rows[cell.row].get(cell.column)

    def copy(self) -> "Table":
        clone = Table.__new__(Table)
        clone.columns = list(self.columns)
        clone.id_column = self.id_column
        clone._rows = {rid: dict(row) for rid, row in self._rows.items()}
        clone._order = list(self._order)
        clone._cache = dict(self._cache)
        clone._cached_columns = dict(self._cached_columns)
        clone._groups = {k: {key: list(v) for key, v in index.items()}
                         for k, index in self._groups.items()}
        clone._by_column = {k: set(v) for k, v in self._by_column.items()}
        clone._engine = None
        clone._engine_stale = False
        return clone

    def as_records(self) -> list[dict[str, Any]]:
        out = []
        for rid in self._order:
            record = {self.id_column: rid}
            record.update({c: self._rows[rid].get(c) for c in self.columns})
            out.append(record)
        return out

    def apply(self, writes: Iterable[Write]) -> None:
        touched: set[str] = set()
        for write in writes:
            row = self._rows.get(write.cell.row)
            if row is None:
                raise KeyError(f"unknown row {write.cell.row!r}")
            if write.cell.column not in self.columns:
                raise KeyError(f"unknown column {write.cell.column!r}")
            row[write.cell.column] = norm(write.value)
            touched.add(write.cell.column)
        if touched:
            self._engine_stale = self._engine is not None
            self._invalidate(touched)

    def _invalidate(self, columns: set[str]) -> None:
        for column in columns:
            for key in self._by_column.pop(column, ()):
                self._cache.pop(key, None)
                read = self._cached_columns.pop(key, frozenset())
                for other in read - {column}:
                    self._by_column.get(other, set()).discard(key)
        for key in [k for k in self._groups if columns.intersection(k)]:
            self._groups.pop(key, None)

    def groups(self, key_columns: tuple[str, ...]) -> dict[tuple[Value, ...], list[str]]:
        index = self._groups.get(key_columns)
        if index is not None:
            return index
        index = {}
        if not key_columns:
            index[()] = list(self._order)
        else:
            for rid in self._order:
                row = self._rows[rid]
                key = tuple(row.get(c) for c in key_columns)
                if any(k is None for k in key):
                    continue
                index.setdefault(key, []).append(rid)
        self._groups[key_columns] = index
        return index

    def answer(self, query: Query) -> Answer:
        cached = self._cache.get(query.name)
        if cached is not None:
            return cached[0]
        answer, provenance = self._evaluate(query)
        self._cache[query.name] = (answer, provenance)
        self._cached_columns[query.name] = query.read_columns
        for column in query.read_columns:
            self._by_column.setdefault(column, set()).add(query.name)
        return answer

    def provenance(self, query: Query) -> frozenset[Cell]:
        self.answer(query)
        return self._cache[query.name][1]

    def evidence(self, queries: Iterable[Query]) -> Evidence:
        queries = list(queries)
        answers = tuple(self.answer(q) for q in queries)
        cells: set[Cell] = set()
        for q in queries:
            cells |= self.provenance(q)
        return Evidence(answers, frozenset(cells))

    def matches(self, evidence: Evidence, queries: Iterable[Query]) -> bool:
        queries = list(queries)
        if len(evidence.answers) != len(queries):
            return False
        return all(self.answer(q) == a for q, a in zip(queries, evidence.answers))

    def _scan(self, atom: Atom) -> list[tuple[str, dict[str, Value]]]:
        ids = atom.rows if atom.rows is not None else self._order
        out = []
        for rid in ids:
            row = self._rows.get(rid)
            if row is None:
                continue
            if all(compare(row.get(c), op, v) for c, op, v in atom.selections):
                out.append((rid, row))
        return out

    def use_engine(self, factory: Any) -> "Table":
        self._engine = factory(self)
        self._engine_stale = False
        self._cache.clear()
        self._cached_columns.clear()
        self._by_column.clear()
        return self

    def _evaluate(self, query: Query) -> tuple[Answer, frozenset[Cell]]:
        if self._engine is not None:
            if self._engine_stale:
                self._engine.refresh(self)
                self._engine_stale = False
            return self._engine.evaluate(query)
        return self._evaluate_python(query)

    def _evaluate_python(self, query: Query) -> tuple[Answer, frozenset[Cell]]:
        blocks = [self._scan(atom) for atom in query.atoms]
        provenance: set[Cell] = set()

        def cells(rid: str, atom: Atom) -> set[Cell]:
            return {Cell(rid, c) for c in atom.read_columns | set(query.join_on)}

        rows: list[tuple[Value, ...]] = []
        if len(query.atoms) == 1:
            atom = query.atoms[0]
            for rid, row in blocks[0]:
                rows.append(tuple(row.get(c) for c in atom.columns))
                provenance |= cells(rid, atom)
        else:
            left, right = query.atoms[0], query.atoms[1]
            index: dict[tuple[Value, ...], list[tuple[str, dict[str, Value]]]] = {}
            for rid, row in blocks[1]:
                key = tuple(row.get(c) for c in query.join_on)
                if any(k is None for k in key):
                    continue
                index.setdefault(key, []).append((rid, row))
            for lid, lrow in blocks[0]:
                key = tuple(lrow.get(c) for c in query.join_on)
                if any(k is None for k in key):
                    continue
                for rid, rrow in index.get(key, ()):
                    rows.append(tuple(lrow.get(c) for c in left.columns) + tuple(rrow.get(c) for c in right.columns))
                    provenance |= cells(lid, left) | cells(rid, right)
        return tuple(sorted(rows, key=_sort_key)), frozenset(provenance)


def _sort_key(row: tuple[Value, ...]) -> tuple:
    return tuple((v is None, "" if v is None else str(v)) for v in row)


def restrict(table: Table, row_ids: Iterable[str]) -> Table:
    keep = set(row_ids)
    return Table(
        ({table.id_column: rid, **row} for rid, row in table.rows() if rid in keep),
        table.id_column,
        table.columns,
    )
