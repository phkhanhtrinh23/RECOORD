from __future__ import annotations

import importlib
import threading
from typing import Any, Iterable

from .model import Answer, Atom, Cell, Query, Value


def _module() -> Any:
    try:
        return importlib.import_module("duckdb")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "the duckdb engine needs the 'duckdb' package: pip install -e '.[sql]'"
        ) from error


def _literal(value: Value) -> str:
    if value is None:
        return "NULL"
    return "'" + str(value).replace("'", "''") + "'"


def _predicate(alias: str, column: str, op: str, value: Value) -> str:
    if value is None:
        return "false"
    left, right = f"{alias}.{_ident(column)}", _literal(value)
    numeric = (f"try_cast({left} AS DOUBLE) IS NOT NULL "
               f"AND try_cast({right} AS DOUBLE) IS NOT NULL")
    as_number = f"try_cast({left} AS DOUBLE) {op} try_cast({right} AS DOUBLE)"
    as_text = f"{left} {op} {right}"
    return (f"coalesce(CASE WHEN {numeric} THEN {as_number} "
            f"ELSE {as_text} END, false)")


def _ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


class DuckEngine:
    def __init__(self, table: Any, relation: str = "data"):
        self.duckdb = _module()
        self.relation = relation
        self.id_column = table.id_column
        self.columns = list(table.columns)
        self.connection = self.duckdb.connect()
        self._lock = threading.Lock()
        self._create(table)

    def _create(self, table: Any) -> None:
        names = [self.id_column] + self.columns
        body = ", ".join(f"{_ident(c)} VARCHAR" for c in names)
        self.connection.execute(f"CREATE TABLE {_ident(self.relation)} ({body})")
        self._fill(table)

    def _fill(self, table: Any) -> None:
        names = [self.id_column] + self.columns
        rows = [[rid] + [row.get(c) for c in self.columns] for rid, row in table.rows()]
        if not rows:
            return
        marks = ", ".join(["?"] * len(names))
        self.connection.executemany(
            f"INSERT INTO {_ident(self.relation)} VALUES ({marks})", rows)

    def refresh(self, table: Any) -> None:
        with self._lock:
            self.connection.execute(f"DELETE FROM {_ident(self.relation)}")
            self._fill(table)

    def _where(self, alias: str, atom: Atom) -> list[str]:
        out = [_predicate(alias, c, op, v) for c, op, v in atom.selections]
        if atom.rows is not None:
            if not atom.rows:
                out.append("false")
            else:
                ids = ", ".join(_literal(r) for r in atom.rows)
                out.append(f"{alias}.{_ident(self.id_column)} IN ({ids})")
        return out

    def sql(self, query: Query) -> str:
        atoms = query.atoms
        selects = [f"t0.{_ident(self.id_column)} AS rid0"]
        if len(atoms) > 1:
            selects.append(f"t1.{_ident(self.id_column)} AS rid1")
        for i, atom in enumerate(atoms):
            selects += [f"t{i}.{_ident(c)} AS v{i}_{j}" for j, c in enumerate(atom.columns)]
        body = f"SELECT {', '.join(selects)} FROM {_ident(self.relation)} t0"
        where = self._where("t0", atoms[0])
        if len(atoms) > 1:
            on = " AND ".join(f"t0.{_ident(c)} = t1.{_ident(c)}" for c in query.join_on)
            body += f" JOIN {_ident(self.relation)} t1 ON {on or 'true'}"
            where += self._where("t1", atoms[1])
        if where:
            body += " WHERE " + " AND ".join(where)
        return body

    def evaluate(self, query: Query) -> tuple[Answer, frozenset[Cell]]:
        atoms = query.atoms
        with self._lock:
            records = self.connection.execute(self.sql(query)).fetchall()
        joined = set(query.join_on)
        offset = 2 if len(atoms) > 1 else 1
        rows: list[tuple[Value, ...]] = []
        provenance: set[Cell] = set()
        for record in records:
            ids = record[:offset]
            rows.append(tuple(record[offset:]))
            for i, atom in enumerate(atoms):
                rid = ids[i]
                provenance |= {Cell(rid, c) for c in atom.read_columns | joined}
        return tuple(sorted(rows, key=_sort_key)), frozenset(provenance)


def _sort_key(row: Iterable[Value]) -> tuple:
    return tuple((v is None, "" if v is None else str(v)) for v in row)
