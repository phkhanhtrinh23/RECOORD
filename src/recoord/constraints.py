from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .model import Cell, Value, Violation, compare
from .table import Table, norm

Term = tuple[str, str]


@dataclass(frozen=True)
class Predicate:
    left: Term
    op: str
    right: Term

    @property
    def is_group_equality(self) -> bool:
        return (
            self.op == "="
            and self.left[0] in ("t1", "t2")
            and self.right[0] in ("t1", "t2")
            and self.left[0] != self.right[0]
            and self.left[1] == self.right[1]
        )

    def columns(self) -> set[str]:
        return {t[1] for t in (self.left, self.right) if t[0] in ("t1", "t2")}


@dataclass(frozen=True)
class DenialConstraint:
    name: str
    predicates: tuple[Predicate, ...]
    kind: str = "dc"
    rhs: str | None = None

    @property
    def key_columns(self) -> tuple[str, ...]:
        return tuple(sorted({p.left[1] for p in self.predicates if p.is_group_equality}))

    @property
    def value_predicates(self) -> tuple[Predicate, ...]:
        return tuple(p for p in self.predicates if not p.is_group_equality)

    @property
    def arity(self) -> int:
        mentions = {t[0] for p in self.predicates for t in (p.left, p.right)}
        return 1 if "t2" not in mentions else 2

    def side_predicates(self, alias: str) -> tuple[Predicate, ...]:
        out = []
        for pred in self.value_predicates:
            aliases = {t[0] for t in (pred.left, pred.right)} - {"const"}
            if aliases == {alias}:
                out.append(pred)
        return tuple(out)

    @property
    def columns(self) -> frozenset[str]:
        return frozenset().union(*(p.columns() for p in self.predicates)) if self.predicates else frozenset()

    @property
    def value_columns(self) -> frozenset[str]:
        value = self.value_predicates
        if not value:
            return self.columns
        return frozenset().union(*(p.columns() for p in value))

    def key_of(self, row: dict[str, Value]) -> tuple[Value, ...]:
        return tuple(row.get(c) for c in self.key_columns)


def parse_constraints(specs: Sequence[dict[str, Any]]) -> list[DenialConstraint]:
    out = []
    for i, spec in enumerate(specs):
        if "fd" in spec or ("lhs" in spec and "rhs" in spec):
            body = spec.get("fd", spec)
            lhs = body["lhs"] if isinstance(body["lhs"], list) else [body["lhs"]]
            rhs = body["rhs"]
            name = spec.get("name") or f"{'+'.join(lhs)}->{rhs}"
            preds = [Predicate(("t1", c), "=", ("t2", c)) for c in lhs]
            preds.append(Predicate(("t1", rhs), "!=", ("t2", rhs)))
            out.append(DenialConstraint(name, tuple(preds), kind="fd", rhs=rhs))
            continue
        preds = tuple(_parse_predicate(p) for p in spec["predicates"])
        out.append(DenialConstraint(spec.get("name") or f"sigma{i + 1}", preds))
    return out


def _parse_predicate(raw: Any) -> Predicate:
    if isinstance(raw, (list, tuple)):
        left, op, right = raw
    else:
        left, op, right = raw["left"], raw["op"], raw["right"]
    return Predicate(_term(left), op, _term(right))


def _term(raw: Any) -> Term:
    if isinstance(raw, str) and raw.startswith(("t1.", "t2.")):
        alias, column = raw.split(".", 1)
        return (alias, column)
    return ("const", raw)


def _resolve(term: Term, t1: dict[str, Value], t2: dict[str, Value]) -> Value:
    if term[0] == "const":
        return norm(term[1])
    return (t1 if term[0] == "t1" else t2).get(term[1])


def _holds(pred: Predicate, t1: dict[str, Value], t2: dict[str, Value]) -> bool:
    return compare(_resolve(pred.left, t1, t2), pred.op, _resolve(pred.right, t1, t2))


def _witness(dc: DenialConstraint, i: str, ri: dict, j: str, rj: dict) -> Violation:
    cells, value_cells = set(), set()
    for pred in dc.predicates:
        for term in (pred.left, pred.right):
            if term[0] == "const":
                continue
            cell = Cell(i if term[0] == "t1" else j, term[1])
            cells.add(cell)
            if not pred.is_group_equality:
                value_cells.add(cell)
    return Violation(dc.name, (i, j), frozenset(value_cells or cells), frozenset(cells))


def groups(table: Table, dc: DenialConstraint) -> dict[tuple[Value, ...], list[str]]:
    return table.groups(dc.key_columns)


def violations(table: Table, constraints: Sequence[DenialConstraint],
               rows: Iterable[str] | None = None) -> list[Violation]:
    return list(iter_violations(table, constraints, rows))


def iter_violations(table: Table, constraints: Sequence[DenialConstraint],
                    rows: Iterable[str] | None = None):
    focus = None if rows is None else set(rows)
    for dc in constraints:
        if dc.arity == 1:
            yield from _unary_witnesses(table, dc, focus)
            continue
        index = groups(table, dc)
        keys = index if focus is None else _keys_of(table, dc, index, focus)
        for key in keys:
            members = index[key]
            if len(members) < 2:
                continue
            if dc.kind == "fd" and dc.rhs is not None:
                yield from _fd_witnesses(table, dc, members, focus)
            else:
                yield from _pair_witnesses(table, dc, members, focus)


def _unary_witnesses(table: Table, dc: DenialConstraint,
                     focus: set[str] | None) -> list[Violation]:
    rows = table.row_ids if focus is None else [r for r in focus if r in table]
    out = []
    for rid in rows:
        row = table.row(rid)
        if all(_holds(p, row, row) for p in dc.predicates):
            cells = frozenset(Cell(rid, t[1]) for p in dc.predicates
                              for t in (p.left, p.right) if t[0] != "const")
            out.append(Violation(dc.name, (rid,), cells, cells))
    return out


def _pair_witnesses(table: Table, dc: DenialConstraint, members: Sequence[str],
                    focus: set[str] | None) -> list[Violation]:
    left = [r for r in members if all(_holds(p, table.row(r), table.row(r))
                                      for p in dc.side_predicates("t1"))]
    right = [r for r in members if all(_holds(p, table.row(r), table.row(r))
                                       for p in dc.side_predicates("t2"))]
    if not left or not right:
        return []
    binary = tuple(p for p in dc.value_predicates
                   if {t[0] for t in (p.left, p.right)} - {"const"} == {"t1", "t2"})
    out = []
    for i in left:
        ri = table.row(i)
        for j in right:
            if i == j:
                continue
            if focus is not None and i not in focus and j not in focus:
                continue
            rj = table.row(j)
            if all(_holds(p, ri, rj) for p in binary):
                out.append(_witness(dc, i, ri, j, rj))
    return out


def _keys_of(table: Table, dc: DenialConstraint, index: dict, focus: set[str]) -> list[tuple]:
    seen: set[tuple] = set()
    for rid in focus:
        if rid in table:
            key = dc.key_of(table.row(rid))
            if key in index:
                seen.add(key)
    return list(seen)


def _fd_witnesses(table: Table, dc: DenialConstraint, members: Sequence[str],
                  focus: set[str] | None) -> list[Violation]:
    by_value: dict[Value, list[str]] = {}
    for rid in members:
        value = table.row(rid).get(dc.rhs)
        if value is not None:
            by_value.setdefault(value, []).append(rid)
    if len(by_value) < 2:
        return []
    out = []
    keys = dc.key_columns
    values = list(by_value)
    for a in range(len(values)):
        for b in range(a + 1, len(values)):
            for i in by_value[values[a]]:
                for j in by_value[values[b]]:
                    if focus is not None and i not in focus and j not in focus:
                        continue
                    value_cells = frozenset((Cell(i, dc.rhs), Cell(j, dc.rhs)))
                    cells = value_cells | frozenset(
                        Cell(r, c) for r in (i, j) for c in keys)
                    out.append(Violation(dc.name, (i, j), value_cells, cells))
    return out


def count_in_group(table: Table, dc: DenialConstraint, row_id: str) -> int:
    key = dc.key_of(table.row(row_id))
    if any(k is None for k in key):
        return 0
    members = groups(table, dc).get(key, [])
    if dc.kind == "fd" and dc.rhs is not None:
        mine = table.row(row_id).get(dc.rhs)
        if mine is None:
            return 0
        return sum(1 for r in members
                   if r != row_id and (v := table.row(r).get(dc.rhs)) is not None and v != mine)
    return sum(1 for v in violations(table, [dc], [row_id]) if row_id in v.rows)


def plurality(table: Table, dc: DenialConstraint, row_id: str) -> Value | None:
    if dc.kind != "fd" or dc.rhs is None:
        return None
    key = dc.key_of(table.row(row_id))
    if any(k is None for k in key):
        return None
    counts = Counter(v for r in groups(table, dc).get(key, [])
                     if (v := table.row(r).get(dc.rhs)) is not None)
    ranked = counts.most_common(2)
    if not ranked or (len(ranked) > 1 and ranked[0][1] == ranked[1][1]):
        return None
    return ranked[0][0]


def plurality_is_wrong(table: Table, truth: Table, constraints: Sequence[DenialConstraint]) -> float:
    total = wrong = 0
    for dc in constraints:
        if dc.kind != "fd" or dc.rhs is None:
            continue
        for members in groups(table, dc).values():
            if len(members) < 2:
                continue
            total += 1
            value = plurality(table, dc, members[0])
            truths = Counter(truth.row(r).get(dc.rhs) for r in members if r in truth)
            if truths and value != truths.most_common(1)[0][0]:
                wrong += 1
    return wrong / total if total else 0.0


def counts_before(table: Table, constraints: Sequence[DenialConstraint],
                  changed: Iterable[Cell]) -> dict[tuple[str, str], int]:
    changed = list(changed)
    out: dict[tuple[str, str], int] = {}
    for dc in constraints:
        for cell in changed:
            if cell.column in dc.columns and cell.row in table:
                out[(dc.name, cell.row)] = count_in_group(table, dc, cell.row)
    return out


def worsens_from(counts: dict[tuple[str, str], int], before_keys: dict[tuple[str, str], tuple],
                 after: Table, constraints: Sequence[DenialConstraint],
                 changed: Iterable[Cell]) -> bool:
    changed = list(changed)
    for dc in constraints:
        for row_id in sorted({c.row for c in changed if c.column in dc.columns}):
            if row_id not in after:
                continue
            after_count = count_in_group(after, dc, row_id)
            if after_count == 0:
                continue
            if dc.kind == "fd" and dc.rhs is not None \
                    and plurality(after, dc, row_id) == after.row(row_id).get(dc.rhs):
                continue
            if after_count >= counts.get((dc.name, row_id), 0):
                return True
            old_key = before_keys.get((dc.name, row_id))
            new_key = dc.key_of(after.row(row_id))
            if old_key is not None and old_key != new_key and not any(k is None for k in new_key):
                members = groups(after, dc).get(new_key, [])
                if any(row_id in v.rows for v in iter_violations(after, [dc], members)):
                    return True
    return False


def keys_before(table: Table, constraints: Sequence[DenialConstraint],
                changed: Iterable[Cell]) -> dict[tuple[str, str], tuple]:
    changed = list(changed)
    return {(dc.name, c.row): dc.key_of(table.row(c.row))
            for dc in constraints for c in changed
            if c.column in dc.columns and c.row in table}


def worsens(before: Table, after: Table, constraints: Sequence[DenialConstraint],
            changed: Iterable[Cell]) -> bool:
    changed = list(changed)
    for dc in constraints:
        touched = {c.row for c in changed if c.column in dc.columns}
        for row_id in sorted(touched):
            after_count = count_in_group(after, dc, row_id)
            if after_count == 0:
                continue
            if dc.kind == "fd" and dc.rhs is not None:
                if plurality(after, dc, row_id) == after.row(row_id).get(dc.rhs):
                    continue
            before_count = count_in_group(before, dc, row_id)
            if after_count >= before_count:
                return True
            if _moved_group_gains(before, after, dc, row_id):
                return True
    return False


def _moved_group_gains(before: Table, after: Table, dc: DenialConstraint, row_id: str) -> bool:
    old_key, new_key = dc.key_of(before.row(row_id)), dc.key_of(after.row(row_id))
    if old_key == new_key or any(k is None for k in new_key):
        return False
    members = groups(after, dc).get(new_key, [])
    gained = [v for v in violations(after, [dc], members) if row_id in v.rows]
    return bool(gained)


def changed_cells(before: Table, after: Table) -> list[Cell]:
    out = []
    for rid, row in after.rows():
        old = before.row(rid) if rid in before else {}
        for column, value in row.items():
            if old.get(column) != value:
                out.append(Cell(rid, column))
    return out


def affected_rows(table: Table, constraints: Sequence[DenialConstraint],
                  changed: Sequence[Cell]) -> set[str]:
    rows = {c.row for c in changed}
    for dc in constraints:
        touched = [c for c in changed if c.column in dc.columns]
        if not touched:
            continue
        keys = {dc.key_of(table.row(c.row)) for c in touched if c.row in table}
        for key, members in groups(table, dc).items():
            if key in keys:
                rows.update(members)
    return rows


def adopts(table: Table, constraints: Sequence[DenialConstraint], protected: frozenset[Cell],
           rows: Iterable[str] | None = None) -> bool:
    focus = None if rows is None else set(rows)
    for dc in constraints:
        if dc.kind == "fd" and dc.rhs is not None and dc.arity == 2:
            if _fd_adopts(table, dc, protected, focus):
                return True
            continue
        if any(v.value_cells <= protected for v in iter_violations(table, [dc], rows)):
            return True
    return False


def _fd_adopts(table: Table, dc: DenialConstraint, protected: frozenset[Cell],
               focus: set[str] | None) -> bool:
    index = groups(table, dc)
    keys = index if focus is None else _keys_of(table, dc, index, focus)
    for key in keys:
        members = index[key]
        if len(members) < 2:
            continue
        seen: set[Value] = set()
        for rid in members:
            if Cell(rid, dc.rhs) not in protected:
                continue
            value = table.row(rid).get(dc.rhs)
            if value is None:
                continue
            seen.add(value)
            if len(seen) > 1:
                return True
    return False
