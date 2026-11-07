from recoord.constraints import (adopts, affected_rows, changed_cells, parse_constraints,
                                 plurality, violations, worsens)
from recoord.model import Cell, Write
from recoord.table import Table


def table(values):
    return Table([{"id": str(i), "k": k, "v": v} for i, (k, v) in enumerate(values)], "id")


FD = parse_constraints([{"name": "k-v", "lhs": ["k"], "rhs": "v"}])
DC = parse_constraints([{"name": "s", "predicates": [["t1.k", "=", "t2.k"], ["t1.v", "<", "t2.v"]]}])


def test_fd_expands_to_a_denial_constraint():
    assert FD[0].key_columns == ("k",) and FD[0].value_columns == frozenset({"v"})


def test_fd_violation_is_found_once():
    assert len(violations(table([("a", "1"), ("a", "2")]), FD)) == 1


def test_null_breaks_no_constraint():
    assert violations(table([("a", "1"), ("a", None)]), FD) == []


def test_inequality_predicate():
    assert len(violations(table([("a", "1"), ("a", "2")]), DC)) == 1


def test_value_cells_exclude_the_group_key():
    v = violations(table([("a", "1"), ("a", "2")]), FD)[0]
    assert v.value_cells == frozenset({Cell("0", "v"), Cell("1", "v")})


def test_plurality_needs_to_be_unique():
    assert plurality(table([("a", "1"), ("a", "1"), ("a", "2")]), FD[0], "0") == "1"
    assert plurality(table([("a", "1"), ("a", "2")]), FD[0], "0") is None


def test_moving_to_the_plurality_is_harmless():
    before = table([("a", "1"), ("a", "1"), ("a", "9")])
    after = before.copy()
    after.apply([Write(Cell("2", "v"), "1")])
    assert not worsens(before, after, FD, [Cell("2", "v")])


def test_a_wrong_minority_value_worsens():
    before = table([("a", "1"), ("a", "1"), ("a", "1")])
    after = before.copy()
    after.apply([Write(Cell("2", "v"), "7")])
    assert worsens(before, after, FD, [Cell("2", "v")])


def test_affected_rows_covers_the_whole_group():
    t = table([("a", "1"), ("a", "2"), ("b", "3")])
    assert affected_rows(t, FD, [Cell("0", "v")]) == {"0", "1"}


def test_adoption_needs_every_value_cell_protected():
    t = table([("a", "1"), ("a", "2")])
    cells = frozenset({Cell("0", "v"), Cell("1", "v")})
    assert adopts(t, FD, cells)
    assert not adopts(t, FD, frozenset({Cell("0", "v")}))


def test_changed_cells():
    before = table([("a", "1")])
    after = before.copy()
    after.apply([Write(Cell("0", "v"), "2")])
    assert changed_cells(before, after) == [Cell("0", "v")]
