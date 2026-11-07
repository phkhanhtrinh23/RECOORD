from recoord.model import Atom, Cell, Query, Write
from recoord.table import Table, restrict


def table():
    return Table([{"id": "1", "a": "x", "b": "2"},
                  {"id": "2", "a": "y", "b": ""},
                  {"id": "3", "a": "x", "b": "4"}], "id")


def test_empty_string_is_null():
    assert table().value(Cell("2", "b")) is None


def test_selection_and_projection():
    q = Query("q", (Atom(columns=("a",), selections=(("b", ">", "1"),)),))
    assert table().answer(q) == (("x",), ("x",))


def test_null_never_satisfies_a_predicate():
    q = Query("q", (Atom(columns=("id",), selections=(("b", "!=", "99"),)),))
    assert len(table().answer(q)) == 2


def test_answers_are_sorted_bags():
    q = Query("q", (Atom(columns=("a",)),))
    assert table().answer(q) == (("x",), ("x",), ("y",))


def test_cache_follows_updates():
    t, q = table(), Query("q", (Atom(columns=("a",), rows=("1",)),))
    assert t.answer(q) == (("x",),)
    t.apply([Write(Cell("1", "a"), "z")])
    assert t.answer(q) == (("z",),)


def test_provenance_is_the_cells_read():
    q = Query("q", (Atom(columns=("a",), selections=(("b", "=", "2"),)),))
    assert table().provenance(q) == {Cell("1", "a"), Cell("1", "b")}


def test_join_matches_on_the_join_key():
    q = Query("q", (Atom(columns=("b",), rows=("1",)), Atom(columns=("b",))), join_on=("a",))
    assert table().answer(q) == (("2", "2"), ("2", "4"))


def test_restrict_keeps_the_schema():
    small = restrict(table(), ["1"])
    assert small.row_ids == ["1"] and small.columns == ["a", "b"]
