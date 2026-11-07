import pytest

from recoord import programs
from recoord.model import Cell, Program


def answers(*rows):
    return (tuple(rows),)


TARGET = (Cell("r", "c"),)


def run(kind, rows, **options):
    spec = Program(kind, TARGET, options)
    writes = programs.run(spec, answers(*rows))
    return None if writes is None else writes[0].value


def test_product_derives_a_pack_total():
    assert run("product", [("6", "330")]) == "1980"


def test_locale_quantity_reads_a_decimal_comma():
    assert run("locale_quantity", [("Mineralwasser, 6er-Pack, 0,33 L je Flasche",)]) == "330"


def test_pack_size_reads_a_german_feed():
    assert run("pack_size", [("AquaCo Mineralwasser 6er-Pack",)]) == "6"


def test_plurality_needs_a_unique_winner():
    assert run("plurality", [("a",), ("a",), ("b",)]) == "a"
    assert run("plurality", [("a",), ("b",)]) is None


def test_lookup_reads_a_private_table():
    spec = Program("lookup", TARGET, {"table": "t"})
    private = {"t": {"de": "EUR"}}
    assert programs.run(spec, answers(("DE",)), private)[0].value == "EUR"
    assert programs.holds(spec, answers(("US",)), private) is False


def test_a_program_that_does_not_apply_returns_none():
    assert run("product", [("six", "330")]) is None


def test_precondition_guards_non_numeric_answers():
    assert not programs.holds(Program("product", TARGET), answers(("six",)))


def test_unknown_kind_is_an_error():
    with pytest.raises(KeyError):
        programs.run(Program("nope", TARGET), answers(("1",)))
