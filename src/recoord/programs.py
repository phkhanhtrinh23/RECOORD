"""Deterministic repair programs.

A program lives at its client, reads only its proposal's evidence answers and
its client's private tables, and recomputes the patch.  Registering a kind here
makes it replayable: given fresh answers, the coordinator can ask the client to
re-run it instead of paying for an LLM call.  Programs never call a model.
"""

from __future__ import annotations

import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from typing import Any, Callable

from .model import Answer, Program, Value, Write

Registry = dict[str, Callable[..., list[Write] | None]]
PROGRAMS: Registry = {}
PRECONDITIONS: Registry = {}


def program(name: str) -> Callable:
    def wrap(fn: Callable) -> Callable:
        PROGRAMS[name] = fn
        return fn
    return wrap


def precondition(name: str) -> Callable:
    def wrap(fn: Callable) -> Callable:
        PRECONDITIONS[name] = fn
        return fn
    return wrap


def holds(spec: Program, answers: tuple[Answer, ...], private: dict[str, Any] | None = None) -> bool:
    """`pre_r`: a Boolean test on the evidence answers."""
    test = PRECONDITIONS.get(spec.kind)
    if test is None:
        return bool(answers) and bool(answers[0])
    return bool(test(spec, answers, private or {}))


def run(spec: Program, answers: tuple[Answer, ...], private: dict[str, Any] | None = None) -> list[Write] | None:
    """`P_r(Q_r(D))`, or None when the program does not apply."""
    fn = PROGRAMS.get(spec.kind)
    if fn is None:
        raise KeyError(f"no program registered for {spec.kind!r}")
    try:
        return fn(spec, answers, private or {})
    except (InvalidOperation, ValueError, TypeError, IndexError, KeyError, ZeroDivisionError):
        return None


# -- helpers ---------------------------------------------------------------

def _cells(answers: tuple[Answer, ...]) -> list[Value]:
    """Flatten the answers in the order the client passed them (A3)."""
    return [value for answer in answers for row in answer for value in row]


def _number(value: Value) -> Decimal:
    text = str(value).strip().replace(" ", "")
    if "," in text and "." in text:
        text = text.replace(",", "")
    else:
        text = text.replace(",", ".")
    return Decimal(text)


def _tidy(value: Decimal) -> str:
    return str(int(value)) if value == value.to_integral() else format(value.normalize(), "f")


# -- arithmetic derivations ------------------------------------------------

@precondition("product")
@precondition("sum")
def _all_numeric(spec: Program, answers: tuple[Answer, ...], private: dict) -> bool:
    values = _cells(answers)
    if not values:
        return False
    try:
        [_number(v) for v in values]
    except (InvalidOperation, ValueError, TypeError):
        return False
    return True


@program("product")
def _product(spec: Program, answers: tuple[Answer, ...], private: dict) -> list[Write] | None:
    values = [_number(v) for v in _cells(answers)]
    total = Decimal(1)
    for value in values:
        total *= value
    return [Write(spec.targets[0], _tidy(total))]


@program("sum")
def _sum(spec: Program, answers: tuple[Answer, ...], private: dict) -> list[Write] | None:
    return [Write(spec.targets[0], _tidy(sum((_number(v) for v in _cells(answers)), Decimal(0))))]


@program("copy")
def _copy(spec: Program, answers: tuple[Answer, ...], private: dict) -> list[Write] | None:
    values = _cells(answers)
    return [Write(target, values[0]) for target in spec.targets]


@program("concat")
def _concat(spec: Program, answers: tuple[Answer, ...], private: dict) -> list[Write] | None:
    sep = spec.options.get("separator", " ")
    return [Write(spec.targets[0], sep.join(str(v) for v in _cells(answers) if v is not None))]


# -- normalisation ---------------------------------------------------------

UNITS = {"ml": Decimal(1), "cl": Decimal(10), "l": Decimal(1000), "g": Decimal(1), "kg": Decimal(1000)}
_QUANTITY = re.compile(r"(\d+(?:[.,]\d+)?)\s*(ml|cl|l|kg|g)\b", re.IGNORECASE)


@program("locale_quantity")
def _locale_quantity(spec: Program, answers: tuple[Answer, ...], private: dict) -> list[Write] | None:
    """Read a quantity written in another locale: "0,33 L je Flasche" -> 330."""
    text = " ".join(str(v) for v in _cells(answers) if v is not None)
    matches = _QUANTITY.findall(text.lower())
    if not matches:
        return None
    number, unit = matches[-1]
    unit_to = spec.options.get("unit", "ml").lower()
    value = _number(number) * UNITS[unit.lower()] / UNITS[unit_to]
    return [Write(spec.targets[0], _tidy(value))]


@precondition("locale_quantity")
def _has_quantity(spec: Program, answers: tuple[Answer, ...], private: dict) -> bool:
    text = " ".join(str(v) for v in _cells(answers) if v is not None)
    return bool(_QUANTITY.search(text.lower()))


@program("pack_size")
def _pack_size(spec: Program, answers: tuple[Answer, ...], private: dict) -> list[Write] | None:
    """Read a pack count: "6 x 330 ml" -> 6."""
    text = " ".join(str(v) for v in _cells(answers) if v is not None).lower()
    match = re.search(r"(\d+)\s*(?:x|er[- ]pack|-pack|pack)", text)
    return [Write(spec.targets[0], match.group(1))] if match else None


@program("normalise_case")
def _normalise_case(spec: Program, answers: tuple[Answer, ...], private: dict) -> list[Write] | None:
    values = _cells(answers)
    style = spec.options.get("style", "lower")
    text = str(values[0])
    out = {"lower": text.lower, "upper": text.upper, "title": text.title}[style]()
    return [Write(spec.targets[0], out.strip())]


_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")


@program("number")
def _number_only(spec: Program, answers: tuple[Answer, ...], private: dict) -> list[Write] | None:
    """Read the number out of a unit-laden field: "12.0 oz." -> 12, "0.05%" -> 0.05."""
    text = " ".join(str(v) for v in _cells(answers) if v is not None)
    match = _NUMBER.search(text)
    if not match:
        return None
    value = _number(match.group(0))
    if spec.options.get("integer"):
        value = value.to_integral_value()
    return [Write(spec.targets[0], _tidy(value))]


@precondition("number")
def _has_number(spec: Program, answers: tuple[Answer, ...], private: dict) -> bool:
    text = " ".join(str(v) for v in _cells(answers) if v is not None)
    return bool(_NUMBER.search(text))


@program("drop_suffix")
def _drop_suffix(spec: Program, answers: tuple[Answer, ...], private: dict) -> list[Write] | None:
    """Cut a trailing part that belongs in another column: "Madison WI" -> "Madison"."""
    values = _cells(answers)
    if not values or values[0] is None:
        return None
    pattern = spec.options.get("pattern", r"\s+[A-Z]{2}$")
    out = re.sub(pattern, "", str(values[0])).strip()
    return [Write(spec.targets[0], out)] if out else None


@precondition("drop_suffix")
def _has_suffix(spec: Program, answers: tuple[Answer, ...], private: dict) -> bool:
    values = _cells(answers)
    pattern = spec.options.get("pattern", r"\s+[A-Z]{2}$")
    return bool(values) and values[0] is not None and bool(re.search(pattern, str(values[0])))


@program("blank")
def _blank(spec: Program, answers: tuple[Answer, ...], private: dict) -> list[Write] | None:
    """Turn a placeholder for a missing value into a real NULL."""
    markers = {str(m).lower() for m in spec.options.get("markers", ["n/a", "na", "null", "-", "?"])}
    values = _cells(answers)
    if not values or values[0] is None or str(values[0]).strip().lower() not in markers:
        return None
    return [Write(spec.targets[0], None)]


@precondition("blank")
def _is_blank_marker(spec: Program, answers: tuple[Answer, ...], private: dict) -> bool:
    return _blank(spec, answers, private) is not None


@program("lookup")
def _lookup(spec: Program, answers: tuple[Answer, ...], private: dict) -> list[Write] | None:
    """Translate a value through a private table the client never uploads (A2)."""
    table = private.get(spec.options.get("table", ""), {})
    key = str(_cells(answers)[0]).strip().lower()
    value = table.get(key)
    return None if value is None else [Write(spec.targets[0], value)]


@precondition("lookup")
def _in_lookup(spec: Program, answers: tuple[Answer, ...], private: dict) -> bool:
    table = private.get(spec.options.get("table", ""), {})
    values = _cells(answers)
    return bool(values) and str(values[0]).strip().lower() in table


@program("plurality")
def _plurality(spec: Program, answers: tuple[Answer, ...], private: dict) -> list[Write] | None:
    """The majority value of the answer, which a group repair reads."""
    counts = Counter(v for v in _cells(answers) if v is not None)
    ranked = counts.most_common(2)
    if not ranked or (len(ranked) > 1 and ranked[0][1] == ranked[1][1]):
        return None
    return [Write(target, ranked[0][0]) for target in spec.targets]


@precondition("plurality")
def _has_plurality(spec: Program, answers: tuple[Answer, ...], private: dict) -> bool:
    return _plurality(spec, answers, private) is not None


@program("constant")
def _constant(spec: Program, answers: tuple[Answer, ...], private: dict) -> list[Write] | None:
    """An LLM judgment frozen into a value: present, but not replayable."""
    values = spec.options.get("values", [])
    return [Write(t, v) for t, v in zip(spec.targets, values)]
