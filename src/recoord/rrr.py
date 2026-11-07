from __future__ import annotations

from enum import Enum

from .model import Answer, Proposal
from .table import Table
from . import programs


class Action(str, Enum):
    RETAIN = "retain"
    REPLAY = "replay"
    REGENERATE = "regenerate"


def current(table: Table, proposal: Proposal) -> bool:
    return table.matches(proposal.evidence, proposal.queries)


def can_replay(proposal: Proposal, answers: tuple[Answer, ...], private: dict | None = None) -> bool:
    if not proposal.replayable or proposal.program is None:
        return False
    if not programs.holds(proposal.program, answers, private):
        return False
    writes = programs.run(proposal.program, answers, private)
    return writes is not None and {w.cell for w in writes} == set(proposal.write_cells)


def decide(proposal: Proposal, table: Table, private: dict) -> Action:
    if current(table, proposal):
        return Action.RETAIN
    answers = table.evidence(proposal.queries).answers
    return Action.REPLAY if can_replay(proposal, answers, private) else Action.REGENERATE


def delta(before: tuple[Answer, ...], after: tuple[Answer, ...]) -> tuple[Answer, ...]:
    out = []
    for i, new in enumerate(after):
        old = set(before[i]) if i < len(before) else set()
        out.append(tuple(row for row in new if row not in old))
    return tuple(out)
