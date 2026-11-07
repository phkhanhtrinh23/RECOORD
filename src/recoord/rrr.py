"""Retain, replay, regenerate (Section 4.2).

When a proposal may rest on stale evidence, it is evaluated on the instance
without its own patch and the first of three actions that restores currency is
taken.  Each action starts from the previous one's delta query, so the first
applicable one is the cheapest for that proposal.
"""

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
    """Is every evidence answer still the one the agent derived its patch on?"""
    return table.matches(proposal.evidence, proposal.queries)


def can_replay(proposal: Proposal, answers: tuple[Answer, ...], private: dict | None = None) -> bool:
    """`rep_r(D')`: replayable, precondition holds, and it writes exactly W(r)."""
    if not proposal.replayable or proposal.program is None:
        return False
    if not programs.holds(proposal.program, answers, private):
        return False
    writes = programs.run(proposal.program, answers, private)
    return writes is not None and {w.cell for w in writes} == set(proposal.write_cells)


def decide(proposal: Proposal, table: Table, private: dict) -> Action:
    """`Omega(r | D')`: the cheapest action that makes `proposal` current.

    This runs at the client, which is the only side holding the program and
    the private tables: the server decides Retain by a delta query, and asks
    the client for the other two.
    """
    if current(table, proposal):
        return Action.RETAIN
    answers = table.evidence(proposal.queries).answers
    return Action.REPLAY if can_replay(proposal, answers, private) else Action.REGENERATE


def delta(before: tuple[Answer, ...], after: tuple[Answer, ...]) -> tuple[Answer, ...]:
    """What a replay call actually ships: the rows that changed."""
    out = []
    for i, new in enumerate(after):
        old = set(before[i]) if i < len(before) else set()
        out.append(tuple(row for row in new if row not in old))
    return tuple(out)
