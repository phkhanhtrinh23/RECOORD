from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, Sequence

from .model import Answer, Proposal, Write
from .table import Table, restrict
from . import programs


class Agent(Protocol):
    name: str

    def propose(self, view: Table, client: str) -> list[Proposal]: ...

    def regenerate(self, proposal: Proposal, view: Table, answers: tuple[Answer, ...]) -> Proposal | None: ...

    @property
    def usage(self) -> "Usage": ...


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: "Usage") -> "Usage":
        return Usage(self.calls + other.calls,
                     self.input_tokens + other.input_tokens,
                     self.output_tokens + other.output_tokens)


@dataclass
class CallLog:
    initial_calls: int = 0
    regeneration_calls: int = 0
    replay_calls: int = 0
    failed_calls: int = 0
    retried_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    upload_bytes: int = 0
    call_bytes: int = 0
    patch_bytes: int = 0

    @property
    def llm_calls(self) -> int:
        return self.initial_calls + self.regeneration_calls

    @property
    def rederivation_calls(self) -> int:
        return self.regeneration_calls

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def traffic_bytes(self) -> int:
        return self.upload_bytes + self.call_bytes + self.patch_bytes

    def as_dict(self) -> dict[str, int]:
        out = {k: int(v) for k, v in self.__dict__.items()}
        out.update(llm_calls=self.llm_calls, rederivation_calls=self.rederivation_calls,
                   tokens=self.tokens, traffic_bytes=self.traffic_bytes)
        return out


@dataclass
class Client:
    name: str
    agent: Agent
    table: Table
    private: dict[str, Any] = field(default_factory=dict)
    row_ids: tuple[str, ...] | None = None
    available: bool = True
    fail_rate: float = 0.0
    seed: int = 0
    _rng: random.Random = field(init=False, repr=False, default_factory=random.Random)

    def __post_init__(self) -> None:
        self._rng = random.Random(f"{self.name}:{self.seed}")

    def view(self, master: Table) -> Table:
        return master if self.row_ids is None else restrict(master, self.row_ids)

    def drops_call(self) -> bool:
        return self.fail_rate > 0 and self._rng.random() < self.fail_rate


class Fleet:
    def __init__(self, clients: Sequence[Client], retries: int = 0):
        self.clients = {c.name: c for c in clients}
        self.retries = retries
        self.log = CallLog()

    def collect(self, master: Table) -> list[Proposal]:
        pool: list[Proposal] = []
        for name in sorted(self.clients):
            client = self.clients[name]
            if not client.available:
                continue
            client.table = client.view(master)
            proposals = client.agent.propose(client.table, name)
            for proposal in proposals:
                pool.append(self._upload(proposal))
            self.log.initial_calls += client.agent.usage.calls
            self.log.input_tokens += client.agent.usage.input_tokens
            self.log.output_tokens += client.agent.usage.output_tokens
        return pool

    def _upload(self, proposal: Proposal) -> Proposal:
        self.log.upload_bytes += _size({
            "patch": [[w.cell.row, w.cell.column, w.value] for w in proposal.writes],
            "queries": [q.sql for q in proposal.queries],
            "replayable": proposal.replayable,
            "score": proposal.score,
        })
        return proposal

    def replay(self, proposal: Proposal, answers: tuple[Answer, ...],
               delta: tuple[Answer, ...] | None = None) -> list[Write] | None:
        client = self.clients.get(proposal.client)
        if client is None or not client.available or proposal.program is None:
            return None
        for attempt in range(self.retries + 1):
            self.log.replay_calls += 1
            self.log.call_bytes += _size(delta if delta is not None else answers)
            if client.drops_call():
                self.log.failed_calls += 1
                if attempt < self.retries:
                    self.log.retried_calls += 1
                    continue
                return None
            break
        if not programs.holds(proposal.program, answers, client.private):
            return None
        writes = programs.run(proposal.program, answers, client.private)
        if writes is None:
            return None
        self.log.call_bytes += _size([[w.cell.row, w.cell.column, w.value] for w in writes])
        return writes

    def regenerate(self, proposal: Proposal, master: Table, answers: tuple[Answer, ...]) -> Proposal | None:
        client = self.clients.get(proposal.client)
        if client is None or not client.available:
            return None
        before = Usage(**client.agent.usage.__dict__)
        for attempt in range(self.retries + 1):
            self.log.call_bytes += _size(answers)
            if client.drops_call():
                self.log.failed_calls += 1
                self.log.regeneration_calls += 1
                if attempt < self.retries:
                    self.log.retried_calls += 1
                    continue
                return None
            break
        view = client.view(master)
        version = client.agent.regenerate(proposal, view, answers)
        after = client.agent.usage
        self.log.regeneration_calls += max(1, after.calls - before.calls)
        self.log.input_tokens += after.input_tokens - before.input_tokens
        self.log.output_tokens += after.output_tokens - before.output_tokens
        if version is None:
            return None
        version = replace(version, lineage=proposal.lineage, generation=proposal.generation + 1,
                          client=proposal.client, id=f"{proposal.lineage}#g{proposal.generation + 1}")
        return self._upload(version)

    def broadcast(self, patch: Sequence[Write]) -> None:
        payload = _size([[w.cell.row, w.cell.column, w.value] for w in patch])
        self.log.patch_bytes += payload * max(1, len(self.clients))
        for client in self.clients.values():
            if client.available:
                client.table.apply(w for w in patch if w.cell.row in client.table)

    def freeze(self, name: str) -> None:
        if name in self.clients:
            self.clients[name].available = False


def _size(payload: Any) -> int:
    return len(json.dumps(payload, default=str).encode())
