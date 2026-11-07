from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .constraints import DenialConstraint
from .model import Cell, Proposal, Value
from .table import Table

WILDCARD = object()


@dataclass
class Hypergraph:
    proposals: dict[str, Proposal] = field(default_factory=dict)
    invalidation: dict[str, set[str]] = field(default_factory=dict)
    readers: dict[str, set[str]] = field(default_factory=dict)
    write_edges: dict[Cell, set[str]] = field(default_factory=dict)
    _component: dict[str, str] = field(default_factory=dict)
    _key_buckets: dict[tuple[str, tuple], set[str]] = field(default_factory=dict)
    _wild_buckets: dict[str, set[str]] = field(default_factory=dict)

    def add(self, proposal: Proposal, base: Table, constraints: Sequence[DenialConstraint]) -> None:
        pid = proposal.lineage
        self.proposals[pid] = proposal
        self.invalidation.setdefault(pid, set())
        self.readers.setdefault(pid, set())
        self._component.setdefault(pid, pid)
        for cell in proposal.write_cells:
            peers = self.write_edges.setdefault(cell, set())
            for other in peers:
                self.link(pid, other)
            peers.add(pid)
        written = self._written_columns()
        for other in list(self.proposals.values()):
            if other.lineage == pid:
                continue
            for a, b in ((proposal, other), (other, proposal)):
                if self._candidate(a, b, base, written):
                    self.invalidation[a.lineage].add(b.lineage)
                    self.readers[b.lineage].add(a.lineage)
                    self.link(a.lineage, b.lineage)
        self._link_by_key(proposal, base, constraints)

    def build(self, proposals: Iterable[Proposal], base: Table, constraints: Sequence[DenialConstraint]) -> "Hypergraph":
        for proposal in proposals:
            self.add(proposal, base, constraints)
        return self

    def _written_columns(self) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for cell in self.write_edges:
            out.setdefault(cell.row, set()).add(cell.column)
        return out

    def _candidate(self, writer: Proposal, reader: Proposal, base: Table,
                   written: dict[str, set[str]]) -> bool:
        for cell in writer.write_cells:
            for query in reader.queries:
                for atom in query.atoms:
                    if cell.column not in atom.read_columns | set(query.join_on):
                        continue
                    if atom.rows is not None and cell.row not in atom.rows:
                        continue
                    if cell.row not in base:
                        continue
                    free = written.get(cell.row, set()) | {cell.column}
                    if atom.satisfiable_on(base.row(cell.row), free):
                        return True
        return False

    def reachable_keys(self, proposal: Proposal, base: Table, dc: DenialConstraint) -> set[tuple]:
        touched = {c.row for c in proposal.write_cells if c.column in dc.columns}
        touched |= {c.row for q in proposal.queries for a in q.atoms for c in self._read_cells(q, a, base)
                    if c.column in dc.columns}
        keys: set[tuple] = set()
        for row_id in touched:
            if row_id not in base:
                continue
            per_column: list[list[Value]] = []
            for column in dc.key_columns:
                options: list[Value] = [base.row(row_id).get(column)]
                for writer in self.write_edges.get(Cell(row_id, column), ()):
                    other = self.proposals[writer]
                    options.append(WILDCARD if other.replayable else other.patch[Cell(row_id, column)])
                per_column.append(options)
            keys |= {tuple(k) for k in _product(per_column)} or {()}
        return keys

    @staticmethod
    def _read_cells(query, atom, base: Table) -> set[Cell]:
        ids = atom.rows if atom.rows is not None else base.row_ids
        columns = atom.read_columns | set(query.join_on)
        return {Cell(r, c) for r in ids for c in columns}

    def _link_by_key(self, proposal: Proposal, base: Table,
                     constraints: Sequence[DenialConstraint]) -> None:
        for dc in constraints:
            keys = self.reachable_keys(proposal, base, dc)
            if not keys:
                continue
            wild = self._wild_buckets.setdefault(dc.name, set())
            for other in wild:
                self.link(proposal.lineage, other)
            for key in keys:
                if any(k is WILDCARD for k in key):
                    wild.add(proposal.lineage)
                    for (name, _k), bucket in self._key_buckets.items():
                        if name != dc.name:
                            continue
                        for other in bucket:
                            self.link(proposal.lineage, other)
                    continue
                bucket = self._key_buckets.setdefault((dc.name, key), set())
                for other in bucket:
                    self.link(proposal.lineage, other)
                bucket.add(proposal.lineage)

    def find(self, pid: str) -> str:
        root = self._component.setdefault(pid, pid)
        while root != self._component[root]:
            root = self._component[root]
        self._component[pid] = root
        return root

    def link(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._component[max(ra, rb)] = min(ra, rb)

    def components(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for pid in self.proposals:
            out.setdefault(self.find(pid), []).append(pid)
        return {k: sorted(v) for k, v in out.items()}

    def candidate_readers(self, cells: Iterable[Cell], among: Iterable[str]) -> set[str]:
        among = set(among)
        writers = {w for cell in cells for w in self.write_edges.get(cell, ())}
        out: set[str] = set()
        for writer in writers:
            out |= self.invalidation.get(writer, set()) & among
        return out

    def replay_cycles(self) -> set[str]:
        nodes = [p for p, prop in self.proposals.items() if prop.replayable]
        edges = {n: self.invalidation.get(n, set()) & set(nodes) for n in nodes}
        on_cycle: set[str] = set()
        state: dict[str, int] = {}
        stack: list[str] = []

        def walk(node: str) -> None:
            state[node] = 1
            stack.append(node)
            for nxt in sorted(edges[node]):
                if state.get(nxt, 0) == 0:
                    walk(nxt)
                elif state[nxt] == 1:
                    on_cycle.update(stack[stack.index(nxt):])
            stack.pop()
            state[node] = 2

        for node in sorted(nodes):
            if state.get(node, 0) == 0:
                walk(node)
        return on_cycle

    def order(self, members: Sequence[str]) -> list[str] | None:
        members = list(members)
        head = sorted(m for m in members if not self.proposals[m].replayable)
        tail = sorted(m for m in members if self.proposals[m].replayable)
        order = topological(tail, {m: self.invalidation.get(m, set()) & set(tail) for m in tail})
        return None if order is None else head + order


def topological(nodes: Sequence[str], edges: dict[str, set[str]]) -> list[str] | None:
    indegree = {n: 0 for n in nodes}
    for targets in edges.values():
        for t in targets:
            if t in indegree:
                indegree[t] += 1
    ready = sorted(n for n, d in indegree.items() if d == 0)
    out: list[str] = []
    while ready:
        node = ready.pop(0)
        out.append(node)
        for nxt in sorted(edges.get(node, ())):
            if nxt in indegree:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    ready.append(nxt)
                    ready.sort()
    return out if len(out) == len(nodes) else None


def has_cycle(nodes: Sequence[str], edges: dict[str, set[str]]) -> bool:
    return topological(nodes, edges) is None


def _product(options: list[list[Value]]) -> list[tuple]:
    out: list[tuple] = [()]
    for column in options:
        out = [row + (value,) for row in out for value in column]
    return out


def _keys_meet(left: set[tuple], right: set[tuple]) -> bool:
    for a in left:
        for b in right:
            if len(a) == len(b) and all(x is WILDCARD or y is WILDCARD or x == y for x, y in zip(a, b)):
                return True
    return False
