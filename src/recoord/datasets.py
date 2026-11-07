"""Dataset specifications: where the data comes from, its DCs, and its clients.

A spec is a YAML file (see `recoord/specs/`).  `kind: csv` points at a dirty and
a clean CSV -- the layout of the Raha/Baran benchmarks, which
`scripts/get_data.sh` fetches.  `kind: catalog` builds the synthetic
multi-vendor catalog and injects errors, so a run needs no download.

Constraints and rules naming absent columns are dropped, so one spec survives a
benchmark whose schema differs a little.
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import yaml

from .agents import Rule, parse_rules
from .constraints import DenialConstraint, parse_constraints
from .catalog import CatalogSpec, build, locale_map, private_tables
from .inject import ErrorPlan, Injection, buckets, inject
from .model import Cell
from .table import Table

SPECS = Path(__file__).parent / "specs"


@dataclass
class ClientSpec:
    name: str
    rules: tuple[str, ...]
    role: str = "normalizer"
    accuracy: float = 0.78
    rows: float = 1.0            # share of the rows this client holds
    private: bool = False
    jitter: float = 0.12         # spread of the simulated confidence scores


@dataclass
class Dataset:
    name: str
    dirty: Table
    clean: Table | None
    constraints: list[DenialConstraint]
    rules: list[Rule]
    clients: list[ClientSpec]
    buckets: dict[str, list[Cell]] = field(default_factory=dict)
    private: dict[str, Any] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)

    def rules_for(self, client: ClientSpec) -> list[Rule]:
        wanted = set(client.rules)
        return [r for r in self.rules if not wanted or r.name in wanted]


def available() -> list[str]:
    return sorted(p.stem for p in SPECS.glob("*.yaml"))


def spec_path(name: str) -> Path:
    path = Path(name)
    if path.exists() and path.suffix in (".yaml", ".yml"):
        return path
    candidate = SPECS / f"{name}.yaml"
    if not candidate.exists():
        raise FileNotFoundError(f"no dataset spec for {name!r}; have {', '.join(available())}")
    return candidate


def load(name: str, *, root: str | Path = "data", seed: int = 0,
         limit: int | None = None, split: str = "test",
         error_rate: float | None = None, held_out: bool = True) -> Dataset:
    path = spec_path(name)
    spec = yaml.safe_load(path.read_text())
    kind = spec.get("kind", "csv")
    if kind == "csv":
        dirty, clean = _load_csv(spec, Path(root), limit)
        log: list[Injection] = []
    elif kind == "catalog":
        dirty, clean, log = _load_catalog(spec, seed, limit, split, error_rate, held_out)
    elif kind == "inline":
        dirty, clean, log = _load_inline(spec), _inline_clean(spec), []
    else:
        raise ValueError(f"unknown dataset kind {kind!r}")

    columns = set(dirty.columns)
    constraints = [c for c in parse_constraints(spec.get("constraints", []))
                   if c.columns <= columns]
    _warn_dropped("constraint", spec.get("constraints", []), constraints, columns)
    rules = [r for r in parse_rules(spec.get("rules", []))
             if {r.target, *r.sources, *r.group_key} <= columns]
    _warn_dropped("rule", spec.get("rules", []), rules, columns)
    clients = [ClientSpec(
        name=c["name"], rules=tuple(c.get("rules", ())), role=c.get("role", "normalizer"),
        accuracy=float(c.get("accuracy", 0.78)), rows=float(c.get("rows", 1.0)),
        private=bool(c.get("private", False)), jitter=float(c.get("jitter", 0.12)),
    ) for c in spec.get("clients", [])]

    return Dataset(
        name=spec.get("name", path.stem), dirty=dirty, clean=clean, constraints=constraints,
        rules=rules, clients=clients, buckets=buckets(log) if log else {},
        private=private_tables(seed) if kind == "catalog" else {},
        notes={"kind": kind, "split": split, "rows": len(dirty), "errors": len(log)},
    )


def _warn_dropped(what: str, raw: Sequence[Any], kept: Sequence[Any],
                  columns: set[str]) -> None:
    """A spec may name a column the table lacks; losing all of them is a typo."""
    if len(kept) == len(raw):
        return
    message = (f"the spec's {what}s name columns this table lacks "
               f"({len(raw) - len(kept)} of {len(raw)} dropped); its columns are "
               f"{', '.join(sorted(columns))}")
    if raw and not kept:
        raise ValueError(message)
    warnings.warn(message, stacklevel=3)


def _read_csv(path: Path, id_column: str) -> Table:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} is empty")
    if id_column not in rows[0]:
        rows = [{id_column: str(i), **row} for i, row in enumerate(rows)]
    columns = [c for c in rows[0] if c != id_column]
    return Table(rows, id_column, columns)


def _canonical(name: str) -> str:
    """A column name without its spelling: "ProviderNumber" -> "providernumber"."""
    return "".join(c for c in name.lower() if c.isalnum())


def _align(dirty: Table, clean: Table) -> Table:
    """Give the clean table the dirty table's column names.

    Published benchmarks do not always spell the two files' headers the same
    way; where the names cannot be matched, the order is all there is to go on.
    """
    if clean.columns == dirty.columns:
        return clean
    by_canonical = {_canonical(c): c for c in dirty.columns}
    if {_canonical(c) for c in clean.columns} == set(by_canonical):
        mapping = {c: by_canonical[_canonical(c)] for c in clean.columns}
    elif len(clean.columns) == len(dirty.columns):
        mapping = dict(zip(clean.columns, dirty.columns))
        warnings.warn("the dirty and clean headers differ; matching them by position",
                      stacklevel=4)
    else:
        raise ValueError(f"the clean table has {len(clean.columns)} columns and the dirty one "
                         f"{len(dirty.columns)}; they cannot be compared cell by cell")
    rows = ({clean.id_column: rid, **{mapping[c]: row.get(c) for c in clean.columns}}
            for rid, row in clean.rows())
    return Table(rows, clean.id_column, dirty.columns)


def _load_csv(spec: dict, root: Path, limit: int | None) -> tuple[Table, Table | None]:
    id_column = spec.get("id_column", "id")
    folder = root / spec.get("folder", spec.get("name", ""))
    dirty = _read_csv(folder / spec.get("dirty", "dirty.csv"), id_column)
    clean_path = folder / spec.get("clean", "clean.csv")
    clean = _align(dirty, _read_csv(clean_path, id_column)) if clean_path.exists() else None
    if limit:
        keep = dirty.row_ids[:limit]
        from .table import restrict
        dirty = restrict(dirty, keep)
        clean = restrict(clean, keep) if clean is not None else None
    return dirty, clean


def _load_catalog(spec: dict, seed: int, limit: int | None, split: str,
                  error_rate: float | None, held_out: bool) -> tuple[Table, Table, list[Injection]]:
    source = dict(spec.get("source", {}))
    if limit:
        source["offers"] = limit
        source["products"] = max(10, limit // 4)
    clean = build(CatalogSpec(seed=seed, **source))
    plan_spec = dict(spec.get("error_plan", {}))
    types = tuple(plan_spec.pop("types", ("typo", "missing", "format", "dependency", "swap")))
    extra = tuple(plan_spec.pop("held_out", ("U1", "U2", "U3")))
    if split != "dev" and held_out:
        types = types + extra
    plan = ErrorPlan(
        rate=float(error_rate if error_rate is not None else plan_spec.pop("rate", 0.12)),
        types=types,
        columns=tuple(plan_spec.pop("columns", ())),
        group_key=tuple(plan_spec.pop("group_key", ())),
        weights=dict(plan_spec.pop("weights", {})),
        column_types={k: tuple(v) for k, v in plan_spec.pop("column_types", {}).items()},
        locale=locale_map(),
    )
    dirty, log = inject(clean, plan, seed=seed)
    return dirty, clean, log


def _load_inline(spec: dict) -> Table:
    return Table(spec["dirty_rows"], spec.get("id_column", "id"),
                 [c for c in spec["dirty_rows"][0] if c != spec.get("id_column", "id")])


def _inline_clean(spec: dict) -> Table | None:
    rows = spec.get("clean_rows")
    if not rows:
        return None
    id_column = spec.get("id_column", "id")
    return Table(rows, id_column, [c for c in rows[0] if c != id_column])
