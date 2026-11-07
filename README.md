# ReCoord

Independent data agents propose repairs to a shared table. The repairs do not
compose: fixing one cell can invalidate the evidence another repair was derived
from, and two repairs that are fine on their own can break a constraint
together. ReCoord merges them anyway.

Each proposal arrives as a patch plus the queries whose answers the agent read,
a confidence score, and — if the agent wrote one — a small deterministic program
that recomputes the patch. The coordinator then

- tracks which patches can change which answers, and checks only those;
- keeps a repair whose evidence still holds, re-runs its program when the
  evidence moved, and re-prompts the agent only when neither works;
- admits repairs greedily by confidence, rejecting those that would conflict;
- nulls whatever cells are left in a constraint violation.

Programs and private tables stay at the client that owns them. The server sees
patches, query text, scores, and the answers it asks for.

## Install and run

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/recoord run catalog
```

That builds a synthetic multi-vendor catalog, injects errors, runs four agents
over it, and merges their proposals with ReCoord and the baselines. No API key
and no download needed.

```bash
./scripts/run_all.sh          # tests, self-checks, runs, sweeps, one report
./scripts/get_data.sh         # the public CSV benchmarks into data/
./scripts/run_all.sh --full   # include them, five runs each
```

## Commands

```bash
recoord datasets                           # what is available
recoord run <dataset> [--methods ...]      # merge, compare, write results
recoord sweep <sweep> <dataset>            # vary one thing at a time
recoord report results/ --compare lww      # aggregate, with paired intervals
recoord check                              # self-checks (see below)
```

Useful flags: `--limit N` to use a prefix of the rows, `--repeat N` for N seeds,
`--agent llm --model <id>` to use a real model, `--analyse` to enumerate the
valid repair sets per component, `--waves`, `--fail-rate`, `--partition`,
`--clients`, `--cap`. `recoord run --help` lists them all.

`recoord sweep` knows: `policies ablation central calibration clients size
errors waves failures budget departure partition agents backbones seeds`.

## Your own data

Write a spec next to the others in [src/recoord/specs/](src/recoord/specs/) —
start from [catalog.yaml](src/recoord/specs/catalog.yaml), which is the long
one, or [demo.yaml](src/recoord/specs/demo.yaml), which is three rows. A spec
holds four things:

- **where the table comes from**: `kind: csv` reads `data/<folder>/dirty.csv`
  and `clean.csv`; `kind: catalog` generates one and injects errors.
- **constraints**: `{lhs: [...], rhs: x}` for a functional dependency, or a
  list of `predicates` for any denial constraint.
- **rules**: one repair pattern each. A rule with a `program`
  ([programs.py](src/recoord/programs.py) has them) is replayable; a rule
  without one is a judgment the agent has to make. `scope` says what it reads:
  `row` its own row, `row_in_group` its whole constraint group but writing one
  cell, `group` the group as a single bundled proposal.
- **clients**: who runs which rules.

```bash
.venv/bin/recoord run myspec.yaml --methods recoord lww
```

## Agents

`--agent simulated` (the default) is **not** a model. It answers judgments from
the ground truth with a per-client error rate, which makes every run
deterministic and free. It is there so the merging logic can be tested and the
pipeline can be reproduced; it says nothing about how an LLM behaves.

`--agent llm` uses a real model through [llm.py](src/recoord/llm.py) (Anthropic
or OpenAI, chosen by the model id, `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`).
Every completion is cached under `.cache/llm`, so a repeat run is free and
reproducible. `--agent rules` is the control: the deterministic rules only,
without the ones an agent writes at runtime.

## Self-checks

`recoord check` verifies, by enumeration on small instances, three things the
code is supposed to guarantee: the output has no stale repair, no constraint
violation and no write conflict; permuting the arrival order of the proposals
does not change the output; and two pools whose patches and scores are
identical, but whose queries differ, get different answers — so a merge policy
that ignores evidence cannot tell them apart.

## What is where

| | |
|---|---|
| [coordinator.py](src/recoord/coordinator.py) | the merge algorithm |
| [hypergraph.py](src/recoord/hypergraph.py) | which patch can disturb which evidence |
| [rrr.py](src/recoord/rrr.py) | retain / replay / regenerate |
| [constraints.py](src/recoord/constraints.py) | denial constraints and violations |
| [federation.py](src/recoord/federation.py) | clients, calls to them, and their cost |
| [agents.py](src/recoord/agents.py) | agents built from rules |
| [baselines.py](src/recoord/baselines.py) | the other merge policies |
| [analysis.py](src/recoord/analysis.py) | per-component optima and conditions |
| [experiments.py](src/recoord/experiments.py), [sweeps.py](src/recoord/sweeps.py) | the drivers |

Optional extras: `pip install -e '.[ilp]'` for exact per-component optima,
`'.[llm]'` for the model clients, `'.[sql]'` for DuckDB.

## Notes

- Two rules that read and write the same column disturb each other's evidence
  both ways. That is a cycle in the replay graph, and the merge resolves it by
  making both non-replayable, so those repairs get re-prompted instead of
  re-run. The `frozen` count in the results says how often this happened; if it
  is high and `replayed` is zero, that is why.
- Coordination cost grows with the size of the constraint groups, not just the
  row count: a dependency whose groups grow with the table (the catalog's
  `country -> currency`) makes it quadratic. The `size` sweep shows the curve.
- Results land in `results/` as JSON (full, with the merged table) and CSV (one
  row per method). `recoord report` reads either.
- A NULL counts as *not repaired* on a wrong cell and as a *wrong repair* on a
  correct one, so nulling is never free.
