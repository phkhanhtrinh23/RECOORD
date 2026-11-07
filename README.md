# ReCoord

Independent data agents propose repairs to a shared table. The repairs do not
compose: fixing one cell can invalidate the evidence another repair was derived
from, and two repairs that are fine on their own can break a constraint
together. ReCoord merges them anyway, and keeps every repair it accepts
supported by evidence that still holds.

Each proposal arrives as a patch, the queries whose answers the agent read, a
confidence score, and — when the agent wrote one — a deterministic program that
recomputes the patch. The coordinator then

- tracks which patches can change which answers, and checks only those;
- keeps a repair whose evidence still holds, re-runs its program when the
  evidence moved, and re-prompts the agent only when neither works;
- admits repairs greedily by confidence under exact conflict tests;
- nulls whatever cells are left in a constraint violation.

Programs and private tables stay at the client that owns them. The server sees
patches, query text, scores, and the answers it asks for — never client data.

## Install and run

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/recoord run catalog
```

That builds a multi-vendor catalog, injects errors, runs four agents over it,
and merges their proposals with ReCoord and eleven baselines. No API key and no
download needed.

```bash
./scripts/run_all.sh          # tests, self-checks, runs, sweeps, one report
./scripts/get_data.sh         # the public CSV benchmarks into data/
./scripts/run_all.sh --full   # include them, five seeds each
```

## Commands

```bash
recoord datasets                           # what is available
recoord run <dataset> [--methods ...]      # merge, compare, write results
recoord sweep <sweep> <dataset>            # vary one thing at a time
recoord report results/ --compare lww      # aggregate, with paired intervals
recoord check                              # the self-checks below
```

`--methods` takes names or a group: `policies`, `ablation`, `central`, `all`.
Other flags: `--limit N` for a prefix of the rows, `--repeat N` for N seeds,
`--agent llm --model <id>` for a real model, `--analyse` to enumerate the valid
repair sets per component, `--shards N` to spread the rounds over workers,
`--waves`, `--clients`, `--cap`, `--fail-rate`, `--partition`, `--joining`,
`--late-constraint`. `recoord run --help` lists them all.

`recoord sweep` knows: `policies ablation central calibration clients size
errors waves failures budget departure joining constraints sharding partition
agents backbones seeds`.

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
  without one is a judgement the agent makes. `scope` says what it reads: `row`
  its own row, `row_in_group` its whole constraint group while writing one
  cell, `group` the group as a single bundled proposal.
- **clients**: who runs which rules.

```bash
.venv/bin/recoord run myspec.yaml --methods recoord lww
```

## Agents

Three interchangeable backends behind one interface, so the coordinator and
every baseline see the same kind of proposal whichever one runs.

- `--agent offline` (default) resolves judgement steps against the labelled
  reference at a per-client accuracy. Deterministic, free, no key.
- `--agent llm` uses a real model through [llm.py](src/recoord/llm.py)
  (Anthropic or OpenAI, picked from the model id, `ANTHROPIC_API_KEY` /
  `OPENAI_API_KEY`). Completions are cached under `.cache/llm`, so a second run
  costs nothing and returns the same answers.
- `--agent rules` is the fixed rule library: the deterministic rules a person
  writes in advance, without the ones an agent only writes at run time.

## Self-checks

`recoord check` verifies by enumeration, on small instances, four properties:
the output has no stale repair, no constraint violation and no write conflict;
permuting the arrival order of the proposals does not change it; splitting the
rounds over any number of workers returns the one-worker result; and two pools
whose patches and scores are identical but whose queries differ get different
answers, so a policy that ignores evidence cannot tell them apart.

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
`'.[llm]'` for the model clients.

## Notes

- Results land in `results/` as JSON (full, with the merged table) and CSV (one
  row per method). `recoord report` reads either, aggregates over seeds, and
  gives paired bootstrap intervals with a Holm correction.
- A NULL counts as *not repaired* on a wrong cell and as a *wrong repair* on a
  correct one, so nulling is never free.
- Each repair's replayability is decided per lineage. Two rules that both read
  and write one column depend on each other in both directions, so they are
  merged as judgements rather than re-run; the `frozen` column counts them.
- Cost scales with the size of the constraint groups, not only the row count.
  The `size` sweep reports the curve, and `--shards` spreads the rounds over
  disjoint components without changing the result.
