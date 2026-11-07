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
.venv/bin/recoord run catalog --limit 300
```

That builds a multi-vendor catalog, injects errors, runs four agents over it,
and merges their proposals with ReCoord and ten baselines. No API key and no
download needed. Drop `--limit` for the whole split: 4,800 offers from
eighteen vendors, with 1,200 more held back under `--split dev`.

Evidence queries run on an in-process relational engine by default, and
`--engine duckdb` answers them through SQL on DuckDB instead. Clients are
polled sixteen at a time; `--concurrency` changes that. Neither setting
changes the result.

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
recoord check                              # verify the guarantees by enumeration
```

## Data Loading

Write a spec next to the others in [recoord/specs/](recoord/specs/) —
start from [catalog.yaml](recoord/specs/catalog.yaml), which is the long
one, or [demo.yaml](recoord/specs/demo.yaml), which is three rows. A spec
holds four things:

- **where the table comes from**: `kind: csv` reads `data/<folder>/dirty.csv`
  and `clean.csv`; `kind: catalog` generates one and injects errors.
- **constraints**: `{lhs: [...], rhs: x}` for a functional dependency, or a
  list of `predicates` for any denial constraint.
- **rules**: one repair pattern each. A rule with a `program`
  ([programs.py](recoord/programs.py) has them) is replayable; a rule
  without one is a judgement the agent makes. `scope` says what it reads: `row`
  its own row, `row_in_group` its whole constraint group while writing one
  cell, `group` the group as a single bundled proposal.
- **clients**: who runs which rules.

```bash
.venv/bin/recoord run myspec.yaml --methods recoord lww
```

## Agents

An agent is a set of rules. A rule with a program repairs a cell on its own; a
rule without one leaves a judgement step, and the three backends differ only in
who answers it. All of them emit the same proposals, so the coordinator and
every baseline behave the same whichever one runs.

- `--agent llm` puts a model behind the judgement steps through
  [llm.py](recoord/llm.py). `--model` takes a backbone name — `gpt-4o-mini`
  (the default), `gpt-4o`, `claude-3.5-sonnet`, `llama-3.1-70b`,
  `qwen2.5-72b` — or any model id. Each resolves to a pinned version and a
  provider; the open-weights two go through any OpenAI-compatible endpoint, so
  set `--base-url` or `OPENAI_BASE_URL` to point at your own. Calls run at
  temperature 0, sixteen at a time, and token counts come from the API
  response. Completions are cached under `.cache/llm`, so a repeat run is free
  and returns the same answers.
- `--agent offline` (the default) answers them from the dataset's reference
  table at a per-client accuracy, and prices the calls with the per-call cost
  model in [agents.py](recoord/agents.py). It needs no key, costs nothing and
  repeats exactly, so it is the backend to develop and test against: every
  policy draws from one identical pool, which makes the merge comparison like
  for like.
- `--agent rules` drops the judgement steps and keeps the programs: the fixed
  rule library a person writes in advance.

```bash
recoord sweep backbones catalog     # the five backbones, head to head
```
