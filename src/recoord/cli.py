from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Sequence

from .baselines import ABLATIONS, CENTRAL, MERGES, POLICIES
from .datasets import available as datasets_available
from .experiments import RunConfig, run, write, write_csv
from . import report as reporting
from . import sweeps

GROUPS = {"policies": POLICIES, "ablation": ABLATIONS, "central": CENTRAL, "all": sorted(MERGES)}

USAGE = """merge repairs proposed by independent data agents

  recoord datasets                  list the dataset specs
  recoord run catalog               merge one dataset, compare policies
  recoord sweep clients catalog     vary one parameter
  recoord report results/           aggregate what has been written
  recoord check                     run the self-checks
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="recoord", description=USAGE,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("datasets", help="list the available dataset specs")

    run_cmd = sub.add_parser("run", help="run one dataset end to end")
    run_cmd.add_argument("dataset")
    _common(run_cmd)
    run_cmd.add_argument("--methods", nargs="+", default=["policies"],
                         help=f"method names or a group: {', '.join(GROUPS)}")
    run_cmd.add_argument("--analyse", action="store_true",
                         help="enumerate the valid repair sets per component")
    run_cmd.add_argument("--analyse-limit", type=int, default=10)
    run_cmd.add_argument("--repeat", type=int, default=1, help="runs with consecutive seeds")

    sweep_cmd = sub.add_parser("sweep", help="run a parameter sweep")
    sweep_cmd.add_argument("sweep", choices=sweeps.available())
    sweep_cmd.add_argument("dataset")
    _common(sweep_cmd)
    sweep_cmd.add_argument("--values", nargs="+", default=None)

    report_cmd = sub.add_parser("report", help="aggregate written results")
    report_cmd.add_argument("paths", nargs="+")
    report_cmd.add_argument("--by", nargs="+", default=["dataset", "method"])
    report_cmd.add_argument("--compare", nargs="?", const="lww", default=None,
                            help="paired bootstrap of recoord against this method")
    report_cmd.add_argument("--csv", default=None)

    check_cmd = sub.add_parser("check", help="verify the merge guarantees by enumeration")
    check_cmd.add_argument("--limit", type=int, default=60)

    args = parser.parse_args(argv)
    commands = {"datasets": _datasets, "run": _run, "sweep": _sweep,
                "report": _report, "check": _check}
    try:
        return commands[args.command](args)
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        print(f"recoord: {error}", file=sys.stderr)
        return 1


def _datasets(args: argparse.Namespace) -> int:
    print("\n".join(datasets_available()))
    return 0


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", default="data", help="where the CSV benchmarks live")
    parser.add_argument("--out", default="results")
    parser.add_argument("--agent", default="offline", choices=["offline", "llm", "rules"],
                        help="where judgement steps come from")
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="use only the first N rows")
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--error-rate", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=0.5, help="tau")
    parser.add_argument("--cap", type=int, default=2, help="regeneration cap gamma")
    parser.add_argument("--clients", type=int, default=None)
    parser.add_argument("--partition", default=None, help="column that owns the rows")
    parser.add_argument("--row-share", type=float, default=1.0)
    parser.add_argument("--fail-rate", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--waves", type=int, default=1)
    parser.add_argument("--shards", type=int, default=1,
                        help="workers over disjoint components")
    parser.add_argument("--late-constraint", type=int, default=None,
                        help="index of a DC that arrives after the merge")
    parser.add_argument("--joining", default=None, help="a client that joins a running merge")
    parser.add_argument("--freeze", default=None, help="a client that leaves mid-merge")
    parser.add_argument("--recalibrate", action="store_true")
    parser.add_argument("--quiet", action="store_true")


def _config(args: argparse.Namespace, **extra: Any) -> RunConfig:
    known = {f.name for f in fields(RunConfig)}
    payload = {k: v for k, v in vars(args).items() if k in known and v is not None}
    payload.update(extra)
    return RunConfig(**payload)


def _methods(names: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    for name in names:
        out.extend(GROUPS.get(name, [name]))
    return tuple(dict.fromkeys(out))


def _run(args: argparse.Namespace) -> int:
    rows: list[dict[str, Any]] = []
    for offset in range(max(1, args.repeat)):
        cfg = _config(args, dataset=args.dataset, methods=_methods(args.methods),
                      seed=args.seed + offset)
        result = run(cfg)
        name = f"{result['dataset']}" + (f"-seed{cfg.seed}" if args.repeat > 1 else "")
        path = write(result, args.out, name)
        rows += result["rows"]
        if not args.quiet:
            print(f"wrote {path}")
    if not args.quiet:
        print()
        print(reporting.headline(rows))
    return 0


def _sweep(args: argparse.Namespace) -> int:
    cfg = _config(args, dataset=args.dataset)
    kwargs: dict[str, Any] = {}
    if args.values:
        kwargs["values"] = [_number(v) for v in args.values]
    rows = sweeps.SWEEPS[args.sweep](cfg, **kwargs)
    out = Path(args.out) / f"sweep-{args.sweep}-{args.dataset}"
    write_csv(out.with_suffix(".csv"), rows)
    out.with_suffix(".json").write_text(json.dumps({"rows": rows}, indent=2, default=str) + "\n")
    if not args.quiet:
        print(f"wrote {out.with_suffix('.csv')}")
        print()
        print(reporting.table(rows, _sweep_columns(args.sweep, rows)))
    return 0


def _sweep_columns(name: str, rows: Sequence[dict[str, Any]]) -> list[str]:
    knobs = [k for k in ("clients", "rows", "error_rate", "cap", "fail_rate", "retries",
                         "waves", "shards", "late_constraint", "joined", "departed", "layout",
                         "agent_kind", "model", "seed_index", "recalibrated", "study")
             if any(k in r for r in rows)]
    quality = [k for k in ("f1", "f1_seen", "f1_held_out") if any(k in r for r in rows)]
    return ["dataset", "method", *knobs, *quality, "stale", "violations", "null_cells",
            "llm_calls", "replay_calls", "tokens", "seconds"]


def _report(args: argparse.Namespace) -> int:
    rows = reporting.load_rows(args.paths)
    if not rows:
        print("no result rows found", file=sys.stderr)
        return 1
    aggregated = reporting.aggregate(rows, by=args.by)
    print(reporting.table(aggregated))
    if args.compare:
        print()
        print(json.dumps(reporting.compare(rows, against=args.compare), indent=2))
    if args.csv:
        write_csv(args.csv, aggregated)
        print(f"\nwrote {args.csv}")
    return 0


def _check(args: argparse.Namespace) -> int:
    from .checks import run_checks
    failures = run_checks(limit=args.limit)
    for line in failures["lines"]:
        print(line)
    return 0 if not failures["failed"] else 1


def _number(text: str) -> Any:
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


if __name__ == "__main__":
    raise SystemExit(main())
