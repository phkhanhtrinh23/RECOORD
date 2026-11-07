#!/usr/bin/env bash
# Everything end to end: install, test, self-checks, runs, sweeps, report.
#
#   ./scripts/run_all.sh              # synthetic data only, a few minutes
#   ./scripts/run_all.sh --full       # also the CSV benchmarks in data/, 5 runs each
#   ./scripts/run_all.sh --quick      # smoke test only
#
# Options: OUT=<dir> SEEDS=<n> LIMIT=<rows> AGENT=simulated|llm|rules
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-}"
OUT="${OUT:-results}"
AGENT="${AGENT:-simulated}"
case "$MODE" in
  --quick) SEEDS="${SEEDS:-1}"; LIMIT="${LIMIT:-60}";  SWEEPS=(ablation) ;;
  --full)  SEEDS="${SEEDS:-5}"; LIMIT="${LIMIT:-0}";   SWEEPS=(policies ablation central calibration clients errors waves failures budget departure partition agents size) ;;
  *)       SEEDS="${SEEDS:-3}"; LIMIT="${LIMIT:-400}"; SWEEPS=(policies ablation central calibration waves failures budget) ;;
esac

PY=.venv/bin/python
if [[ ! -x "$PY" ]]; then
  echo "== creating .venv"
  python3 -m venv .venv
fi
echo "== installing"
$PY -m pip install -q --upgrade pip
$PY -m pip install -q -e '.[dev]'

echo "== tests"
$PY -m pytest -q

echo "== self-checks"
$PY -m recoord.cli check

limit_args=()
[[ "$LIMIT" != "0" ]] && limit_args=(--limit "$LIMIT")

DATASETS=(demo catalog)
if [[ "$MODE" == "--full" ]]; then
  for name in hospital flights beers rayyan tax; do
    [[ -s "data/$name/dirty.csv" ]] && DATASETS+=("$name")
  done
  if [[ ${#DATASETS[@]} -eq 2 ]]; then
    echo "   (no CSV benchmarks in data/ -- run ./scripts/get_data.sh first)"
  fi
fi

echo "== runs: ${DATASETS[*]}"
for name in "${DATASETS[@]}"; do
  $PY -m recoord.cli run "$name" --out "$OUT/runs" --methods policies \
      --agent "$AGENT" --repeat "$SEEDS" --quiet "${limit_args[@]}"
  echo "   $name done"
done

echo "== sweeps on catalog: ${SWEEPS[*]}"
for sweep in "${SWEEPS[@]}"; do
  $PY -m recoord.cli sweep "$sweep" catalog --out "$OUT/sweeps" \
      --agent "$AGENT" --quiet "${limit_args[@]}" || echo "   sweep $sweep failed"
done

echo
echo "== report"
$PY -m recoord.cli report "$OUT/runs" --compare lww --csv "$OUT/summary.csv"
echo
echo "results in $OUT/  (runs/ compares policies, sweeps/ holds the parameter studies)"
