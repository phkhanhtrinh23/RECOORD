#!/usr/bin/env bash
# Fetch the public CSV benchmarks into data/<name>/{dirty.csv,clean.csv}.
#
#   ./scripts/get_data.sh            # all of them
#   ./scripts/get_data.sh hospital   # just one
#
# These come from the Baran repository. Tax is large; skip it with SKIP_TAX=1.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="https://raw.githubusercontent.com/BigDaMa/raha/master/datasets"
ALL=(hospital flights beers rayyan tax)
WANT=("${@:-${ALL[@]}}")

for name in "${WANT[@]}"; do
  if [[ "$name" == "tax" && "${SKIP_TAX:-0}" == "1" ]]; then
    echo "skipping tax"; continue
  fi
  mkdir -p "data/$name"
  for file in dirty clean; do
    target="data/$name/$file.csv"
    if [[ -s "$target" ]]; then
      echo "have $target"; continue
    fi
    echo "fetching $name/$file.csv"
    if ! curl -fsSL "$BASE/$name/$file.csv" -o "$target"; then
      rm -f "$target"
      echo "  could not fetch $name/$file.csv -- download it by hand into data/$name/" >&2
    fi
  done
done

echo
echo "datasets present:"
for name in "${ALL[@]}"; do
  [[ -s "data/$name/dirty.csv" ]] && printf '  %-10s %s rows\n' "$name" "$(($(wc -l < "data/$name/dirty.csv") - 1))"
done
