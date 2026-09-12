#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"

WITH_EVENTS=0
FORCE=0

usage() {
  cat <<'USAGE'
Usage: bash Scripts/fetch_missing_data.sh [--with-events] [--force]

Checks the data needed by ResearchModelTraining.ipynb and runs the local
data refresh/rebuild scripts when files are missing.

Options:
  --with-events  Also rebuild the large local GDELT/news event file.
                 This can take a long time and creates ignored files >100 MB.
  --force        Re-run refresh/rebuild steps even when output files exist.
  -h, --help     Show this help.

Environment:
  PYTHON=/path/to/python  Choose the Python interpreter. Defaults to python.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-events)
      WITH_EVENTS=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$ROOT_DIR"

need_file() {
  local path="$1"
  [[ -s "$path" ]]
}

run_step() {
  echo
  echo "==> $*"
  "$@"
}

echo "Arabica Futures data check"
echo "Repo: $ROOT_DIR"
echo "Python: $($PYTHON_BIN --version 2>&1)"

if ! need_file "data/centralData/yahoo_cot_full_outer_by_date.csv"; then
  cat >&2 <<'EOF'

Missing required base file:
  data/centralData/yahoo_cot_full_outer_by_date.csv

This repository expects the combined Yahoo price + COT source CSV to be
present from GitHub. Restore it from the repository before running this script.
EOF
  exit 1
fi

if [[ "$FORCE" -eq 1 || ! -s "data/weather/open_meteo_coffee_regions_daily.csv" ]]; then
  run_step "$PYTHON_BIN" "Scripts/project/scripts/refresh_weather_cache.py" "--force-all"
else
  echo "OK: data/weather/open_meteo_coffee_regions_daily.csv"
fi

if [[ "$FORCE" -eq 1 || ! -s "data/centralData/yahoo_cot_full_outer_by_date_cot_ffill.csv" ]]; then
  run_step "$PYTHON_BIN" "Scripts/project/scripts/fill_cot_forward_daily.py"
else
  echo "OK: data/centralData/yahoo_cot_full_outer_by_date_cot_ffill.csv"
fi

if [[ "$FORCE" -eq 1 || ! -s "data/centralData/arabica_ml_model_ready.csv" ]]; then
  run_step "$PYTHON_BIN" "Scripts/project/scripts/prepare_ml_dataset.py"
else
  echo "OK: data/centralData/arabica_ml_model_ready.csv"
fi

if [[ "$WITH_EVENTS" -eq 1 ]]; then
  if [[ "$FORCE" -eq 1 || ! -s "data/events/gdelt_coffee_events_2000_2026_filtered_scored.csv" ]]; then
    run_step "$PYTHON_BIN" "Scripts/project/scripts/gdelt_stream_all_years_coffee_events.py" \
      "--start-year" "2000" \
      "--end-year" "2026" \
      "--skip-generated-historical"
  else
    echo "OK: data/events/gdelt_coffee_events_2000_2026_filtered_scored.csv"
  fi
else
  echo
  echo "Skipping large GDELT/news rebuild."
  echo "Run with --with-events if you need data/events/gdelt_coffee_events_2000_2026_filtered_scored.csv locally."
fi

echo
echo "Data check complete."
