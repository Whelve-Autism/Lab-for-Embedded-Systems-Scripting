#!/usr/bin/env bash
# =============================================================================
# Project 3 — Automated experiment pipeline for NICE (Project 2)
#
# Role of this script
#   1) Loop over FE budgets and invoke run_nice_p3.py
#   2) Append stdout/stderr to experiment.log (run id + parameters + P3_RESULT)
#   3) Extract only this batch's P3_RESULT lines into a preserved summary
#
# Log format (append-only)
#   ---- RUN_START ts=... run_id=... max_fes=... ----
#   <python stdout, including progress lines>
#   P3_RESULT|run_id=...|max_fes=...|seed=...|n_funcs=...|mean_abs_error=...|n_fail=...|out=...
#   ---- RUN_END   ts=... run_id=... status=ok|fail ----
#
# Usage
#   ./run_experiments.sh              # full P3 configs
#   ./run_experiments.sh --clean       # archive previous logs, then run
#   ./run_experiments.sh --help
#
# Smoke test (does not replace full submission runs)
#   FE_LIST="2000" FUNCS="1-2" WORKERS=2 ./run_experiments.sh
#
# Cron example (absolute paths required)
#   0 * * * * /bin/bash /ABS/PATH/project_3/run_experiments.sh >> /ABS/PATH/project_3/cron_execution.log 2>&1
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

PYTHON_ENTRY="${SCRIPT_DIR}/run_nice_p3.py"
EXPERIMENT_LOG="${SCRIPT_DIR}/experiment.log"
SUMMARY_REPORT="${SCRIPT_DIR}/summary_report.txt"
RESULTS_ROOT="${SCRIPT_DIR}/results/P3"
ARCHIVE_DIR="${SCRIPT_DIR}/archive"

# Default experiment grid (override via environment)
FE_LIST="${FE_LIST-20000 50000 100000}"
SEED="${SEED-1001}"
FUNCS="${FUNCS-1-30}"
WORKERS="${WORKERS-16}"
FE_VALUES=()

usage() {
  cat <<'EOF'
Project 3 automation: run_experiments.sh

Options:
  --help      Show this help
  --clean     Move existing experiment.log / summary_report.txt into archive/
              before an intentional manual restart. Per-batch results and
              summaries are retained. Cron hourly runs should NOT pass --clean.

Environment overrides (optional):
  FE_LIST     Space-separated FE budgets   (default: "20000 50000 100000")
  SEED        Exactly one non-negative integer seed (default: 1001)
  FUNCS       CEC2014 function spec        (default: 1-30)
  WORKERS     Parallel workers             (default: 16)
  PYTHON_BIN  Python interpreter           (default: .venv/bin/python, then python3/python)

Examples:
  chmod +x run_experiments.sh
  ./run_experiments.sh
  ./run_experiments.sh --clean
  FE_LIST="2000" FUNCS="1-2" ./run_experiments.sh

Manual pages / help demos (for report screenshots):
  man bash
  help for
  python3 run_nice_p3.py --help
EOF
}

ts_now() {
  date '+%Y-%m-%d %H:%M:%S %z'
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

resolve_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    if [[ -x "$PYTHON_BIN" ]]; then
      echo "$PYTHON_BIN"
      return
    fi
    command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "PYTHON_BIN not found: $PYTHON_BIN"
    echo "$PYTHON_BIN"
    return
  fi
  # Prefer the local project_3 virtualenv when present.
  if [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
    echo "${SCRIPT_DIR}/.venv/bin/python"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return
  fi
  if command -v python >/dev/null 2>&1; then
    echo "python"
    return
  fi
  die "No python3/python found in PATH (run ./install_env.sh first)"
}

validate_inputs() {
  [[ -f "$PYTHON_ENTRY" ]] || die "Missing Python entry: $PYTHON_ENTRY"
  [[ -d "${SCRIPT_DIR}/../project_2" ]] || die "Project 2 directory not found next to project_3"
  [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || die "WORKERS must be a positive integer, got: $WORKERS"
  [[ "$SEED" =~ ^(0|[1-9][0-9]*)$ ]] || die "SEED must be one non-negative integer, not a list or range: $SEED"
  [[ "$FUNCS" =~ [^[:space:]] ]] || die "FUNCS must not be empty or whitespace"
  [[ "$FE_LIST" =~ [^[:space:]] ]] || die "FE_LIST must contain at least one budget"
  # Normalize newlines, then validate every token before creating batch files.
  IFS=$' \t\n' read -r -a FE_VALUES <<< "${FE_LIST//$'\n'/ }"
  local fe seen=" "
  for fe in "${FE_VALUES[@]}"; do
    [[ "$fe" =~ ^[1-9][0-9]*$ ]] || die "Invalid FE budget in FE_LIST: $fe"
    [[ "$seen" != *" $fe "* ]] || die "Duplicate FE budget in FE_LIST: $fe"
    seen+="$fe "
    if (( fe < 100 )); then
      die "FE budget too small for NICE (need >= 100): $fe"
    fi
  done
}

archive_logs() {
  mkdir -p "$ARCHIVE_DIR"
  local stamp
  stamp="$(date '+%Y%m%d_%H%M%S')_$$"
  if [[ -f "$EXPERIMENT_LOG" ]]; then
    mv "$EXPERIMENT_LOG" "${ARCHIVE_DIR}/experiment_${stamp}.log"
    echo "[INFO] archived experiment.log -> archive/experiment_${stamp}.log"
  fi
  if [[ -f "$SUMMARY_REPORT" ]]; then
    mv "$SUMMARY_REPORT" "${ARCHIVE_DIR}/summary_report_${stamp}.txt"
    echo "[INFO] archived summary_report.txt -> archive/summary_report_${stamp}.txt"
  fi
}

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
DO_CLEAN=0
for arg in "$@"; do
  case "$arg" in
    --help|-h)
      usage
      exit 0
      ;;
    --clean)
      DO_CLEAN=1
      ;;
    *)
      die "Unknown option: $arg (try --help)"
      ;;
  esac
done

validate_inputs
PYTHON_CMD="$(resolve_python)"

if (( DO_CLEAN == 1 )); then
  archive_logs
fi

mkdir -p "$RESULTS_ROOT"
BATCH_ID="$(date '+%Y%m%d_%H%M%S')_$$"
BATCH_DIR="${RESULTS_ROOT}/${BATCH_ID}"
BATCH_SUMMARY="${BATCH_DIR}/summary_report.txt"
# Plain mkdir deliberately refuses to reuse an existing batch directory.
mkdir "$BATCH_DIR"
touch "$EXPERIMENT_LOG"

{
  echo "================================================================"
  echo "BATCH_START ts=$(ts_now) batch_id=${BATCH_ID} host=$(hostname 2>/dev/null || echo unknown)"
  echo "cwd=$SCRIPT_DIR"
  echo "python=$PYTHON_CMD"
  echo "FE_LIST=$FE_LIST SEED=$SEED FUNCS=$FUNCS WORKERS=$WORKERS"
  echo "batch_dir=$BATCH_DIR"
  echo "================================================================"
} >> "$EXPERIMENT_LOG"

echo "[INFO] Project 3 batch ${BATCH_ID} start at $(ts_now)"
echo "[INFO] FE budgets: $FE_LIST"

ANY_FAIL=0

for FE in "${FE_VALUES[@]}"; do
  RUN_ID="${BATCH_ID}_NICE_FE_${FE}"
  OUT_DIR="${BATCH_DIR}/NICE_FE_${FE}"
  mkdir "$OUT_DIR"

  {
    echo ""
    echo "---- RUN_START ts=$(ts_now) run_id=${RUN_ID} max_fes=${FE} seed=${SEED} funcs=${FUNCS} ----"
  } >> "$EXPERIMENT_LOG"

  echo "[INFO] Running ${RUN_ID} (max_fes=${FE}) ..."

  if "$PYTHON_CMD" "$PYTHON_ENTRY" \
    --run-id "$RUN_ID" \
    --max-fes "$FE" \
    --funcs "$FUNCS" \
    --seeds "$SEED" \
    --workers "$WORKERS" \
    --out "$OUT_DIR" >> "$EXPERIMENT_LOG" 2>&1; then
    rc=0
  else
    rc=$?
  fi

  if (( rc == 0 )); then
    status="ok"
    echo "[INFO] ${RUN_ID} finished OK"
  else
    status="fail"
    ANY_FAIL=1
    echo "[WARN] ${RUN_ID} failed with exit code ${rc}" >&2
    {
      # The Python runner emits P3_RESULT only on success. Bash owns the
      # single failure record, including startup/import errors.
      echo "P3_RESULT|run_id=${RUN_ID}|max_fes=${FE}|seed=${SEED}|n_funcs=NA|mean_abs_error=nan|n_fail=-1|out=${OUT_DIR}|status=fail|exit=${rc}"
    } >> "$EXPERIMENT_LOG"
  fi

  {
    echo "---- RUN_END   ts=$(ts_now) run_id=${RUN_ID} status=${status} exit=${rc} ----"
  } >> "$EXPERIMENT_LOG"
done

# ---------------------------------------------------------------------------
# grep | awk → summary_report.txt
# Field layout of P3_RESULT (pipe-separated):
#   $1=P3_RESULT  $2=run_id=...  $3=max_fes=...  $4=seed=...
#   $5=n_funcs=...  $6=mean_abs_error=...  $7=n_fail=...  $8=out=...
# ---------------------------------------------------------------------------
{
  echo "Project 3 Summary Report"
  echo "Batch-ID: ${BATCH_ID}"
  echo "Generated: $(ts_now)"
  echo "Source log: ${EXPERIMENT_LOG}"
  echo "Extraction: grep '^P3_RESULT|' experiment.log | awk ... (current Batch-ID only)"
  echo "Parameters: FE_LIST=$FE_LIST SEED=$SEED FUNCS=$FUNCS WORKERS=$WORKERS"
  echo ""
  echo "Parameter combinations and mean absolute errors"
  echo "(E_mean = average over test functions of |f_obtained - f*| )"
  echo "--------------------------------------------------------------------"
  printf "%-44s %-10s %-10s %-10s %-8s %s\n" "run_id" "max_fes" "n_funcs" "n_fail" "status" "mean_abs_error"
  echo "--------------------------------------------------------------------"

  # shellcheck disable=SC2016
  grep '^P3_RESULT|' "$EXPERIMENT_LOG" | awk -F'|' \
    -v prefix="${BATCH_ID}_" -v expected="${#FE_VALUES[@]}" '
  BEGIN { printed=0; bad=0; }
  {
    run_id=""; max_fes=""; n_funcs=""; mean_err=""; n_fail="";
    for (i = 1; i <= NF; i++) {
      split($i, kv, "=");
      key=kv[1];
      val=substr($i, length(key)+2);
      if (key=="run_id") run_id=val;
      else if (key=="max_fes") max_fes=val;
      else if (key=="n_funcs") n_funcs=val;
      else if (key=="mean_abs_error") mean_err=val;
      else if (key=="n_fail") n_fail=val;
    }
    if (index(run_id, prefix) != 1) next;
    if (seen[run_id]++) {
      print "[ERROR] Duplicate result for " run_id > "/dev/stderr";
      bad=1;
      next;
    }
    state=(n_fail == "0" ? "ok" : "fail");
    printf "%-44s %-10s %-10s %-10s %-8s %s\n", run_id, max_fes, n_funcs, n_fail, state, mean_err;
    printed++;
  }
  END {
    if (bad || printed != expected) {
      print "[ERROR] Current batch has " printed " result rows; expected " expected > "/dev/stderr";
      exit 1;
    }
  }'

  echo "--------------------------------------------------------------------"
  echo ""
  echo "Raw P3_RESULT lines (must agree with the table above):"
  grep '^P3_RESULT|' "$EXPERIMENT_LOG" | awk -F'|' -v prefix="run_id=${BATCH_ID}_" '
    index($2, prefix) == 1 { print }
  '
  echo ""
  echo "Only the current batch is included. Earlier batches remain in experiment.log."
  echo "This summary is preserved at: ${BATCH_SUMMARY}"
} > "$BATCH_SUMMARY"

# Keep a convenient latest-summary copy without losing any earlier batch.
cp "$BATCH_SUMMARY" "$SUMMARY_REPORT"

{
  echo "BATCH_END ts=$(ts_now) batch_id=${BATCH_ID} summary=${BATCH_SUMMARY} any_fail=${ANY_FAIL}"
  echo "================================================================"
  echo ""
} >> "$EXPERIMENT_LOG"

echo "[INFO] Preserved batch summary: $BATCH_SUMMARY"
echo "[INFO] Latest summary: $SUMMARY_REPORT"
echo "[INFO] Project 3 batch ${BATCH_ID} end at $(ts_now)"

if (( ANY_FAIL != 0 )); then
  exit 1
fi
exit 0
