#!/usr/bin/env bash
# Create / refresh the local Project 3 virtualenv under .venv/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BOOT="${PYTHON_BOOT:-python3}"
command -v "$PYTHON_BOOT" >/dev/null 2>&1 || {
  echo "[ERROR] $PYTHON_BOOT not found" >&2
  exit 1
}

if [[ ! -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
  echo "[INFO] creating $SCRIPT_DIR/.venv ..."
  "$PYTHON_BOOT" -m venv "$SCRIPT_DIR/.venv"
fi

PY="$SCRIPT_DIR/.venv/bin/python"
PIP="$SCRIPT_DIR/.venv/bin/pip"

echo "[INFO] upgrading pip ..."
"$PIP" install -U pip -q
echo "[INFO] installing requirements.txt ..."
"$PIP" install -r "$SCRIPT_DIR/requirements.txt" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn

"$PY" - <<'PY'
import cma, numpy, opfunu, pandas
print("[OK] env ready:")
print("  python ", __import__("sys").version.split()[0])
print("  numpy  ", numpy.__version__)
print("  pandas ", pandas.__version__)
print("  opfunu ", getattr(opfunu, "__version__", "?"))
print("  cma    ", cma.__version__)
PY

echo "[INFO] done. Default interpreter for run_experiments.sh -> .venv/bin/python"
