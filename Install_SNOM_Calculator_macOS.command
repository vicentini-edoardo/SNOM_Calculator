#!/bin/bash
# Double-click launcher: first run installs everything (no Python/Git needed), then starts the app.
set -o pipefail
cd "$(dirname "$0")" || exit 1
UV="$PWD/.uv/uv"
CONSTRAINTS=constraints.txt
[ "$(uname -s)-$(uname -m)" = "Darwin-x86_64" ] && CONSTRAINTS=constraints-intel-mac.txt
if [ ! -x "$UV" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | UV_UNMANAGED_INSTALL="$PWD/.uv" sh || { echo "Download failed. Check your internet connection."; read -r -p "Press Enter to close"; exit 1; }
  [ -x "$UV" ] || { echo "Download failed: uv was not installed."; read -r -p "Press Enter to close"; exit 1; }
fi
MOVED=0
if [ -f .venv/snom-install-path ] && [ "$(cat .venv/snom-install-path)" != "$PWD" ]; then
  MOVED=1
fi
if [ "$MOVED" = 1 ] || ! .venv/bin/python -c 'import sys; assert (3, 11) <= sys.version_info[:2] < (3, 14); import snom_calculator.app' >/dev/null 2>&1 \
    || ! "$UV" pip check --python .venv/bin/python >/dev/null 2>&1; then
  echo "Installing or repairing the environment (a few minutes)..."
  if [ "$MOVED" = 1 ] || ! .venv/bin/python -c 'import sys; assert (3, 11) <= sys.version_info[:2] < (3, 14)' >/dev/null 2>&1; then
    { "$UV" venv --clear --no-python-downloads --python ">=3.11,<3.14" .venv || "$UV" venv --clear --python 3.13 .venv; } || { echo "Python setup failed."; read -r -p "Press Enter to close"; exit 1; }
  fi
  "$UV" pip install --python .venv/bin/python -c "$CONSTRAINTS" -e . \
    || { echo "Install failed. Run this launcher again to retry."; read -r -p "Press Enter to close"; exit 1; }
fi
printf '%s\n' "$PWD" > .venv/snom-install-path || exit 1
SNOM_MATERIAL_LIBRARY_FILE="$PWD/custom_material_library.json" SNOM_NATIVE_SAVE_DIALOG=1 exec .venv/bin/python -m streamlit run app.py --server.address 127.0.0.1
