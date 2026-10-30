#!/bin/bash
# Update only the GitHub material catalog in the calculator's environment.
cd "$(dirname "$0")" || exit 1

fail() {
  echo "$1"
  read -r -p "Press Enter to close"
  exit 1
}

[ -x .venv/bin/python ] || fail "Run Install_SNOM_Calculator_macOS.command first to install the app."
echo "Close the running calculator before updating."
echo "Downloading the latest Materials Library from GitHub..."
CATALOG_URL=https://github.com/vicentini-edoardo/Materials_Library/archive/HEAD.zip
if [ -x .uv/uv ]; then
  .uv/uv pip install --python .venv/bin/python --no-deps --no-cache \
    --reinstall-package materials-library "$CATALOG_URL" \
    || fail "Update failed. Check your internet connection and retry."
else
  .venv/bin/python -m pip install --no-deps --no-cache-dir --force-reinstall "$CATALOG_URL" \
    || fail "Update failed. Check your internet connection and pip installation, then retry."
fi
.venv/bin/python -c 'from snom_calculator.materials import catalog_names; print(f"Catalog available: {len(catalog_names())} materials.")' \
  || fail "Catalog check failed. See docs/install-guide.md to restore the tested version."
echo "Restart s-SNOM Spectrum Calculator to use the updated materials."
read -r -p "Press Enter to close"
exit 0
