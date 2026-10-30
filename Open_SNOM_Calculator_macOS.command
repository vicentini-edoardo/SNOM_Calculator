#!/bin/bash
# Start the installed app; installation and repairs belong to the Install launcher.
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ] || { [ -f .venv/snom-install-path ] && [ "$(cat .venv/snom-install-path)" != "$PWD" ]; }; then
  echo "Run Install_SNOM_Calculator_macOS.command first to install or repair the environment."
  read -r -p "Press Enter to close"
  exit 1
fi
SNOM_MATERIAL_LIBRARY_FILE="$PWD/custom_material_library.json" SNOM_NATIVE_SAVE_DIALOG=1 exec .venv/bin/python -m streamlit run app.py --server.address 127.0.0.1
