"""Save exports through a native dialog in explicitly opted-in local runs."""

import os
from pathlib import Path
import subprocess
import sys


def _choose_save_path(file_name: str) -> Path | None:
    if sys.platform == "darwin":
        script = '''on run argv
try
    set destination to choose file name with prompt "Save As" default name (item 1 of argv)
    return POSIX path of destination
on error number -128
    return ""
end try
end run'''
        command = ["osascript", "-e", script, file_name]
    else:
        # A separate process keeps Tk on its main thread, outside Streamlit's worker.
        script = '''import sys
import tkinter as tk
from tkinter.filedialog import asksaveasfilename
root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)
print(asksaveasfilename(parent=root, title="Save As", initialfile=sys.argv[1],
                      defaultextension=sys.argv[2]))
root.destroy()
'''
        command = [sys.executable, "-c", script, file_name, Path(file_name).suffix]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    path = result.stdout.strip()
    return Path(path) if path else None


def save_button(container, label, *, data, file_name, mime, key=None, **kwargs):
    key = key or f"save_{label}"
    if os.environ.get("SNOM_NATIVE_SAVE_DIALOG") != "1":
        container.download_button(
            label, data=data, file_name=file_name,
            mime=mime, key=key, **kwargs,
        )
        return
    kwargs.pop("on_click", None)
    if container.button(label, key=key, **kwargs):
        try:
            path = _choose_save_path(file_name)
            if path is None:
                return
            payload = data() if callable(data) else data
            path.write_bytes(payload.encode("utf-8") if isinstance(payload, str) else payload)
        except (OSError, subprocess.SubprocessError) as exc:
            container.error(f"Could not save file: {exc}")
        else:
            container.success(f"Saved to {path}")
