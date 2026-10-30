# Install and run the s-SNOM Spectrum Calculator locally

## Quick start (no Python or Git needed)

1. On GitHub click **Code → Download ZIP** and unzip it.
2. Double-click **Install_SNOM_Calculator_Windows.bat** (Windows) or **Install_SNOM_Calculator_macOS.command** (macOS).
   - macOS first time: the launcher may be blocked. Click **Done**, open **System Settings → Privacy & Security**, scroll to **Security**, click **Open Anyway** next to the launcher and confirm with your password or Touch ID. Repeat once per launcher. On older macOS: right-click → Open.
   - Windows first time: if SmartScreen shows "Windows protected your PC", click **More info** → **Run anyway**.
3. The installer uses an installed Python 3.11-3.13 if found (otherwise downloads 3.13) and downloads the packages (a few minutes, internet needed). It then opens the app. For later launches, double-click **Open_SNOM_Calculator_Windows.bat** (Windows) or **Open_SNOM_Calculator_macOS.command** (macOS), already included in the repository. The Open launcher starts Streamlit directly without installing anything. Run the Install launcher again if installation or repair is needed.
4. Your browser opens at <http://localhost:8501>. Close the terminal window to stop the app.

This starts the app bound to `127.0.0.1` so it is accessible only on your computer. To reinstall or update, download the new ZIP
(or delete the `.venv` folder) and double-click again.

Keep the whole calculator folder together when moving it. The Open launcher resolves its current folder automatically; after a move, run the Install launcher once to repair the local Python environment, which can require internet access. Your `custom_material_library.json` moves with the folder.

For convenient desktop access, optionally create a shortcut to **Open_SNOM_Calculator_Windows.bat** (Windows: right-click → Show more options → Send to → Desktop (create shortcut)) or an alias to **Open_SNOM_Calculator_macOS.command** (macOS: right-click → Make Alias, then move the alias to the Desktop). The installer does not create desktop files. If you later move the calculator folder and the shortcut stops working, recreate it from the new location.

The rest of this page is the manual route.

## Update the built-in material catalog

After installing the app, close its terminal window and double-click **Update_Material_Library_Windows.bat** (Windows) or **Update_Material_Library_macOS.command** (macOS) in the calculator folder. On macOS, use right-click → Open if needed. The updater downloads the latest Materials Library from GitHub into this app's `.venv`, checks the catalog import, and waits so you can read the result. Restart the app using its usual launcher to see new materials.

Internet access is required, but Git is not. Other installed packages and browser-session custom materials are not changed. The updater also supports manual `.venv` installations with pip.

The latest catalog can differ from the tested commit pinned in `pyproject.toml`. When the Install launcher repairs the environment (for example after a move), or when you reinstall the app's dependencies, the pinned commit is restored. Run the updater again afterwards if desired. To explicitly restore the tested catalog, use:

```bash
.uv/uv pip install --python .venv/bin/python --reinstall-package materials-library -c constraints.txt -e .
```

Windows:

```bat
.uv\uv.exe pip install --python .venv\Scripts\python.exe --reinstall-package materials-library -c constraints.txt -e .
```

On Intel Macs, use `constraints-intel-mac.txt`. For manual installations without `.uv`, repeat manual installation step 4 below.

# Manual install

Step-by-step guide for a first-time user. Commands are for macOS/Linux; Windows
differences are marked.

## 1. Install the prerequisites

You need two tools: **Python 3.13** and **Git**.

| Tool | Check | Install |
|------|-------|---------|
| Python 3.13 (3.11 or 3.12 also work) | `python3.13 --version` | macOS: `brew install python@3.13`. Windows/Linux: download from <https://www.python.org/downloads/> (Windows: tick "Add python.exe to PATH"). |
| Git | `git --version` | macOS: `xcode-select --install`. Windows: <https://git-scm.com/download/win>. Linux: `sudo apt install git`. |

Git is only used to clone the code in step 2 (you can download the ZIP instead).
You also need an internet connection and access to GitHub: the material catalog
is downloaded as a zip archive from GitHub during step 4 (Git is not needed for that).

## 2. Get the code

```bash
git clone https://github.com/vicentini-edoardo/SNOM_Calculator.git
cd SNOM_Calculator
```

## 3. Create a virtual environment

A virtual environment keeps the app's packages separate from the rest of your system.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

Windows:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
```

The prompt now starts with `(.venv)`. Run the `activate` line again every time
you open a new terminal.

## 4. Install the dependencies

```bash
python -m pip install -c constraints.txt setuptools wheel
python -m pip install --no-build-isolation -r requirements.txt
```

This installs the exact tested versions of Streamlit, NumPy, SciPy and the
material catalog. It can take a few minutes.

On Intel Macs, `constraints.txt` pins llvmlite 0.47, which has no Intel Mac wheel.
Use the Intel constraints instead:

```bash
python -m pip install -c constraints-intel-mac.txt setuptools wheel
python -m pip install --no-build-isolation -c constraints-intel-mac.txt .
```

Optional check: `python -m pip check` should print `No broken requirements found.`

## 5. Start the app

```bash
streamlit run app.py
```

Your browser opens at <http://localhost:8501>. Stop the app with `Ctrl+C` in the terminal.

## 6. Next time

```bash
cd SNOM_Calculator
source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
streamlit run app.py
```

To update: `git pull`, then repeat step 4.

## Optional: use the engines from Python or Jupyter 

```bash
python -m pip install -e . jupyterlab
python -m jupyterlab examples/engine_spectra.ipynb
```

Select the kernel from this `.venv`.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Windows launcher reports `Unable to connect to the remote server` or `Download failed` | This happens while downloading uv, before the calculator starts. Use the updated launcher, which enables TLS 1.2 and tries the official GitHub installer if astral.sh is unreachable. If both fail, check network/proxy/firewall access or use the manual Python installation above. Python and package downloads still require internet access. |
| `python3.13: command not found` | Python 3.13 not installed or not on PATH. Redo step 1, or use `python3.12` / `python3.11`. |
| Install fails fetching `materials-library` | No access to github.com (the catalog is a zip download, Git is not needed). Check network/proxy/firewall. |
| `streamlit: command not found` | Virtual environment not active. Run the `activate` line from step 3. |
| PowerShell blocks `Activate.ps1` | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then retry. |
| Port 8501 busy | `streamlit run app.py --server.port 8502` |
