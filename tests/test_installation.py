"""Exercise installation recovery without downloads or launching a server."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("blocked", [0, 1, 2])
def test_windows_bootstrap_download_fallback(tmp_path, blocked):
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:
        pytest.skip("PowerShell is required to exercise the Windows bootstrap")
    line = next(line for line in (ROOT / "Install_SNOM_Calculator_Windows.bat").read_text().splitlines()
                if "-ExecutionPolicy Bypass -Command" in line)
    command = line.split('-Command "', 1)[1][:-1]
    # Replace only network access and installer execution; run the real bootstrap.
    harness = f"""
$script:requests = 0
function irm($url) {{
    Add-Content requests.txt $url
    $script:requests++
    if ($script:requests -le {blocked}) {{ throw 'Connection blocked' }}
    'fake installer'
}}
function iex {{
    param([Parameter(ValueFromPipeline=$true)]$installer)
    process {{ Set-Content installed.txt $installer }}
}}
{command}
"""
    result = subprocess.run([powershell, "-NoProfile", "-Command", harness],
                            cwd=tmp_path, capture_output=True, text=True, timeout=10)
    requests = (tmp_path / "requests.txt").read_text(encoding="utf-8-sig").splitlines()
    assert requests == ["https://astral.sh/uv/install.ps1",
                        "https://github.com/astral-sh/uv/releases/latest/download/uv-installer.ps1"][:min(blocked + 1, 2)]
    assert (tmp_path / "installed.txt").exists() == (blocked < 2)
    assert (result.returncode == 0) == (blocked < 2), result.stderr


def executable(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)


@pytest.fixture
def launcher(tmp_path):
    if os.name == "nt":
        pytest.skip("Bash launcher requires a POSIX host")
    script = tmp_path / "Install_SNOM_Calculator_macOS.command"
    shutil.copy2(ROOT / script.name, script)
    shutil.copy2(ROOT / "Open_SNOM_Calculator_macOS.command", tmp_path / "Open_SNOM_Calculator_macOS.command")
    (tmp_path / "Desktop").mkdir()
    executable(tmp_path / "fake-python", '''#!/bin/bash
if [ "$1" = "-c" ]; then
  [ ! -f .venv/unsupported ] || exit 1
  case "$2" in *snom_calculator.app*) [ -f .venv/installed ] || exit 1;; esac
  exit 0
fi
printf '%s\\n' "$@" > launch-args
printf '%s\\n' "$SNOM_MATERIAL_LIBRARY_FILE" > library-path
''')
    executable(tmp_path / ".uv/uv", '''#!/bin/bash
case "$1 $2" in
  "venv "*) mkdir -p .venv/bin; cp fake-python .venv/bin/python; rm -f .venv/unsupported;;
  "pip install") [ ! -f install-fails ] || exit 1; touch .venv/installed;;
  "pip check") [ -f .venv/installed ] || exit 1;;
  *) exit 2;;
esac
''')
    return script


@pytest.mark.parametrize("existing", ["absent", "empty", "missing-packages", "unsupported", "ready"])
def test_launcher_recovers_and_binds_to_loopback(launcher, existing):
    root = launcher.parent
    if existing != "absent":
        (root / ".venv").mkdir()
    if existing in ("missing-packages", "unsupported", "ready"):
        (root / ".venv/bin").mkdir()
        shutil.copy2(root / "fake-python", root / ".venv/bin/python")
    if existing == "unsupported":
        (root / ".venv/unsupported").touch()
    if existing == "ready":
        (root / ".venv/installed").touch()
    result = subprocess.run(["bash", str(launcher)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert (root / "library-path").read_text().strip() == str(root / "custom_material_library.json")
    assert (root / "launch-args").read_text().splitlines() == [
        "-m", "streamlit", "run", "app.py", "--server.address", "127.0.0.1",
    ]
    # Repository launchers must also recover from a later interrupted install.
    (root / ".venv/installed").unlink()
    result = subprocess.run(["bash", str(launcher)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert (root / ".venv/installed").exists()


def test_failed_download_stops_before_environment_creation(launcher):
    root = launcher.parent
    (root / ".uv/uv").unlink()
    executable(root / "bin/curl", "#!/bin/bash\nexit 22\n")
    env = dict(os.environ, PATH=str(root / "bin") + os.pathsep + os.environ["PATH"])
    result = subprocess.run(["bash", str(launcher)], env=env, input="\n",
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    assert "Download failed" in result.stdout
    assert "First run" not in result.stdout
    assert not (root / ".venv").exists()


def test_failed_install_does_not_launch_and_can_be_retried(launcher):
    root = launcher.parent
    (root / "install-fails").touch()
    result = subprocess.run(["bash", str(launcher)], input="\n",
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    assert "Install failed" in result.stdout
    assert not (root / "launch-args").exists()
    (root / "install-fails").unlink()
    result = subprocess.run(["bash", str(launcher)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert (root / "launch-args").exists()


def test_source_archive_contains_installation_inputs(tmp_path):
    for name in ("pyproject.toml", "README.md", "MANIFEST.in", "requirements.txt",
                 "constraints.txt", "constraints-intel-mac.txt", "app.py", "CONTRIBUTING.md",
                 "Install_SNOM_Calculator_macOS.command", "Install_SNOM_Calculator_Windows.bat",
                 "Open_SNOM_Calculator_macOS.command", "Open_SNOM_Calculator_Windows.bat",
                 "Update_Material_Library_macOS.command", "Update_Material_Library_Windows.bat"):
        shutil.copy2(ROOT / name, tmp_path / name)
    shutil.copytree(ROOT / "src", tmp_path / "src",
                    ignore=shutil.ignore_patterns("*.egg-info", "__pycache__"))
    result = subprocess.run([sys.executable, "-c",
                             "from setuptools.build_meta import build_sdist; build_sdist('dist')"],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    with tarfile.open(next((tmp_path / "dist").glob("*.tar.gz"))) as archive:
        names = {Path(name).name for name in archive.getnames()}
    assert {"requirements.txt", "constraints.txt", "constraints-intel-mac.txt", "Install_SNOM_Calculator_macOS.command",
            "Install_SNOM_Calculator_Windows.bat", "Open_SNOM_Calculator_macOS.command",
            "Open_SNOM_Calculator_Windows.bat", "Update_Material_Library_macOS.command",
            "Update_Material_Library_Windows.bat"} <= names


@pytest.mark.parametrize("state", ["missing", "ready", "download-fails"])
def test_material_update_reports_result_without_launching_app(tmp_path, state):
    if os.name == "nt":
        pytest.skip("Bash updater requires a POSIX host")
    root = tmp_path / "install with spaces"
    root.mkdir()
    script = root / "Update_Material_Library_macOS.command"
    shutil.copy2(ROOT / script.name, script)
    if state != "missing":
        executable(root / ".venv/bin/python", "#!/bin/bash\nexit 0\n")
        executable(root / ".uv/uv", f'''#!/bin/bash
printf '%s\\n' "$@" > update-args
exit {1 if state == "download-fails" else 0}
''')
    result = subprocess.run(["bash", str(script)], cwd=tmp_path, input="\n",
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == (0 if state == "ready" else 1)
    if state == "missing":
        assert "Install_SNOM_Calculator_macOS.command" in result.stdout
        assert not (root / "update-args").exists()
    else:
        args = (root / "update-args").read_text().splitlines()
        assert "--no-deps" in args
        assert "https://github.com/vicentini-edoardo/Materials_Library/archive/HEAD.zip" in args
        assert ("Restart s-SNOM Spectrum Calculator" in result.stdout) == (state == "ready")


def test_launcher_survives_folder_move(launcher):
    result = subprocess.run(["bash", str(launcher)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    moved = launcher.parent.with_name("moved calculator with spaces")
    launcher.parent.rename(moved)
    result = subprocess.run(["bash", str(moved / "Open_SNOM_Calculator_macOS.command")],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    assert "Install_SNOM_Calculator_macOS.command" in result.stdout
    result = subprocess.run(["bash", str(moved / launcher.name)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert (moved / "library-path").read_text().strip() == str(moved / "custom_material_library.json")
    assert (moved / ".venv/snom-install-path").read_text().strip() == str(moved)
    assert not list((moved / "Desktop").iterdir())


def test_installers_do_not_generate_launchers_or_desktop_shortcuts():
    for ext in ("macOS.command", "Windows.bat"):
        source = (ROOT / f"Install_SNOM_Calculator_{ext}").read_text()
        assert "Desktop" not in source
        assert "Open_SNOM_Calculator" not in source
        assert (ROOT / f"Open_SNOM_Calculator_{ext}").exists()


def test_open_launcher_only_starts_streamlit(launcher):
    root = launcher.parent
    executable(root / ".venv/bin/python", (root / "fake-python").read_text())
    (root / ".uv/uv").unlink()
    result = subprocess.run(["bash", str(root / "Open_SNOM_Calculator_macOS.command")],
                            cwd=root.parent, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert (root / "launch-args").read_text().splitlines() == [
        "-m", "streamlit", "run", "app.py", "--server.address", "127.0.0.1",
    ]
    assert (root / "library-path").read_text().strip() == str(root / "custom_material_library.json")
    assert not (root / ".venv/installed").exists()


def test_open_launcher_missing_environment_does_not_install(launcher):
    result = subprocess.run(["bash", str(launcher.parent / "Open_SNOM_Calculator_macOS.command")],
                            input="\n", capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    assert "Install_SNOM_Calculator_macOS.command" in result.stdout
    assert not (launcher.parent / ".venv").exists()
