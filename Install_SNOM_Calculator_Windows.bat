@echo off
rem Double-click launcher: first run installs everything (no Python/Git needed), then starts the app.
setlocal
cd /d "%~dp0" || exit /b 1
if not exist ".uv\uv.exe" (
  set "UV_UNMANAGED_INSTALL=%CD%\.uv"
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12; try { $installer = irm https://astral.sh/uv/install.ps1 } catch { Write-Host 'Trying the official GitHub installer...'; $installer = irm https://github.com/astral-sh/uv/releases/latest/download/uv-installer.ps1 }; $installer | iex"
  if not exist ".uv\uv.exe" ( echo Download failed. Check network/proxy access to astral.sh and github.com. See docs\install-guide.md for manual installation. & pause & exit /b 1 )
)
set "MOVED=0"
set "SNOM_PREVIOUS_DIR="
if exist ".venv\snom-install-path" (
  set /p "SNOM_PREVIOUS_DIR="<".venv\snom-install-path"
)
if defined SNOM_PREVIOUS_DIR if /i not "%SNOM_PREVIOUS_DIR%"=="%CD%" set "MOVED=1"
if "%MOVED%"=="1" goto install
.venv\Scripts\python.exe -c "import sys; assert (3, 11) <= sys.version_info[:2] < (3, 14); import snom_calculator.app" >nul 2>&1
if errorlevel 1 goto install
".uv\uv.exe" pip check --python .venv\Scripts\python.exe >nul 2>&1
if not errorlevel 1 goto ready

:install
echo Installing or repairing the environment ^(a few minutes^)...
.venv\Scripts\python.exe -c "import sys; assert (3, 11) <= sys.version_info[:2] < (3, 14)" >nul 2>&1
if "%MOVED%"=="0" if not errorlevel 1 goto dependencies
".uv\uv.exe" venv --clear --no-python-downloads --python ">=3.11,<3.14" .venv || ".uv\uv.exe" venv --clear --python 3.13 .venv
if errorlevel 1 goto install_failed

:dependencies
".uv\uv.exe" pip install --python .venv\Scripts\python.exe -c constraints.txt -e .
if errorlevel 1 goto install_failed

:ready
.venv\Scripts\python.exe -c "import os; from pathlib import Path; Path('.venv/snom-install-path').write_text(os.getcwd() + '\n')"
if errorlevel 1 goto install_failed
set SNOM_NATIVE_SAVE_DIALOG=1
set "SNOM_MATERIAL_LIBRARY_FILE=%CD%\custom_material_library.json"
.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
set "SNOM_EXIT_CODE=%ERRORLEVEL%"
pause
exit /b %SNOM_EXIT_CODE%

:install_failed
echo Install failed. Run this launcher again to retry.
pause
exit /b 1
