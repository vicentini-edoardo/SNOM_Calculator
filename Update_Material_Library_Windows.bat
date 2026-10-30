@echo off
rem Update only the GitHub material catalog in the calculator's environment.
setlocal
cd /d "%~dp0" || exit /b 1
if not exist ".venv\Scripts\python.exe" (
  echo Run Install_SNOM_Calculator_Windows.bat first to install the app.
  pause
  exit /b 1
)
echo Close the running calculator before updating.
echo Downloading the latest Materials Library from GitHub...
set "CATALOG_URL=https://github.com/vicentini-edoardo/Materials_Library/archive/HEAD.zip"
if exist ".uv\uv.exe" (
  ".uv\uv.exe" pip install --python .venv\Scripts\python.exe --no-deps --no-cache --reinstall-package materials-library "%CATALOG_URL%"
) else (
  .venv\Scripts\python.exe -m pip install --no-deps --no-cache-dir --force-reinstall "%CATALOG_URL%"
)
if errorlevel 1 (
  echo Update failed. Check your internet connection and installer, then retry.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -c "from snom_calculator.materials import catalog_names; print(f'Catalog available: {len(catalog_names())} materials.')"
if errorlevel 1 (
  echo Catalog check failed. See docs\install-guide.md to restore the tested version.
  pause
  exit /b 1
)
echo Restart s-SNOM Spectrum Calculator to use the updated materials.
pause
exit /b 0
