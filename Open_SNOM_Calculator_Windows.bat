@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto not_installed
if not exist ".venv\snom-install-path" goto ready
set /p SNOM_INSTALL_PATH=<.venv\snom-install-path
if /i not "%SNOM_INSTALL_PATH%"=="%CD%" goto not_installed
:ready
set SNOM_NATIVE_SAVE_DIALOG=1
set "SNOM_MATERIAL_LIBRARY_FILE=%CD%\custom_material_library.json"
.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
set "SNOM_EXIT_CODE=%ERRORLEVEL%"
pause
exit /b %SNOM_EXIT_CODE%
:not_installed
echo Run Install_SNOM_Calculator_Windows.bat first to install or repair the environment.
pause
exit /b 1
