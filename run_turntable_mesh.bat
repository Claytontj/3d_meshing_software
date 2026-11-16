@echo off
setlocal
set "PYTHONPATH=%~dp0src"
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" -m turntable_mesh --config "%~dp0config.toml"
) else (
    py -3 -m turntable_mesh --config "%~dp0config.toml"
)
endlocal
