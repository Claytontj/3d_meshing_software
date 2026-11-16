@echo off
setlocal
set "PYTHONPATH=%~dp0src"
"C:\Users\clayt\miniforge3\envs\3d_mesh_app\python.exe" -m turntable_mesh --config "%~dp0config.toml"
endlocal
