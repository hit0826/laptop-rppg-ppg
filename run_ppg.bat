@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0ppg_camera_measure.ps1" -Camera auto -Duration 45
endlocal
