@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0rppg_camera_measure.ps1" -Camera 0 -Duration 60
endlocal
