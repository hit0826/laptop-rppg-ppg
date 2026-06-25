@echo off
setlocal
cd /d "%~dp0"

echo.
echo Laptop rPPG / PPG Vital Signs
echo 1. rPPG face measurement  - camera 0, 60 seconds
echo 2. PPG finger measurement - auto camera, 45 seconds
echo 3. Synthetic demo / tests
echo.
set /p choice=Select 1, 2, or 3: 

if "%choice%"=="1" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0rppg_camera_measure.ps1" -Camera 0 -Duration 60
) else if "%choice%"=="2" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0ppg_camera_measure.ps1" -Camera auto -Duration 45
) else if "%choice%"=="3" (
    py -m unittest discover -s tests
    if errorlevel 1 exit /b 1
    py run_synthetic_demo.py
    if errorlevel 1 exit /b 1
) else (
    echo Invalid selection.
    exit /b 1
)

endlocal
