param(
    [string]$Camera = "auto",
    [int]$Duration = 0,
    [switch]$Headless,
    [switch]$Record
)

Set-Location $PSScriptRoot
$ArgsList = @("run_app.py", "--mode", "finger", "--camera", $Camera)
if ($Duration -gt 0) { $ArgsList += @("--duration", $Duration) }
if ($Headless) { $ArgsList += "--headless" }
if ($Record) { $ArgsList += "--record" }
py @ArgsList
