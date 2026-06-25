param(
    [string]$Camera = "auto",
    [int]$Duration = 45,
    [switch]$Preview,
    [switch]$NoPreview
)

Set-Location $PSScriptRoot
$SessionDir = Join-Path $PSScriptRoot "sessions"
$StartedAt = Get-Date
$ArgsList = @(
    "run_app.py",
    "--mode", "face",
    "--camera", $Camera,
    "--duration", $Duration,
    "--record",
    "--save-dir", $SessionDir
)
if ($NoPreview) { $ArgsList += "--headless" }

py @ArgsList
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$Latest = Get-ChildItem -LiteralPath $SessionDir -Filter "session-*.csv" -ErrorAction SilentlyContinue |
    Where-Object { $_.LastWriteTime -ge $StartedAt } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if ($null -ne $Latest) {
    py analyze_session.py --csv $Latest.FullName --mode face
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    py plot_session.py --csv $Latest.FullName --mode face
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    Write-Host "No new session CSV was saved. Check ROI visibility in the preview window, or use -NoPreview for headless capture."
    exit 4
}
