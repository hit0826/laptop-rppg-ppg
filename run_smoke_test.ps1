Set-Location $PSScriptRoot
py -m compileall -q src tests run_app.py run_synthetic_demo.py probe_camera.py analyze_session.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
py -m unittest discover -s tests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
py run_synthetic_demo.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
py probe_camera.py --camera auto
exit $LASTEXITCODE
