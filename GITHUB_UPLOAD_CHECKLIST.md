# GitHub Upload Checklist

Upload the contents of this folder as the repository root.

## Include

- `README.md`
- `LICENSE`
- `.gitignore`
- `requirements.txt`
- `pyproject.toml`
- `run.bat`
- `run_rppg.bat`
- `run_ppg.bat`
- `*.ps1` launcher scripts
- `*.py` command scripts
- `src/`
- `tests/`

## Exclude

- `.venv/`, `venv/`, and other local Python environments
- `__pycache__/` and `.pytest_cache/`
- `sessions/` measurement outputs
- personal CSV/JSON/PNG captures
- large video files
- local editor folders such as `.vscode/` and `.idea/`

## Final Local Check

```powershell
cd "C:\Users\h0208\OneDrive\바탕 화면\rppg"
py -m pip install -r requirements.txt
py -m unittest discover -s tests
py run_synthetic_demo.py
```

Camera measurements:

```powershell
.\rppg_camera_measure.ps1 -Camera 0 -Duration 60
.\ppg_camera_measure.ps1 -Camera auto -Duration 45
```
