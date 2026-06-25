# camera-based laptop-rppg-ppg bio-signal measurement

## 한국어

노트북 웹캠 기반으로 심박수(BPM)를 실험적으로 측정하는 Python 프로젝트입니다.
자신의 bpm을 측정해봅시다

- `rPPG`: 얼굴 ROI의 색 변화로 비접촉 심박 신호를 추정합니다.
- `PPG`: 손가락을 웹캠에 올려 녹색 채널 기반 접촉식에 가까운 맥파 신호를 추정합니다.
- 측정 후 `sessions/` 폴더에 CSV, JSON 요약, PNG 그래프를 저장합니다.

### 포함된 파일

- `rppg_camera_measure.ps1`: rPPG 얼굴 측정 실행 스크립트
- `ppg_camera_measure.ps1`: PPG 손가락 측정 실행 스크립트
- `run.bat`: Windows에서 rPPG/PPG/테스트를 선택 실행하는 메뉴
- `run_rppg.bat`: rPPG 기본 측정 바로 실행
- `run_ppg.bat`: PPG 기본 측정 바로 실행
- `run_app.py`: 실시간 카메라 앱 진입점
- `probe_camera.py`: 사용 가능한 웹캠 자동 확인
- `analyze_session.py`: 저장된 측정 CSV 분석
- `plot_session.py`: 저장된 측정 CSV 그래프 생성
- `src/vitals_cam/`: 실제 rPPG/PPG 처리 코드
- `tests/`: 신호 처리 단위 테스트

### 필요한 환경

- Windows PC
- 웹캠
- Python 3.10 이상
- `numpy`, `scipy`, `opencv-python`, `mediapipe`, `matplotlib`

`mediapipe` 설치가 실패하면 Python 3.11 환경에서 다시 시도하는 것이 보통 가장 안정적입니다.

### 설치

```powershell
cd "C:\Users\h0208\OneDrive\바탕 화면\rppg"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

### 실행

사용자가 실제로 쓰는 기본 측정 명령입니다.

#### rPPG 얼굴 측정

```powershell
cd "C:\Users\h0208\OneDrive\바탕 화면\rppg"
.\rppg_camera_measure.ps1 -Camera 0 -Duration 60
```

#### PPG 손가락 측정

```powershell
cd "C:\Users\h0208\OneDrive\바탕 화면\rppg"
.\ppg_camera_measure.ps1 -Camera auto -Duration 45
```

Windows 배치 파일로 실행하려면:

```powershell
.\run.bat
```

개별 바로 실행:

```powershell
.\run_rppg.bat
.\run_ppg.bat
```

### 카메라 확인

```powershell
py probe_camera.py --camera auto
py probe_camera.py --camera 0
```

### 테스트

카메라 없이 실행 가능한 기본 검증:

```powershell
py -m unittest discover -s tests
py run_synthetic_demo.py
```

카메라까지 포함한 전체 스모크 테스트:

```powershell
.\run_smoke_test.ps1
```

### 측정 팁

rPPG 얼굴 측정:

- 얼굴이 프레임에 크게 보이게 앉습니다.
- 처음 20초 정도는 움직임을 줄입니다.
- 모니터 깜빡임만 의존하지 말고 안정적인 조명을 사용합니다.
- 강한 역광, 그림자, 큰 머리 움직임을 피합니다.

PPG 손가락 측정:

- 손가락 끝을 웹캠 위에 가볍게 올립니다.
- 너무 세게 누르지 않습니다.
- 신호 품질이 올라갈 때까지 손가락을 움직이지 않습니다.

### 결과 파일

측정 결과는 `sessions/`에 저장됩니다.

- `session-rppg-face-YYYYMMDD-HHMMSS.csv`
- `session-rppg-face-YYYYMMDD-HHMMSS.json`
- `session-rppg-face-YYYYMMDD-HHMMSS.png`
- `session-ppg-finger-YYYYMMDD-HHMMSS.csv`
- `session-ppg-finger-YYYYMMDD-HHMMSS.json`
- `session-ppg-finger-YYYYMMDD-HHMMSS.png`

개인 생체신호 데이터가 될 수 있으므로 `sessions/`는 `.gitignore`에 포함되어 있습니다.

### 폴더 구조

```text
.
├─ src/
│  └─ vitals_cam/
│     ├─ app.py
│     ├─ analysis.py
│     ├─ camera.py
│     ├─ roi.py
│     ├─ session.py
│     ├─ signal_processing.py
│     └─ synthetic.py
├─ tests/
│  └─ test_signal_processing.py
├─ .gitignore
├─ LICENSE
├─ README.md
├─ requirements.txt
├─ pyproject.toml
├─ run.bat
├─ run_rppg.bat
├─ run_ppg.bat
├─ rppg_camera_measure.ps1
└─ ppg_camera_measure.ps1
```




### Install

```powershell
cd "C:\Users\h0208\OneDrive\바탕 화면\rppg"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

### Run

rPPG face measurement:

```powershell
.\rppg_camera_measure.ps1 -Camera 0 -Duration 60
```

PPG fingertip measurement:

```powershell
.\ppg_camera_measure.ps1 -Camera auto -Duration 45
```

Quick Windows menu:

```powershell
.\run.bat
```

### Verify

```powershell
py -m unittest discover -s tests
py run_synthetic_demo.py
```

Use `GITHUB_UPLOAD_CHECKLIST.md` before uploading the repository.
