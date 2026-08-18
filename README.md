# veritae-detection-server

Veritae 프로젝트의 AI 생성물 탐지 서버 (이미지, SPAI 모델). Spring 서버(veritae-server)의 오케스트레이션 요청을 받아 이미지 파일의 AI 생성 확률 점수를 반환한다.

이 서버는 3060Ti가 달린 별도 데스크탑에서 상시 구동된다. 아래 절차는 **그 데스크탑에 아무것도 설치되어 있지 않다는 전제**로, 처음부터 끝까지 순서대로 따라가면 된다. 명령어는 PowerShell 기준.

## 왜 conda 환경이 두 개인지

- `spai` 환경: SPAI 모델 실행 전용 (PyTorch, CUDA 필요, 무거움)
- `detection-api` 환경: 이 FastAPI 서버 실행 전용 (FastAPI, uvicorn만 필요, 가벼움)

이 서버는 SPAI를 파이썬에서 직접 import하지 않고, **별도 프로세스로 실행**시킨 뒤 결과 CSV를 읽어온다 (SPAI가 라이브러리가 아니라 CLI 도구로 설계되어 있기 때문). 그래서 두 환경을 완전히 분리해도 되고, 오히려 분리하는 게 의존성 충돌을 피하는 데 유리하다.

## 이 문서에서 쓰는 작업 폴더 경로

아래 모든 명령어는 작업 폴더를 **`C:\ai`로 고정**해서 작성했다. 그대로 복사해서 쓰면 된다. 다른 위치를 쓰고 싶다면, 이 문서에 나오는 `C:\ai`를 전부 그 경로로 바꿔서 실행하면 된다 (일부만 바꾸면 경로가 안 맞아서 에러 난다).

---

## 0. 사전 준비 확인

PowerShell을 열고 아래를 각각 실행해서 이미 설치된 게 있는지 확인한다. `찾을 수 없음` 계열 에러가 나오면 다음 단계에서 설치하면 된다.

```powershell
git --version
conda --version
nvidia-smi
```

`nvidia-smi`는 3060Ti 드라이버가 정상 설치되어 있는지 확인하는 용도다. 여기서 GPU 정보가 안 뜨면 [NVIDIA 드라이버](https://www.nvidia.com/Download/index.aspx)부터 설치해야 한다.

## 1. Git 설치 (없다면)

```powershell
winget install --id Git.Git -e --source winget
```

설치 후 PowerShell 새로 열기.

## 2. VS Code 설치 (선택, 추천)

필수는 아니다 — 이후 모든 단계는 PowerShell만으로 그대로 진행 가능하다. 다만 나중에 버그 수정 등으로 코드를 직접 열어보거나, 서버를 켜고 끄는 걸 터미널 창 여러 개 대신 IDE 하나로 편하게 하고 싶으면 깔아두면 좋다.

```powershell
winget install --id Microsoft.VisualStudioCode -e --source winget
```

설치 후에는 `code C:\ai\veritae-detection-server` 로 폴더를 열거나, VS Code에서 `파일 > 폴더 열기`로 열면 된다. 이후 나오는 PowerShell 명령어들은 VS Code 하단의 **통합 터미널**(`Ctrl+\``)에 그대로 쳐도 되고, 지금처럼 별도 PowerShell 창에 쳐도 된다 — 결과는 동일하다. VS Code는 명령어를 바꾸지 않고, 그냥 그 명령어를 칠 곳과 파일을 볼 곳을 하나로 묶어줄 뿐이다.

## 3. Miniconda 설치 (없다면)

```powershell
winget install --id Anaconda.Miniconda3 -e --source winget
```

설치 후 PowerShell 새로 열기. `conda --version`으로 확인.

## 4. 작업 폴더 만들고 레포 두 개 clone

```powershell
mkdir C:\ai
cd C:\ai
git clone https://github.com/sungbeenCho/veritae-detection-server.git
git clone https://github.com/mever-team/spai.git
```

## 5. `spai` conda 환경 구성 (SPAI 모델 실행용)

```powershell
conda create -n spai python=3.11 -y
conda activate spai
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
cd C:\ai\spai
pip install -r requirements.txt
```

시간이 꽤 걸린다 (PyTorch + CUDA 다운로드).

## 6. SPAI 모델 가중치 다운로드

1. [Google Drive 링크](https://drive.google.com/file/d/1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI/view?usp=sharing)에서 가중치 파일 다운로드
2. `C:\ai\spai\weights\` 폴더를 만들고 그 안에 다운로드한 파일을 넣는다 (파일명은 `spai.pth`로 맞추면 아래 기본값과 그대로 맞는다)

```powershell
mkdir C:\ai\spai\weights
# 다운로드한 파일을 C:\ai\spai\weights\spai.pth 로 이동
```

## 7. `spai` 환경의 python.exe 절대경로 확인

아래 환경변수 설정에 필요하다.

```powershell
conda activate spai
(Get-Command python).Source
```

출력된 경로를 메모해둔다 (예: `C:\Users\<user>\miniconda3\envs\spai\python.exe`).

## 8. `detection-api` conda 환경 구성 (FastAPI 서버 실행용)

```powershell
conda deactivate
conda create -n detection-api python=3.11 -y
conda activate detection-api
cd C:\ai\veritae-detection-server
pip install -r requirements.txt
```

## 9. 환경변수 설정

이 서버가 SPAI를 어디서 어떻게 실행할지 알려주는 값들이다. PowerShell 세션마다 설정해야 하니, 매번 치기 귀찮으면 아래를 `C:\ai\veritae-detection-server\run.ps1` 같은 스크립트로 저장해두고 실행하면 편하다.

```powershell
$env:SPAI_REPO_DIR = "C:\ai\spai"
$env:SPAI_PYTHON   = "C:\Users\<user>\miniconda3\envs\spai\python.exe"   # 7번에서 확인한 경로
$env:SPAI_CFG      = "./configs/spai.yaml"
$env:SPAI_MODEL    = "./weights/spai.pth"
$env:SPAI_RESIZE_TO = "1024"   # 안 정하면 기본값 1024. 8GB급 GPU에서 폰 원본 사진(3000px+)을 그대로 돌리면 CUDA OOM 남
```

## 10. 서버 실행

PowerShell 창이든 VS Code 통합 터미널이든 상관없다.

```powershell
conda activate detection-api
cd C:\ai\veritae-detection-server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

끌 때는 이 터미널에서 `Ctrl+C`.

## 11. 정상 동작 확인

새 PowerShell 창을 열고 (서버는 10번에서 계속 켜둔 채로):

```powershell
curl http://localhost:8000/health
# {"status":"ok"} 가 나오면 서버는 정상

curl -F "file=@C:\path\to\test.jpg;type=image/jpeg" http://localhost:8000/process/image
# {"ai_detection":{"model":"spai","score":0.xx}} 가 나오면 SPAI 연동까지 정상
```

## 12. 홈 LAN에서 Spring이 접근할 수 있게 방화벽 허용

관리자 권한 PowerShell에서:

```powershell
New-NetFirewallRule -DisplayName "Veritae Detection Server" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```

## 13. 데스크탑의 로컬 IP 확인 (Spring 쪽 설정에 필요)

```powershell
ipconfig
```

`IPv4 주소` 값을 확인해서, Spring 서버(veritae-server)의 `application.properties`에 다음처럼 설정하면 된다 (Spring 쪽 연동 코드는 다음 작업에서 진행 예정):

```properties
detection.service.url=http://<위에서 확인한 IP>:8000
```

---

## 음성(AntiDeepfake) 셋업

이 서버는 이미지뿐만 아니라 음성 파일의 AI 생성(deepfake) 여부도 탐지할 수 있다. SPAI와 달리 AntiDeepfake는 CPU에서 충분히 빠르게 동작하므로, 별도의 GPU 설정 없이 설치할 수 있다.

### 1. `antideepfake` conda 환경 구성

```powershell
conda create -n antideepfake python==3.9.0 -y
conda activate antideepfake
pip install torch==2.6.0 torchaudio==2.6.0
```

**중요:** SPAI와 달리 `--index-url`을 붙이지 않는다. CPU 전용 빌드이므로 PyTorch의 기본 pip 버전으로 충분하다.

시간이 꽤 걸린다 (PyTorch 다운로드).

### 2. fairseq 특정 커밋 체크아웃 및 editable 설치

```powershell
cd C:\ai
git clone https://github.com/pytorch/fairseq.git
cd fairseq
git checkout 862efab86f649c04ea31545ce28d13c59560113d
pip install -e .
```

### 3. 나머지 패키지 설치

```powershell
pip install librosa scikit-learn julius soundfile h5py
```

### 4. AntiDeepfake 저장소 클론

```powershell
cd C:\ai
git clone https://github.com/nii-yamagishilab/AntiDeepfake.git
```

### 5. 모델 체크포인트 다운로드

```powershell
mkdir C:\ai\AntiDeepfake\downloads
cd C:\ai\AntiDeepfake\downloads
wget -O mms_300m.ckpt https://zenodo.org/records/15580543/files/mms_300m.ckpt
```

Windows에 wget이 없으면 PowerShell의 `curl`을 사용할 수 있다:

```powershell
curl -o mms_300m.ckpt https://zenodo.org/records/15580543/files/mms_300m.ckpt
```

### 6. `antideepfake` 환경의 python.exe 절대경로 확인

아래 환경변수 설정에 필요하다.

```powershell
conda activate antideepfake
(Get-Command python).Source
```

출력된 경로를 메모해둔다 (예: `C:\Users\<user>\miniconda3\envs\antideepfake\python.exe`).

### 7. veritae-detection-server 최신 코드 적용

```powershell
cd C:\ai\veritae-detection-server
git pull origin main
```

### 8. 환경변수 설정

이 서버가 AntiDeepfake를 어디서 어떻게 실행할지 알려주는 값들이다. PowerShell 세션마다 설정해야 하니, 매번 치기 귀찮으면 아래를 `C:\ai\veritae-detection-server\run.ps1` 같은 스크립트에 추가해서 실행하면 편하다.

```powershell
$env:ANTIDEEPFAKE_REPO_DIR       = "C:\ai\AntiDeepfake"
$env:ANTIDEEPFAKE_CHECKPOINT     = "C:\ai\AntiDeepfake\downloads\mms_300m.ckpt"
$env:ANTIDEEPFAKE_PYTHON         = "C:\Users\<user>\miniconda3\envs\antideepfake\python.exe"   # 위에서 확인한 경로
```

### 9. 서버 실행 (또는 재시작)

이미 detection-api 서버가 켜져 있다면, 환경변수가 적용되도록 다시 시작해야 한다.

```powershell
conda activate detection-api
cd C:\ai\veritae-detection-server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 10. 정상 동작 확인

새 PowerShell 창을 열고 (서버는 9번에서 계속 켜둔 채로):

```powershell
curl -X POST -F "file=@C:\path\to\test.wav" http://localhost:8000/process/audio
```

정상이면 다음과 같은 응답이 나온다:

```json
{
  "ai_detection": {
    "model": "antideepfake",
    "score": 0.xx,
    "evidence": []
  }
}
```

score가 높으면(AI 생성 확률이 높으면) `evidence` 배열에 탐지된 근거들(의심 구간의 시작/종료, 설명)이 채워진다.

**이 단계에서 주의:** 이 curl 호출이 실제로 작동하는 것이 매우 중요하다. 추론 스크립트를 작성할 때는 실제 체크포인트 없이 소스코드만 읽고 다음 세 지점을 추정했기 때문에, 이것이 그 추정들을 실제로 검증하는 첫 번째 시험대다. 만약 에러가 나거나 score가 항상 0 또는 1처럼 이상하면, 다음 세 지점을 의심해야 한다:
- fairseq의 frame stride 설정이 실제 체크포인트와 맞는가
- AntiDeepfake 코드의 `FAKE_CLASS_INDEX` 값이 맞는가
- `load_weights` 함수의 반환 방식이 예상대로 동작하는가

자세한 내용은 프로젝트 문서의 음성 AI 판독 설계 섹션을 참고하고, 필요하면 추론 스크립트의 로그 출력을 추가해서 각 단계를 점검하자.

---

## 로컬 개발 (테스트 실행)

이 레포를 수정하는 개발 머신(이 컴퓨터)에서 테스트를 돌릴 때는 SPAI 없이도 가능하다 (테스트는 `run_spai_inference`를 mock 처리한다):

```powershell
pip install -r requirements-dev.txt
pytest
```

## API

### `GET /health`
헬스체크. `{"status": "ok"}` 반환.

### `POST /process/image`
`multipart/form-data`로 이미지 파일(`file` 필드, jpeg/png/webp)을 받아 AI 생성 확률 점수를 반환.

```json
{
  "ai_detection": { "model": "spai", "score": 0.87 }
}
```
