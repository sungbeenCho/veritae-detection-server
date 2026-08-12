# veritae-detection-server

Veritae 프로젝트의 AI 생성물 탐지 서버 (이미지, SPAI 모델). Spring 서버(veritae-server)의 오케스트레이션 요청을 받아 이미지 파일의 AI 생성 확률 점수를 반환한다.

이 서버는 3060Ti가 달린 별도 데스크탑에서 상시 구동된다. 아래 절차는 **그 데스크탑에 아무것도 설치되어 있지 않다는 전제**로, 처음부터 끝까지 순서대로 따라가면 된다. 명령어는 PowerShell 기준.

## 왜 conda 환경이 두 개인지

- `spai` 환경: SPAI 모델 실행 전용 (PyTorch, CUDA 필요, 무거움)
- `detection-api` 환경: 이 FastAPI 서버 실행 전용 (FastAPI, uvicorn만 필요, 가벼움)

이 서버는 SPAI를 파이썬에서 직접 import하지 않고, **별도 프로세스로 실행**시킨 뒤 결과 CSV를 읽어온다 (SPAI가 라이브러리가 아니라 CLI 도구로 설계되어 있기 때문). 그래서 두 환경을 완전히 분리해도 되고, 오히려 분리하는 게 의존성 충돌을 피하는 데 유리하다.

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

## 2. Miniconda 설치 (없다면)

```powershell
winget install --id Anaconda.Miniconda3 -e --source winget
```

설치 후 PowerShell 새로 열기. `conda --version`으로 확인.

## 3. 레포 두 개 clone

작업 폴더를 하나 정한다 (예: `C:\ai`).

```powershell
mkdir C:\ai
cd C:\ai
git clone https://github.com/sungbeenCho/veritae-detection-server.git
git clone https://github.com/mever-team/spai.git
```

## 4. `spai` conda 환경 구성 (SPAI 모델 실행용)

```powershell
conda create -n spai python=3.11 -y
conda activate spai
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
cd C:\ai\spai
pip install -r requirements.txt
```

시간이 꽤 걸린다 (PyTorch + CUDA 다운로드).

## 5. SPAI 모델 가중치 다운로드

1. [Google Drive 링크](https://drive.google.com/file/d/1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI/view?usp=sharing)에서 가중치 파일 다운로드
2. `C:\ai\spai\weights\` 폴더를 만들고 그 안에 다운로드한 파일을 넣는다 (파일명은 `spai.pth`로 맞추면 아래 기본값과 그대로 맞는다)

```powershell
mkdir C:\ai\spai\weights
# 다운로드한 파일을 C:\ai\spai\weights\spai.pth 로 이동
```

## 6. `spai` 환경의 python.exe 절대경로 확인

아래 환경변수 설정에 필요하다.

```powershell
conda activate spai
(Get-Command python).Source
```

출력된 경로를 메모해둔다 (예: `C:\Users\<user>\miniconda3\envs\spai\python.exe`).

## 7. `detection-api` conda 환경 구성 (FastAPI 서버 실행용)

```powershell
conda deactivate
conda create -n detection-api python=3.11 -y
conda activate detection-api
cd C:\ai\veritae-detection-server
pip install -r requirements.txt
```

## 8. 환경변수 설정

이 서버가 SPAI를 어디서 어떻게 실행할지 알려주는 값들이다. PowerShell 세션마다 설정해야 하니, 매번 치기 귀찮으면 아래를 `C:\ai\veritae-detection-server\run.ps1` 같은 스크립트로 저장해두고 실행하면 편하다.

```powershell
$env:SPAI_REPO_DIR = "C:\ai\spai"
$env:SPAI_PYTHON   = "C:\Users\<user>\miniconda3\envs\spai\python.exe"   # 6번에서 확인한 경로
$env:SPAI_CFG      = "./configs/spai.yaml"
$env:SPAI_MODEL    = "./weights/spai.pth"
```

## 9. 서버 실행

```powershell
conda activate detection-api
cd C:\ai\veritae-detection-server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 10. 정상 동작 확인

새 PowerShell 창을 열고:

```powershell
curl http://localhost:8000/health
# {"status":"ok"} 가 나오면 서버는 정상

curl -F "file=@C:\path\to\test.jpg;type=image/jpeg" http://localhost:8000/process/image
# {"ai_detection":{"model":"spai","score":0.xx}} 가 나오면 SPAI 연동까지 정상
```

## 11. 홈 LAN에서 Spring이 접근할 수 있게 방화벽 허용

관리자 권한 PowerShell에서:

```powershell
New-NetFirewallRule -DisplayName "Veritae Detection Server" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```

## 12. 데스크탑의 로컬 IP 확인 (Spring 쪽 설정에 필요)

```powershell
ipconfig
```

`IPv4 주소` 값을 확인해서, Spring 서버(veritae-server)의 `application.properties`에 다음처럼 설정하면 된다 (Spring 쪽 연동 코드는 다음 작업에서 진행 예정):

```properties
detection.service.url=http://<위에서 확인한 IP>:8000
```

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
