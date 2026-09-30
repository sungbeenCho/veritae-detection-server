# tests/regression/README.md

가짜정보탐지 정확도/속도 회귀 테스트셋. pytest가 아니라 **데스크탑에서 수동으로** 돌린다 -
실제 위키 인덱스(build_wiki_index.py로 미리 구축)와 Ollama가 켜져 있어야 한다.

## 언제 돌리나

- 프롬프트(`scripts/misinfo_lib.py`의 `PROMPT_TEMPLATE`)를 바꿀 때
- `MISINFO_EVIDENCE_CHUNK_COUNT`, Ollama 모델 이름 등 설정값을 바꿀 때
- 위키 인덱스를 재구축한 뒤

## 실행

```powershell
conda activate text-extraction
cd C:\ai\veritae-detection-server\tests\regression
python run_regression.py --wiki-index C:\ai\veritae-detection-server\data\wiki_index.sqlite3 --ollama-url http://localhost:11434 --ollama-model qwen3.5:4b
```

## 판단 기준

- `cases_basic.json`(36건), `cases_hard.json`(16건) 각각 정확도와 "위험 오답"(참인데 거짓으로 판정) 건수를 출력한다.
- 위험 오답이 0이 아니면 배포/변경 전에 원인을 확인한다 - 2026-09-29 실측 기준 기대치는 두 세트 모두 위험 오답 0건이다.
