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

## 데이터셋

- **`cases_e2e.json`(20건, 기본 실행 대상 = 머지 게이트)**: 실제 위키 검색으로 후보를 찾고 e5로
  재정렬해 Ollama에 판정을 맡기는, 실제 파이프라인 그대로 돌리는 유일한 세트다. 2026-09-29
  실측 기준 16/20.
- **`cases_basic.json`(36건), `cases_hard.json`(16건, 참고용 - 머지 게이트 아님)**: 원래 NLI
  분류기 실험용 데이터로, 문장마다 근거 문단(premise)이 함께 주어져 있었다. 이 브랜치로 옮겨
  오면서 `sentence`만 남고 premise가 빠졌는데, 연예인 사망설/"이 약은…"/"A시에…" 같은 문항은
  근거 문단이 있어야 성립해서 실제 위키 검색으로는 재현이 안 된다 - 즉 36/36·16/16 기준과 지금
  비교할 수 없다(2026-09-30 리뷰에서 확인, 컨트롤러 판단). premise를 살린 NLI 전용 모드를 새로
  만드는 건 범위 밖(YAGNI)으로 보류했다 - 지금은 `--include-nli-legacy` 플래그를 줬을 때만
  참고용으로 같이 돈다.

## 판단 기준

- 각 그룹이 정확도와 "위험 오답"(참인데 거짓으로 판정) 건수를 출력한다.
- `cases_e2e.json`에서 위험 오답이 0이 아니면 배포/변경 전에 원인을 확인한다.
- `cases_basic.json`/`cases_hard.json`(참고용)의 결과는 지금 파이프라인과 직접 비교할 수
  없으니 정확도 숫자 자체보다는 "터지지 않고 도는지" 정도만 참고한다.
