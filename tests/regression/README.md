# tests/regression/README.md

가짜정보탐지 정확도/속도 회귀 테스트셋. pytest가 아니라 **데스크탑에서 수동으로** 돌린다 -
실제 위키 인덱스(build_wiki_index.py로 미리 구축), 반박 확인용 NLI 모델(train_nli.py로 미리 학습,
기본 위치 `data\nli_model`, 다른 곳이면 `--nli-model`로 지정), Ollama가 모두 준비돼 있어야 한다.

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
  재정렬해 Ollama에 판정을 맡기는, 실제 파이프라인 그대로 돌리는 세트다. 2026-09-29 실측 16/20,
  2026-09-30 실측 19/20(거짓 11건 중 아폴로 1건만 놓침).
- **`cases_grounding.json`(14건, 기본 실행 대상 = 머지 게이트)**: 2026-10-01 실제 분석에서
  "라벨은 반박인데 근거는 속설을 뒷받침"한 사례를 계기로 추가했다. 전해 들은 말("~라고
  들었어요", 안의 내용으로 판정), OCR 오타, 속설을 속설이라고 소개만 하는 참인 문장("~라는
  속설이 있다" - 반박되면 안 됨), 근거 두 개를 이어 봐야 하는 문장을 담는다.
- **`cases_basic.json`(36건), `cases_hard.json`(16건, 참고용 - 머지 게이트 아님)**: 원래 NLI
  분류기 실험용 데이터로, 문장마다 근거 문단(premise)이 함께 주어져 있었다. 이 브랜치로 옮겨
  오면서 `sentence`만 남고 premise가 빠졌는데, 연예인 사망설/"이 약은…"/"A시에…" 같은 문항은
  근거 문단이 있어야 성립해서 실제 위키 검색으로는 재현이 안 된다 - 즉 36/36·16/16 기준과 지금
  비교할 수 없다(2026-09-30 리뷰에서 확인, 컨트롤러 판단). premise를 살린 NLI 전용 모드를 새로
  만드는 건 범위 밖(YAGNI)으로 보류했다 - 지금은 `--include-nli-legacy` 플래그를 줬을 때만
  참고용으로 같이 돈다.

## 판단 기준

- 각 그룹이 정확도, "위험 오답"(참인데 거짓으로 판정) 건수, 가장 느린 문장의 시간을 출력한다.
- 반박으로 나온 건마다 이유와 인용된 근거 원문이 함께 출력된다. **라벨만 보지 말고 근거가
  실제로 주장을 부정하는 내용인지 한 건씩 읽어서 확인한다.**
- 통과 조건: `cases_e2e.json`과 `cases_grounding.json` 모두 위험 오답 0건, 그리고
  `cases_e2e.json`에서 직전 실측에 잡던 거짓 문장을 하나도 놓치지 않을 것.
- `cases_basic.json`/`cases_hard.json`(참고용)의 결과는 지금 파이프라인과 직접 비교할 수
  없으니 정확도 숫자 자체보다는 "터지지 않고 도는지" 정도만 참고한다.
