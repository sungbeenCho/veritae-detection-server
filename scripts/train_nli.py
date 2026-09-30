# scripts/train_nli.py
"""가짜정보 반박 확인용 NLI 분류기 학습 - klue/roberta-large를 KLUE-NLI로 미세조정한다.

라이선스가 명시된 한국어 NLI 분류기가 없어(2026-10-01 조사: HuggingFace의 KLUE-NLI 모델 대부분이 라이선스
표기 없음) 직접 학습한다. 원재료인 KLUE 데이터셋과 KLUE-RoBERTa 모델은 모두 CC BY-SA 4.0이라(KLUE 공식 저장소
github.com/KLUE-benchmark/KLUE의 License 항목 - HuggingFace 모델 카드에는 표기가 없다), 학습 결과도
CC BY-SA 4.0을 따른다(출처 표기 필요 - README 참고). 같은 방법으로 학습된 공개 모델(라이선스 없음)로 노트북에서
미리 시험했을 때 실제 위키 문단에서 진짜 반박 문장과 아닌 문장을 가장 잘 구분했다(nli_check.py 참고).

GPU가 필요하다(3060Ti 8GB 기준 약 1~2시간). CUDA용 PyTorch가 든 별도 conda 환경에서 실행한다 - README 참고.
사용: python train_nli.py --output C:\\ai\\veritae-detection-server\\data\\nli_model
"""
import argparse
import inspect
import math
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "klue/roberta-large"
# KLUE-NLI 라벨 순서. 저장되는 모델 설정에 이름을 넣어 두면 nli_check.py가 모순 라벨을 이름으로 찾는다.
LABELS = ["entailment", "neutral", "contradiction"]
TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    # 아래 두 옵션은 스크립트가 끝까지 도는지 작은 모델로 빨리 확인할 때만 쓴다(실사용 모델은 기본값으로 학습).
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--max-samples", type=int, default=0, help="0=전체")
    args = parser.parse_args()

    dataset = load_dataset("klue/klue", "nli")
    if args.max_samples:
        for split in dataset:
            dataset[split] = dataset[split].select(range(min(args.max_samples, len(dataset[split]))))
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch):
        # KLUE 토크나이저는 두 문장 구분용 token_type_ids(0/1)를 만들지만 RoBERTa는 한 종류(0)만 받아서
        # 그대로 넘기면 학습 시작 즉시 IndexError로 멈춘다(2026-10-01 노트북 사전 검증). nli_check.py도 뺀다.
        return tokenizer(
            batch["premise"], batch["hypothesis"], truncation=True, max_length=128, return_token_type_ids=False,
        )

    encoded = dataset.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)},
    )

    def accuracy(eval_pred):
        logits, labels = eval_pred
        return {"accuracy": float((np.argmax(logits, axis=-1) == labels).mean())}

    steps_per_epoch = math.ceil(len(encoded["train"]) / (TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION))
    warmup_steps = int(steps_per_epoch * args.epochs * 0.1)

    # 평가 주기 옵션 이름이 transformers 4.41에서 evaluation_strategy -> eval_strategy로 바뀌었다.
    # 판정 환경 버전에 맞춰 설치하므로 어느 쪽이든 돌게 한다.
    eval_option = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments).parameters else "evaluation_strategy"

    training_args = TrainingArguments(
        **{eval_option: "epoch"},
        output_dir=str(args.output / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        # 8GB GPU에 large 모델을 올리려면 작은 배치 + 누적 + 체크포인팅 + fp16이 필요하다(실질 배치 32).
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        per_device_eval_batch_size=16,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        # warmup_ratio는 transformers 5에서 없어져(2026-10-01 노트북 사전 검증) 스텝 수로 직접 넘긴다.
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=100,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=accuracy,
    )
    trainer.train()

    metrics = trainer.evaluate()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print(f"완료: KLUE-NLI 검증 정확도 {metrics['eval_accuracy']:.4f} -> {args.output}")


if __name__ == "__main__":
    main()
