# scripts/train_nli.py
"""가짜정보 반박 확인용 NLI 분류기 학습 - klue/roberta-large를 KLUE-NLI로 미세조정한다.

라이선스가 명시된 한국어 NLI 분류기가 없어(2026-10-01 조사: HuggingFace의 KLUE-NLI 모델 대부분이 라이선스
표기 없음) 직접 학습한다. 원재료인 KLUE 데이터셋과 KLUE-RoBERTa 모델은 모두 CC BY-SA 4.0이라(KLUE 공식 저장소
github.com/KLUE-benchmark/KLUE의 License 항목 - HuggingFace 모델 카드에는 표기가 없다), 학습 결과도
CC BY-SA 4.0을 따른다(출처 표기 필요 - README 참고). 같은 방법으로 학습된 공개 모델(라이선스 없음)로 노트북에서
미리 시험했을 때 실제 위키 문단에서 진짜 반박 문장과 아닌 문장을 가장 잘 구분했다(nli_check.py 참고).

GPU가 필요하다(3060Ti 8GB 기준 약 1~2시간). CUDA용 PyTorch가 든 별도 conda 환경에서 실행한다 - README 참고.
사용: python train_nli.py --output C:\\ai\\veritae-detection-server\\data\\nli_model

보충 학습(--init-model + --augment): 이미 학습한 모델에서 이어서, 위키가 속설을 반박할 때 쓰는 평가형 문장
("X라는 것은 사실이 아니다", "X는 근거 없는 미신이다")을 nli_augment.py로 만들어 원래 KLUE 데이터와 섞어 1번 더
학습한다. 원래 KLUE 실력이 떨어지지 않았는지와 학습에 안 쓴 문장 틀에서의 정확도를 학습 전후로 출력한다.
사용: python train_nli.py --init-model <기존 모델 폴더> --augment --epochs 1 --output <새 폴더>
"""
import argparse
import inspect
import math
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, Value, concatenate_datasets, load_dataset
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
COLUMNS = ["premise", "hypothesis", "label"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    # 아래 두 옵션은 스크립트가 끝까지 도는지 작은 모델로 빨리 확인할 때만 쓴다(실사용 모델은 기본값으로 학습).
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--max-samples", type=int, default=0, help="0=전체")
    parser.add_argument("--init-model", type=Path, help="이어서 학습할 기존 모델 폴더(보충 학습)")
    parser.add_argument("--augment", action="store_true", help="평가형 반박 문장 보충 예시를 섞는다(nli_augment.py)")
    parser.add_argument("--augment-sentences", type=int, default=4000, help="보충 예시를 만들 KLUE 문장 수")
    parser.add_argument("--replay", type=int, default=12000, help="보충 학습 때 같이 섞을 원래 KLUE 학습 예시 수")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from nli_augment import build_examples

    dataset = load_dataset("klue/klue", "nli")
    if args.max_samples:
        for split in dataset:
            dataset[split] = dataset[split].select(range(min(args.max_samples, len(dataset[split]))))
    dataset = {split: dataset[split].select_columns(COLUMNS).cast_column("label", Value("int64")) for split in dataset}

    model_source = str(args.init_model) if args.init_model else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(model_source)

    augment_eval = None
    if args.augment:
        train_sentences = list(dict.fromkeys(dataset["train"]["hypothesis"]))[: args.augment_sentences]
        augmented = Dataset.from_list(build_examples(train_sentences, seed=0)).cast_column("label", Value("int64"))
        train = dataset["train"].shuffle(seed=0)
        replay = train.select(range(min(args.replay, len(train)))) if args.init_model else train
        dataset["train"] = concatenate_datasets([augmented, replay]).shuffle(seed=0)
        held_out = build_examples(list(dict.fromkeys(dataset["validation"]["hypothesis"])), seed=1, held_out=True)
        augment_eval = Dataset.from_list(held_out).cast_column("label", Value("int64"))
        print(f"보충 예시 {len(augmented)}개 + 원래 KLUE {len(replay)}개로 학습, 평가용 보충 예시 {len(augment_eval)}개")

    def tokenize(batch):
        # KLUE 토크나이저는 두 문장 구분용 token_type_ids(0/1)를 만들지만 RoBERTa는 한 종류(0)만 받아서
        # 그대로 넘기면 학습 시작 즉시 IndexError로 멈춘다(2026-10-01 노트북 사전 검증). nli_check.py도 뺀다.
        return tokenizer(
            batch["premise"], batch["hypothesis"], truncation=True, max_length=128, return_token_type_ids=False,
        )

    encoded = {split: data.map(tokenize, batched=True) for split, data in dataset.items()}
    encoded_augment_eval = augment_eval.map(tokenize, batched=True) if augment_eval is not None else None

    model = AutoModelForSequenceClassification.from_pretrained(
        model_source,
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

    # 보충 학습은 1번만 돌고 끝에서 직접 평가하므로 중간 저장(large 모델은 옵티마이저 포함 약 4GB)을 하지 않는다.
    continuing = args.init_model is not None
    training_args = TrainingArguments(
        **{eval_option: "no" if continuing else "epoch"},
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
        save_strategy="no" if continuing else "epoch",
        save_total_limit=1,
        load_best_model_at_end=not continuing,
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
    def report(when: str) -> None:
        klue = trainer.evaluate()["eval_accuracy"]
        line = f"[{when}] KLUE-NLI 검증 정확도 {klue:.4f}"
        if encoded_augment_eval is not None:
            augment = trainer.evaluate(eval_dataset=encoded_augment_eval)["eval_accuracy"]
            line += f", 평가형 반박 문장(학습에 안 쓴 틀) 정확도 {augment:.4f}"
        print(line, flush=True)

    if continuing:
        report("학습 전")
    trainer.train()
    report("학습 후")

    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print(f"완료: 모델 저장 -> {args.output}")


if __name__ == "__main__":
    main()
