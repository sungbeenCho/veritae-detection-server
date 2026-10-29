"""가짜정보 하위 프로세스(misinfo_infer.py)의 단계별 시간 기록 도구.
2026-10-13: 이미지 한 장(문장 2개)이 2분 걸렸는데 어느 단계 때문인지 알 방법이 없었다."""
import json
import logging
import sys
import types

import pytest

import misinfo_infer
import nli_check
from misinfo_timing import StepTimer, configure_script_logging, format_ollama_stats, logged_step


def test_step_timer_adds_up_repeated_steps_in_order():
    timer = StepTimer()
    with timer.step("search"):
        pass
    with timer.step("judge_llm"):
        pass
    with timer.step("search"):
        pass

    assert list(timer.seconds) == ["search", "judge_llm"]
    assert timer.summary().count("search=") == 1


def test_step_timer_records_step_even_when_it_raises():
    timer = StepTimer()
    with pytest.raises(ValueError):
        with timer.step("nli"):
            raise ValueError("실패")

    assert "nli" in timer.seconds


def test_step_timer_summary_formats_seconds_with_one_decimal():
    timer = StepTimer()
    timer.seconds.update({"search": 0.04, "judge_llm": 12.345})

    assert timer.summary() == "search=0.0s judge_llm=12.3s"


def test_format_ollama_stats_converts_nanoseconds_and_shows_token_counts():
    data = {
        "load_duration": 13_000_000_000,
        "prompt_eval_duration": 8_500_000_000, "prompt_eval_count": 2100,
        "eval_duration": 1_200_000_000, "eval_count": 60,
    }

    assert format_ollama_stats(data, wall_seconds=22.7) == (
        "wall=22.7s load=13.0s prompt_eval=8.5s(2100tok) eval=1.2s(60tok)"
    )


def test_format_ollama_stats_tolerates_missing_fields():
    assert format_ollama_stats({"response": "{}"}, wall_seconds=1.0) == (
        "wall=1.0s load=0.0s prompt_eval=0.0s(0tok) eval=0.0s(0tok)"
    )


def test_logged_step_logs_name_and_seconds(caplog):
    with caplog.at_level(logging.INFO, logger="misinfo"):
        with logged_step("load_e5"):
            pass

    assert "load_e5 seconds=" in caplog.text


@pytest.fixture
def clean_misinfo_logger():
    logger = logging.getLogger("misinfo")
    handlers, level = list(logger.handlers), logger.level
    logger.handlers.clear()
    logger.setLevel(logging.NOTSET)
    yield logger
    logger.handlers[:] = handlers
    logger.setLevel(level)


def test_script_logging_writes_info_to_stderr_for_the_server_to_collect(clean_misinfo_logger, capsys):
    configure_script_logging()

    logging.getLogger("misinfo").info("sentence 1/2 total=3.2s")

    assert "sentence 1/2 total=3.2s" in capsys.readouterr().err


def test_script_logging_is_idempotent(clean_misinfo_logger):
    configure_script_logging()
    configure_script_logging()

    assert len(clean_misinfo_logger.handlers) == 1


def test_ollama_call_logs_load_prompt_and_generation_time(monkeypatch, caplog):
    """Ollama가 알려주는 모델 로드/근거 읽기/생성 시간이 로그에 남아야 2분의 내역을 알 수 있다."""
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({
                "response": "{\"ok\": true}", "load_duration": 13_000_000_000,
                "prompt_eval_duration": 8_000_000_000, "prompt_eval_count": 2000,
                "eval_duration": 1_000_000_000, "eval_count": 50,
            }).encode()

    monkeypatch.setattr(misinfo_infer.urllib.request, "urlopen", lambda req, timeout: FakeResponse())

    with caplog.at_level(logging.INFO, logger="misinfo"):
        response = misinfo_infer.ask_ollama("http://ollama", "qwen", "프롬프트", misinfo_infer.JUDGE_SCHEMA)

    assert response == "{\"ok\": true}"
    assert "ollama wall=" in caplog.text
    assert "load=13.0s prompt_eval=8.0s(2000tok) eval=1.0s(50tok)" in caplog.text


def test_ollama_unload_is_timed(monkeypatch, caplog):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(misinfo_infer.urllib.request, "urlopen", lambda req, timeout: FakeResponse())

    with caplog.at_level(logging.INFO, logger="misinfo"):
        misinfo_infer.unload_ollama_model("http://ollama", "qwen")

    assert "unload_ollama seconds=" in caplog.text


def _fake_torch_and_transformers(monkeypatch, model_factory_name="AutoModel", model=None):
    class FakeModel:
        config = types.SimpleNamespace(
            id2label={0: "entailment", 1: "neutral", 2: "contradiction"}, type_vocab_size=1
        )

        def eval(self):
            return self

        def to(self, device):
            return self

    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda path: "tokenizer"),
        AutoModel=types.SimpleNamespace(from_pretrained=lambda model_id: FakeModel()),
        AutoModelForSequenceClassification=types.SimpleNamespace(from_pretrained=lambda path: FakeModel()),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def test_e5_model_load_is_timed(monkeypatch, caplog):
    _fake_torch_and_transformers(monkeypatch)
    monkeypatch.setattr(misinfo_infer, "_embed_tokenizer", None)
    monkeypatch.setattr(misinfo_infer, "_embed_model", None)
    monkeypatch.setattr(misinfo_infer, "_embed_device", "cpu")

    with caplog.at_level(logging.INFO, logger="misinfo"):
        misinfo_infer._load_embed_model()

    assert "load_e5 seconds=" in caplog.text


def test_nli_model_load_is_timed(monkeypatch, caplog):
    _fake_torch_and_transformers(monkeypatch)

    with caplog.at_level(logging.INFO, logger="misinfo"):
        nli_check.NliModel("C:/fake/nli_model_aug")._load()

    assert "load_nli seconds=" in caplog.text
