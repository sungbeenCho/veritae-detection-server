import logging

import pytest

from app.logging_config import configure_logging


@pytest.fixture
def clean_app_logger():
    """다른 테스트에 영향이 없게 "app" 로거 상태를 되돌린다."""
    app_logger = logging.getLogger("app")
    handlers, level = list(app_logger.handlers), app_logger.level
    app_logger.handlers.clear()
    app_logger.setLevel(logging.NOTSET)
    yield app_logger
    app_logger.handlers[:] = handlers
    app_logger.setLevel(level)


def test_app_info_logs_are_hidden_without_configuration(clean_app_logger):
    """uvicorn은 자기 로거만 설정해서, 설정이 없으면 앱의 INFO 로그(GPU 대기열 시간 등)가 콘솔에 안 나온다(2026-10-13 확인)."""
    assert not logging.getLogger("app.services.gpu_queue").isEnabledFor(logging.INFO)


def test_configure_logging_shows_app_info_logs_on_stderr(clean_app_logger, capsys):
    configure_logging()

    logging.getLogger("app.services.gpu_queue").info("gpu_queue_released run_seconds=42")

    assert "gpu_queue_released run_seconds=42" in capsys.readouterr().err


def test_configure_logging_is_idempotent(clean_app_logger):
    configure_logging()
    configure_logging()

    assert len(clean_app_logger.handlers) == 1
