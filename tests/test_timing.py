import logging

import pytest

from app.services.timing import timed


def test_timed_passes_arguments_and_return_value_through():
    wrapped = timed("stage", lambda a, b, key=None: (a, b, key))

    assert wrapped(1, 2, key="x") == (1, 2, "x")


def test_timed_logs_label_and_seconds(caplog):
    with caplog.at_level(logging.INFO, logger="app.services.timing"):
        timed("spai", lambda: "ok")()

    assert "stage_done label=spai seconds=" in caplog.text


def test_timed_logs_failure_and_reraises(caplog):
    def boom():
        raise RuntimeError("실패")

    with caplog.at_level(logging.INFO, logger="app.services.timing"):
        with pytest.raises(RuntimeError, match="실패"):
            timed("scam", boom)()

    assert "stage_failed label=scam seconds=" in caplog.text
