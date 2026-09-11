import importlib.util
import json
import sys
from pathlib import Path

import pytest


PLUGIN_DIR = Path(r"C:\Users\Xvn\AppData\Local\hermes\plugins\lfm-local-worker")
PACKAGE_NAME = "_test_nemotron_ultra_worker"


def _load_worker():
    for name in list(sys.modules):
        if name == PACKAGE_NAME or name.startswith(PACKAGE_NAME + "."):
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = package
    spec.loader.exec_module(package)
    return sys.modules[f"{PACKAGE_NAME}.nim_worker"]


def test_worker_reports_nemotron_ultra_model(monkeypatch):
    worker = _load_worker()
    monkeypatch.setattr(worker, "_acquire_slot", lambda: None)
    monkeypatch.setattr(
        worker,
        "_request",
        lambda *args, **kwargs: {
            "content": "doğrulanmış yanıt",
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            "finish_reason": "stop",
        },
    )

    result = json.loads(
        worker.run_nim_worker(
            {"mode": "answer", "task": "Bu metni özetle", "text": "Kısa kaynak"}
        )
    )

    assert result["success"] is True
    assert result["model"] == "nvidia/nemotron-3-ultra-550b-a55b"


def test_ultra_limiter_allows_40_requests_and_blocks_41st(monkeypatch, tmp_path):
    worker = _load_worker()
    monkeypatch.setattr(worker, "_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(worker, "WINDOW_SECONDS", 60.0)

    for _ in range(40):
        worker._acquire_slot(timeout=0.01)

    with pytest.raises(TimeoutError, match="rate-limit queue timeout"):
        worker._acquire_slot(timeout=0.01)

    state_path = (
        tmp_path
        / "runtime"
        / "nvidia_nim_nemotron_3_ultra_550b_a55b_rate.json"
    )
    assert state_path.exists()
    assert len(json.loads(state_path.read_text(encoding="utf-8"))["timestamps"]) == 40


def test_sensitive_input_is_rejected_before_limiter(monkeypatch):
    worker = _load_worker()
    monkeypatch.setattr(
        worker,
        "_acquire_slot",
        lambda: (_ for _ in ()).throw(AssertionError("limiter/network must not run")),
    )

    result = json.loads(
        worker.run_nim_worker(
            {
                "mode": "answer",
                "task": "Summarize this record",
                "text": "customer email=test@example.com status=active",
            }
        )
    )

    assert result["success"] is False
    assert result["reason"] == "pii_pattern"


def test_retryable_http_error_retries_once_and_counts_both_attempts(monkeypatch):
    import io
    import urllib.error

    worker = _load_worker()
    attempts = []
    slots = []
    monkeypatch.setattr(worker, "_acquire_slot", lambda: slots.append("slot"))
    monkeypatch.setattr(worker.time, "sleep", lambda seconds: None)

    def flaky_request(*args, **kwargs):
        attempts.append("request")
        if len(attempts) == 1:
            raise urllib.error.HTTPError(
                worker.ENDPOINT, 503, "overloaded", {}, io.BytesIO(b'{"error":"overloaded"}')
            )
        return {
            "content": "başarılı",
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            "finish_reason": "stop",
        }

    monkeypatch.setattr(worker, "_request", flaky_request)
    result = json.loads(
        worker.run_nim_worker(
            {"mode": "answer", "task": "Bu metni özetle", "text": "Kısa kaynak"}
        )
    )

    assert result["success"] is True
    assert len(attempts) == 2
    assert len(slots) == 2
    assert result["metrics"]["attempts"] == 2
