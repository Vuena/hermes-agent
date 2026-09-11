import importlib.util
import inspect
import json
import sys
import threading
import time
from pathlib import Path

import pytest

from agent import global_pre_router as router


def test_router_keeps_bounded_non_sensitive_short_text_off_ultra():
    decision = router.decide("Explain in two sentences what a reverse proxy is.")
    assert decision["route"] != "nim"


def test_router_keeps_general_turkish_routine_off_ultra():
    decision = router.decide("Evde çalışan biri için verimli bir çalışma rutini oluştur.")
    assert decision["route"] != "nim"


def test_router_sends_multi_source_synthesis_to_nemotron_ultra():
    decision = router.decide(
        "Üç kaynak arasındaki çelişkileri çöz ve kaynaklar arası sentez üret."
    )
    assert decision["route"] == "nim"
    assert decision["reason"] == "nemotron_ultra_synthesis"


def test_admission_sends_fact_check_to_nemotron_ultra():
    decision = router.decide_admission(
        "Bu kaynaklardaki iddialar için kapsamlı fakt kontrolü yap."
    )
    assert decision["route"] == "nim"
    assert decision["reason"] == "nemotron_ultra_synthesis"


def test_router_keeps_file_creation_as_tool_action():
    decision = router.decide("Dosya oluştur ve içine sonucu yaz.")
    assert decision["route"] == "gpt"
    assert decision["reason"] == "critical_or_tool_action"


def test_router_keeps_critical_actions_on_gpt():
    decision = router.decide("Delete the old deployment now")
    assert decision["route"] == "gpt"
    assert decision["reason"] == "critical_or_tool_action"


def test_router_keeps_private_bounded_text_local():
    decision = router.decide("Classify this private record: email=test@example.com, status=active")
    assert decision["route"] == "lfm"
    assert decision["sensitive"] is True


def test_stateful_reference_followup_stays_with_gpt():
    history = [{"role": "user", "content": "Önceki mesaj"}, {"role": "assistant", "content": "Yanıt"}]
    decision = router.decide("Bunu özetle", conversation_history=history)
    assert decision["route"] == "gpt"
    assert decision["reason"] == "stateful_followup"


def test_direct_nim_worker_rejects_pii_before_network(monkeypatch):
    plugin_dir = Path(r"C:\Users\Xvn\AppData\Local\hermes\plugins\lfm-local-worker")
    package_name = "_test_lfm_privacy_worker"
    spec = importlib.util.spec_from_file_location(
        package_name,
        plugin_dir / "__init__.py",
        submodule_search_locations=[str(plugin_dir)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    nim_module = sys.modules[f"{package_name}.nim_worker"]
    monkeypatch.setattr(nim_module, "_acquire_slot", lambda: (_ for _ in ()).throw(AssertionError("limiter/network must not run")))
    payload = json.loads(module.run_nim_worker({
        "mode": "answer",
        "task": "Summarize this record",
        "text": "customer email=test@example.com status=active",
        "source_is_task": False,
    }))
    assert payload["success"] is False
    assert payload["reason"] == "pii_pattern"


def test_worker_plugin_dir_falls_back_from_profile_to_shared_root(monkeypatch, tmp_path):
    profile_home = tmp_path / "profiles" / "denetci"
    shared_plugin = tmp_path / "plugins" / "lfm-local-worker"
    shared_plugin.mkdir(parents=True)
    (shared_plugin / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(router, "_home", lambda: profile_home)

    assert router._worker_plugin_dir() == shared_plugin


def test_dispatch_marks_router_request_as_single_user_prompt(monkeypatch):
    seen = {}

    class FakeWorker:
        @staticmethod
        def run_nim_worker(params):
            seen.update(params)
            return json.dumps({"success": True, "result": "ok", "provider": "nvidia"})

        @staticmethod
        def run_worker(params):
            raise AssertionError("LFM must not run on the NIM route")

    monkeypatch.setattr(router, "_load_lfm_module", lambda: FakeWorker)
    result = router.dispatch("Üç kaynak arasındaki çelişkileri çöz ve kaynaklar arası sentez üret.")
    assert result["handled"] is True
    assert seen["source_is_task"] is True
    assert seen["task"] == seen["text"]


def test_dispatch_returns_terminal_nim_without_frontier(monkeypatch):
    class FakeWorker:
        @staticmethod
        def run_nim_worker(params):
            return json.dumps({
                "success": True,
                "provider": "nvidia",
                "model": "test-nim",
                "result": "terminal answer",
                "metrics": {"prompt_tokens": 10, "completion_tokens": 3},
            })

        @staticmethod
        def run_worker(params):
            raise AssertionError("LFM must not run on the NIM route")

    monkeypatch.setattr(router, "_load_lfm_module", lambda: FakeWorker)
    result = router.dispatch("Üç kaynak arasındaki çelişkileri çöz ve kaynaklar arası sentez üret.")
    assert result is not None
    assert result["handled"] is True
    assert result["route"] == "nim"
    assert result["final_response"] == "terminal answer"


def test_local_inference_slot_serializes_threads():
    intervals = []

    def job():
        with router._local_inference_slot():
            start = time.perf_counter()
            time.sleep(0.12)
            intervals.append((start, time.perf_counter()))

    threads = [threading.Thread(target=job) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(intervals) == 2
    first, second = sorted(intervals)
    assert first[1] <= second[0] or second[1] <= first[0]


def test_admission_has_four_terminal_routes():
    assert router.decide_admission(
        "Classify this private record: email=test@example.com"
    )["route"] == "local"
    assert router.decide_admission(
        "Üç kaynak arasındaki çelişkileri çöz ve kaynaklar arası sentez üret."
    )["route"] == "nim"
    assert router.decide_admission(
        "Write a pure Python function that deduplicates a list while preserving order."
    )["route"] == "muse"
    assert router.decide_admission(
        "Deploy the service and verify production health."
    )["route"] == "combo"


def test_admission_never_sends_sensitive_work_to_nim_or_muse():
    decision = router.decide_admission(
        "Draft a reply for customer@example.com about password=topsecret"
    )
    assert decision["route"] in {"local", "combo"}
    assert decision["privacy_gate"] == "sensitive"


@pytest.mark.parametrize(
    "message",
    [
        "Bu cevabı daha anlaşılır ve kısa yaz.",
        "Bu yaklaşımın artılarını ve eksilerini anlat.",
    ],
)
def test_admission_uses_muse_for_normal_stateful_followups(message):
    history = [
        {"role": "user", "content": "Bir yaklaşımı değerlendirelim."},
        {"role": "assistant", "content": "İlk değerlendirme."},
    ]

    decision = router.decide_admission(message, conversation_history=history)

    assert decision["route"] == "muse"
    assert decision["reason"] == "standard_or_stateful"


def test_admission_uses_muse_as_default_for_noncritical_standard_work():
    decision = router.decide_admission(
        "Bir SaaS ürünü için müşteri destek yanıtının tonunu değerlendir."
    )

    assert decision["route"] == "muse"
    assert decision["reason"] == "standard_general"


def test_admission_keeps_explicit_deep_expert_work_on_combo():
    decision = router.decide_admission(
        "Bu eşzamanlılık hatası için derinlemesine kök neden analizi yap."
    )

    assert decision["route"] == "combo"
    assert decision["reason"] == "complex_expert_reasoning"



def test_shadow_admission_still_recommends_ultra_when_live_router_is_disabled(monkeypatch):
    monkeypatch.setattr(router, "_enabled", lambda: False)

    decision = router.decide_admission(
        "Üç kaynak arasındaki çelişkileri çöz ve kaynaklar arası sentez üret."
    )

    assert decision["route"] == "nim"
    assert decision["reason"] == "nemotron_ultra_synthesis"


def test_dispatch_budget_enforces_two_calls_one_transition_one_combo():
    budget = router.DispatchBudget()
    budget.reserve("nim")
    budget.reserve("combo")

    with pytest.raises(router.DispatchBudgetExceeded, match="provider dispatch cap"):
        budget.reserve("combo")

    assert budget.dispatches == 2
    assert budget.transitions == 1
    assert budget.combo_dispatches == 1


def test_dispatch_budget_rejects_second_combo_even_without_total_cap():
    budget = router.DispatchBudget(max_dispatches=3)
    budget.reserve("combo")

    with pytest.raises(router.DispatchBudgetExceeded, match="Combo dispatch cap"):
        budget.reserve("combo")


def test_shadow_observe_records_recommendation_without_dispatch(monkeypatch):
    events = []
    monkeypatch.setattr(router, "_shadow_enabled", lambda: True)
    monkeypatch.setattr(router, "_telemetry", events.append)

    decision = router.shadow_observe(
        "Write a pure Python function that deduplicates a list while preserving order.",
        actual_provider="opencode-go",
        actual_model="muse-spark-1.2-contributor",
        request_id="turn-1",
    )

    assert decision["route"] == "muse"
    assert events == [{
        **decision,
        "event_type": "shadow_decision",
        "request_id": "turn-1",
        "actual_provider": "opencode-go",
        "actual_model": "muse-spark-1.2-contributor",
        "dispatch_certainty": "not_sent",
        "handled": False,
    }]


def test_shadow_telemetry_preserves_actual_route_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(router, "_home", lambda: tmp_path)
    monkeypatch.setattr(router, "_telemetry_enabled", lambda: True)

    router._telemetry({
        "event_type": "shadow_decision",
        "route": "nim",
        "reason": "routine_medium_text",
        "privacy_gate": "clear",
        "actual_provider": "opencode-go",
        "actual_model": "muse-spark-1.2-contributor",
        "dispatch_certainty": "not_sent",
    })

    event = json.loads(
        (tmp_path / "runtime" / "pre_router_events.jsonl").read_text(encoding="utf-8")
    )
    assert event["privacy_gate"] == "clear"
    assert event["actual_provider"] == "opencode-go"
    assert event["actual_model"] == "muse-spark-1.2-contributor"
    assert event["dispatch_certainty"] == "not_sent"


def test_conversation_loop_calls_shadow_observer_after_turn_context():
    from agent import conversation_loop

    target_fn = getattr(conversation_loop, "_run_conversation_turn", conversation_loop.run_conversation)
    source = inspect.getsource(target_fn)
    context_pos = source.index("_ctx = build_turn_context(")
    shadow_pos = source.index("shadow_observe(")
    loop_pos = source.index("while (s.api_call_count")

    assert context_pos < shadow_pos < loop_pos


def test_shadow_outcome_records_real_usage_without_content(monkeypatch):
    events = []
    monkeypatch.setattr(router, "_shadow_enabled", lambda: True)
    monkeypatch.setattr(router, "_telemetry", events.append)

    router.record_shadow_outcome(
        request_id="turn-1",
        actual_provider="opencode-go",
        actual_model="muse-spark-1.2-contributor",
        api_calls=1,
        input_tokens=100,
        output_tokens=20,
        completed=True,
        failed=False,
        interrupted=False,
        stop_reason="text_response(stop)",
    )

    assert events == [{
        "event_type": "shadow_outcome",
        "request_id": "turn-1",
        "route": "actual",
        "actual_provider": "opencode-go",
        "actual_model": "muse-spark-1.2-contributor",
        "api_calls": 1,
        "input_tokens": 100,
        "output_tokens": 20,
        "handled": True,
        "usable_final": True,
        "stop_reason": "text_response(stop)",
        "dispatch_certainty": "sent",
        "error_type": "",
    }]


def test_turn_finalizer_records_shadow_outcome_before_return():
    from agent import turn_finalizer

    source = inspect.getsource(turn_finalizer.finalize_turn)
    result_pos = source.index("result = {")
    outcome_pos = source.index("record_shadow_outcome(")
    return_pos = source.rindex("return result")

    assert result_pos < outcome_pos < return_pos
