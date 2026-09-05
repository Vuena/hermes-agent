"""Global GPT-before pre-router for Hermes Agent.

This module is deliberately deterministic and provider-agnostic.  It runs inside
``agent.conversation_loop`` so CLI, Telegram/gateway, TUI and API requests share
the same policy before the normal frontier-agent loop makes a model request.

Terminal routes are best-effort: an LFM/NIM failure returns ``handled=False``
and the caller continues through the normal GPT agent path.  Secrets and source
content are never written to telemetry.
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


class DispatchBudgetExceeded(RuntimeError):
    """Raised before a provider call would violate the per-turn hard caps."""


@dataclass
class DispatchBudget:
    """Account for physical provider calls made during one user turn."""

    max_dispatches: int = 2
    max_transitions: int = 1
    max_combo_dispatches: int = 1
    dispatches: int = 0
    transitions: int = 0
    combo_dispatches: int = 0
    last_route: str | None = None

    def reserve(self, route: str) -> None:
        normalized = str(route or "").strip().lower()
        if self.dispatches >= self.max_dispatches:
            raise DispatchBudgetExceeded("provider dispatch cap exceeded")
        next_transitions = self.transitions + int(
            self.last_route is not None and self.last_route != normalized
        )
        if next_transitions > self.max_transitions:
            raise DispatchBudgetExceeded("provider transition cap exceeded")
        if normalized == "combo" and self.combo_dispatches >= self.max_combo_dispatches:
            raise DispatchBudgetExceeded("Combo dispatch cap exceeded")
        self.dispatches += 1
        self.transitions = next_transitions
        self.combo_dispatches += int(normalized == "combo")
        self.last_route = normalized

_LFM_PACKAGE = "_hermes_lfm_local_worker"
_LFM_MODULE = None
_LFM_LOCK = threading.Lock()

# Deterministic privacy scanner.  These are intentionally conservative: a false
# positive only keeps work on the local route or escalates to GPT; it must never
# send a likely secret/PII to NVIDIA NIM.
_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.I),
    re.compile(r"\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|password|passwd|secret)\s*[:=]", re.I),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}", re.I),
    re.compile(r"\b(?:sk|pk|rk|ghp|github_pat|xox[baprs]|npm)_[A-Za-z0-9_-]{12,}\b", re.I),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
)
_PII_PATTERNS = (
    re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"(?<!\d)(?:\+?\d[\d ()-]{8,}\d)(?!\d)"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
)
_CRITICAL_TERMS = re.compile(
    r"\b(delete|destroy|deploy|publish|transfer|pay|payment|purchase|buy|sell|send|permission|credential|security|" 
    r"sil|y[uü]kle|da[gğ][ıi]t|yay[iı]nla|transfer|[öo]deme|sat[iı]n al|sat|g[öo]nder|yetki|[sş]ifre|g[uü]venlik)\b",
    re.I,
)
_TOOL_TERMS = re.compile(
    r"(?:\b(?:run|execute|terminal|shell|browser|search the web|look up|download|upload|call|api)\b|"
    r"\b(?:create|edit|modify|write|open)\s+(?:a\s+)?(?:file|document|folder|directory|script|url|browser)\b|"
    r"\b(?:dosya|belge|klas[oö]r|dizin|script|url)\s+(?:olu[sş]tur|d[uü]zenle|yaz|a[cç])\b|"
    r"\b(?:dosyaya|dosyay[iı]|belgeye|belgey[iı])\s+(?:yaz|d[uü]zenle|a[cç])\b|"
    r"\b(?:webde|internette|internetten)\s+ara\b|"
    r"\bterminalde\s+(?:[cç]al[iı][sş]t[iı]r|y[uü]r[uü]t)\b|"
    r"\btaray[iı]c[iı]da\s+a[cç]\b|"
    r"\b(?:indir|y[uü]kle)\s+(?:dosyay[iı]|ar[cç]iv[iı]|model[iı]|paket[iı])\b)",
    re.I,
)
_BOUNDED_TERMS = re.compile(
    r"\b(ocr|extract|extraction|parse|classify|classification|summari[sz]e|summary|translate|translation|rewrite|grammar|" 
    r"format|normalize|identify|read|list|table|csv|json|triage|regex|" 
    r"[oö]zetle|[oö]zet|[cç][iı]kar|ay[iı]r|s[iı]n[iı]fland[iı]r|[cç]evir|yeniden yaz|dilbilgisi|bi[cç]imlendir|" 
    r"normalle[sş]tir|tan[iı]ml a|oku|listele|tablo|kod inceleme|hata ay[iı]kla)\b",
    re.I,
)
_ROUTINE_TERMS = re.compile(
    r"\b(explain|what is|how does|compare|draft|outline|polish|summarize|translate|rewrite|" 
    r"nedir|nas[iı]l [cç][aâ]l[iı][sş][iı]r|a[cç][iı]kla|kar[sş][iı]la[sş]t[iı]r|taslak|"
    r"[oö]zet|[cç]evir|yeniden yaz|d[uü]zelt|maddele|plan\w*|rutin\w*|"
    r"[oö]ner\w*|[oö]neri|fikir|alternatif|ipu[cç]lar[iı]|yard[iı]mc[iı] ol)\b",
    re.I,
)
_FOLLOWUP_REFERENCES = re.compile(
    r"\b(?:this|that|it|these|those|above|below|previous|earlier|continue|again|"
    r"bunu|şunu|onu|bunları|şunları|yukarıdaki|aşağıdaki|önceki|az önce|devam et|tekrar)\b",
    re.I,
)
_STANDARD_CODE_TERMS = re.compile(
    r"(?:\b(?:write|draft|implement|refactor|review|explain)\b.{0,80}"
    r"\b(?:code|function|class|module|test|regex|sql|python|javascript|typescript|"
    r"rust|go|java|c\+\+)\b|"
    r"\b(?:kod|fonksiyon|sınıf|modül|test|regex|sql|python|javascript|typescript)\b"
    r".{0,80}\b(?:yaz|tasarla|uygula|düzenle|incele|açıkla)\b)",
    re.I | re.S,
)
_COMPLEX_EXPERT_TERMS = re.compile(
    r"\b(?:deep(?:ly)?|in-depth|root cause|formal proof|threat model|adversarial audit|"
    r"derinlemesine|k[oö]k neden|bi[cç]imsel kan[iı]t|tehdit modeli|adversarial denetim|"
    r"ileri d[uü]zey uzman|karma[sş][iı]k muhakeme)\b",
    re.I,
)


def _home() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except Exception:
        return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))


def _local_image_path(message: Any) -> str | None:
    text, _ = _text_and_image(message)
    match = re.search(r"\[Image attached at:\s*([^\]]+)", text, re.I)
    if not match:
        return None
    candidate = os.path.expanduser(match.group(1).strip())
    return candidate if Path(candidate).is_file() else None


def _config() -> dict[str, Any]:
    try:
        import yaml
        path = _home() / "config.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _enabled() -> bool:
    raw = os.environ.get("HERMES_PRE_ROUTER", "")
    if raw.strip().lower() in {"0", "false", "off", "no"}:
        return False
    cfg = _config().get("pre_router", {})
    return bool(cfg.get("enabled", True)) if isinstance(cfg, dict) else True


def _telemetry_enabled() -> bool:
    cfg = _config().get("pre_router", {})
    return bool(cfg.get("telemetry", True)) if isinstance(cfg, dict) else True


def _shadow_enabled() -> bool:
    cfg = _config().get("pre_router", {})
    return bool(cfg.get("shadow", False)) if isinstance(cfg, dict) else False


def _confidence_threshold() -> float:
    cfg = _config().get("pre_router", {})
    try:
        return min(1.0, max(0.0, float(cfg.get("confidence_threshold", 0.85))))
    except (TypeError, ValueError):
        return 0.85


def _local_queue_timeout() -> float:
    cfg = _config().get("pre_router", {})
    try:
        return max(1.0, min(900.0, float(cfg.get("local_queue_timeout_seconds", 120))))
    except (TypeError, ValueError):
        return 120.0


def _local_min_free_vram_mb() -> int:
    cfg = _config().get("pre_router", {})
    try:
        return max(0, min(8192, int(cfg.get("local_min_free_vram_mb", 3000))))
    except (TypeError, ValueError):
        return 3000


_LOCAL_INFERENCE_LOCK = threading.Lock()


def _try_lock_file(handle: Any) -> bool:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    import fcntl
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _unlock_file(handle: Any) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def _local_inference_slot() -> Iterator[float]:
    """Serialize local GPU inference across threads and Hermes processes."""
    timeout = _local_queue_timeout()
    started = time.perf_counter()
    if not _LOCAL_INFERENCE_LOCK.acquire(timeout=timeout):
        raise TimeoutError("local inference queue timeout")
    handle = None
    try:
        runtime = _home() / "runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        lock_path = runtime / "local_inference.lock"
        handle = lock_path.open("a+b")
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"0")
            handle.flush()
        deadline = time.monotonic() + timeout
        while not _try_lock_file(handle):
            if time.monotonic() >= deadline:
                raise TimeoutError("local inference process queue timeout")
            time.sleep(0.10)
        yield round(time.perf_counter() - started, 3)
    finally:
        if handle is not None:
            _unlock_file(handle)
            handle.close()
        _LOCAL_INFERENCE_LOCK.release()


def _gpu_preflight() -> tuple[bool, int | None]:
    """Return whether enough free VRAM remains for a bounded local model."""
    minimum = _local_min_free_vram_mb()
    if minimum <= 0:
        return True, None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        values = [int(x.strip()) for x in result.stdout.splitlines() if x.strip().isdigit()]
        if not values:
            return True, None
        free_mb = max(values)
        return free_mb >= minimum, free_mb
    except (OSError, subprocess.SubprocessError, ValueError):
        return True, None


def _text_and_image(message: Any) -> tuple[str, bool]:
    if isinstance(message, str):
        return message, False
    if isinstance(message, list):
        parts: list[str] = []
        image = False
        for item in message:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                kind = str(item.get("type", ""))
                if "image" in kind or "image_url" in item or "image_path" in item:
                    image = True
                content = item.get("text") or item.get("content")
                if isinstance(content, str):
                    parts.append(content)
        return "\n".join(parts), image
    return str(message or ""), False


def _has_sensitive(text: str) -> tuple[bool, str]:
    for pattern in _SECRET_PATTERNS:
        if pattern.search(text):
            return True, "secret_pattern"
    for pattern in _PII_PATTERNS:
        if pattern.search(text):
            return True, "pii_pattern"
    return False, ""


def decide_admission(
    message: Any,
    *,
    conversation_history: list[dict[str, Any]] | None = None,
    platform: str = "",
    image_path: str | None = None,
) -> dict[str, Any]:
    """Choose one of the four sibling terminal routes without dispatching."""
    text, embedded_image = _text_and_image(message)
    sensitive, sensitive_reason = _has_sensitive(text)
    privacy_gate = "sensitive" if sensitive else "clear"
    history = conversation_history or []

    if platform.lower() in {"subagent", "cron_internal"}:
        return {
            "route": "combo", "confidence": 1.0, "reason": "internal_agent",
            "privacy_gate": privacy_gate, "sensitive": sensitive,
        }
    if _CRITICAL_TERMS.search(text) or _TOOL_TERMS.search(text):
        return {
            "route": "combo", "confidence": 0.98,
            "reason": "critical_or_tool_action", "privacy_gate": privacy_gate,
            "sensitive": sensitive,
        }
    if embedded_image or image_path:
        route = "local" if image_path else "combo"
        return {
            "route": route, "confidence": 0.92,
            "reason": "bounded_vision" if route == "local" else "vision_without_local_path",
            "privacy_gate": privacy_gate, "sensitive": sensitive,
        }
    if sensitive:
        bounded = bool(_BOUNDED_TERMS.search(text)) and len(text) <= 24_000
        return {
            "route": "local" if bounded else "combo",
            "confidence": 0.92 if bounded else 0.9,
            "reason": "private_bounded_text" if bounded else sensitive_reason,
            "privacy_gate": privacy_gate, "sensitive": True,
        }
    if len(text) > 24_000 or _COMPLEX_EXPERT_TERMS.search(text):
        return {
            "route": "combo", "confidence": 0.92,
            "reason": "complex_expert_reasoning", "privacy_gate": privacy_gate,
            "sensitive": False,
        }
    if _STANDARD_CODE_TERMS.search(text) and len(text) <= 24_000:
        return {
            "route": "muse", "confidence": 0.9, "reason": "standard_code",
            "privacy_gate": privacy_gate, "sensitive": False,
        }
    if history:
        return {
            "route": "muse", "confidence": 0.9, "reason": "standard_or_stateful",
            "privacy_gate": privacy_gate, "sensitive": False,
        }
    legacy = decide(
        message,
        conversation_history=conversation_history,
        platform=platform,
        image_path=image_path,
        respect_enabled=False,
    )
    route = {"lfm": "local", "gpt": "muse"}.get(legacy["route"], legacy["route"])
    reason = "standard_general" if legacy["route"] == "gpt" else legacy["reason"]
    return {
        **legacy,
        "route": route,
        "reason": reason,
        "privacy_gate": privacy_gate,
    }


def shadow_observe(
    message: Any,
    *,
    conversation_history: list[dict[str, Any]] | None = None,
    platform: str = "",
    image_path: str | None = None,
    actual_provider: str = "",
    actual_model: str = "",
    request_id: str | None = None,
) -> dict[str, Any] | None:
    """Record the broker recommendation while leaving live dispatch untouched."""
    if not _shadow_enabled():
        return None
    decision = decide_admission(
        message,
        conversation_history=conversation_history,
        platform=platform,
        image_path=image_path,
    )
    _telemetry({
        **decision,
        "event_type": "shadow_decision",
        "request_id": request_id,
        "actual_provider": actual_provider,
        "actual_model": actual_model,
        "dispatch_certainty": "not_sent",
        "handled": False,
    })
    return decision


def record_shadow_outcome(
    *,
    request_id: str | None,
    actual_provider: str,
    actual_model: str,
    api_calls: int,
    input_tokens: int | None,
    output_tokens: int | None,
    completed: bool,
    failed: bool,
    interrupted: bool,
    stop_reason: str,
) -> None:
    """Record the real turn outcome paired with a prior shadow recommendation."""
    if not _shadow_enabled():
        return
    usable_final = bool(completed and not failed and not interrupted)
    _telemetry({
        "event_type": "shadow_outcome",
        "request_id": request_id,
        "route": "actual",
        "actual_provider": actual_provider,
        "actual_model": actual_model,
        "api_calls": int(api_calls or 0),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "handled": usable_final,
        "usable_final": usable_final,
        "stop_reason": stop_reason,
        "dispatch_certainty": "sent" if api_calls else "not_sent",
        "error_type": "turn_failed" if failed else "",
    })


def _load_lfm_module():
    global _LFM_MODULE
    if _LFM_MODULE is not None:
        return _LFM_MODULE
    with _LFM_LOCK:
        if _LFM_MODULE is not None:
            return _LFM_MODULE
        plugin_dir = _home() / "plugins" / "lfm-local-worker"
        init_path = plugin_dir / "__init__.py"
        if not init_path.is_file():
            raise RuntimeError(f"local worker plugin not found: {plugin_dir}")
        spec = importlib.util.spec_from_file_location(
            _LFM_PACKAGE,
            init_path,
            submodule_search_locations=[str(plugin_dir)],
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load local worker plugin")
        module = importlib.util.module_from_spec(spec)
        import sys
        sys.modules[_LFM_PACKAGE] = module
        spec.loader.exec_module(module)
        _LFM_MODULE = module
        return module


def _telemetry(event: dict[str, Any]) -> None:
    if not _telemetry_enabled():
        return
    try:
        runtime = _home() / "runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        event = {
            "ts": round(time.time(), 3),
            "request_id": event.get("request_id") or uuid.uuid4().hex,
            "event_type": event.get("event_type", "route"),
            "route": event.get("route", "gpt"),
            "provider": event.get("provider", ""),
            "model": event.get("model", ""),
            "reason": event.get("reason", ""),
            "privacy_gate": event.get("privacy_gate", ""),
            "actual_provider": event.get("actual_provider", ""),
            "actual_model": event.get("actual_model", ""),
            "resolved_model": event.get("resolved_model", ""),
            "dispatch_certainty": event.get("dispatch_certainty", ""),
            "stop_reason": event.get("stop_reason", ""),
            "usable_final": event.get("usable_final"),
            "schema_valid": event.get("schema_valid"),
            "sensitive": bool(event.get("sensitive", False)),
            "handled": bool(event.get("handled", False)),
            "fallback": bool(event.get("fallback", False)),
            "elapsed_seconds": event.get("elapsed_seconds", 0),
            "queue_wait_seconds": event.get("queue_wait_seconds", 0),
            "api_calls": event.get("api_calls"),
            "free_vram_mb": event.get("free_vram_mb"),
            "input_tokens": event.get("input_tokens"),
            "output_tokens": event.get("output_tokens"),
            "error_type": event.get("error_type", ""),
        }
        with (runtime / "pre_router_events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        pass


def record_gpt_usage(
    *,
    request_id: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
    elapsed_seconds: float = 0.0,
    provider: str = "",
    model: str = "",
) -> None:
    """Append provider-reported GPT usage for the savings dashboard.

    Route-decision events intentionally remain separate from usage events so
    retries and multiple provider calls can be counted without duplicating the
    route mix. No prompt content is recorded.
    """
    _telemetry({
        "event_type": "usage",
        "request_id": request_id,
        "route": "gpt",
        "reason": "frontier_provider_usage",
        "handled": True,
        "elapsed_seconds": round(float(elapsed_seconds or 0), 3),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "provider": provider,
        "model": model,
    })


def decide(
    message: Any,
    *,
    conversation_history: list[dict[str, Any]] | None = None,
    platform: str = "",
    mode: str = "",
    source_path: str | None = None,
    image_path: str | None = None,
    respect_enabled: bool = True,
) -> dict[str, Any]:
    """Return a deterministic route decision without calling a model."""
    text, has_image = _text_and_image(message)
    has_image = has_image or bool(image_path)
    lowered = text.strip().lower()
    sensitive, sensitive_reason = _has_sensitive(text)
    history = conversation_history or []
    if respect_enabled and not _enabled():
        return {"route": "gpt", "confidence": 1.0, "reason": "disabled", "sensitive": sensitive}
    if platform.lower() in {"subagent", "cron_internal"}:
        return {"route": "gpt", "confidence": 1.0, "reason": "internal_agent", "sensitive": sensitive}
    if not lowered or lowered.startswith("/") or lowered.startswith("[system note"):
        return {"route": "gpt", "confidence": 1.0, "reason": "control_or_empty", "sensitive": sensitive}
    if _CRITICAL_TERMS.search(text) or _TOOL_TERMS.search(text):
        return {"route": "gpt", "confidence": 0.98, "reason": "critical_or_tool_action", "sensitive": sensitive}
    if has_image:
        if image_path and (len(text) <= 2000 or _BOUNDED_TERMS.search(text)):
            return {"route": "lfm", "confidence": 0.92, "reason": "bounded_vision", "sensitive": sensitive, "mode": "vision"}
        return {"route": "gpt", "confidence": 0.9, "reason": "vision_without_local_path", "sensitive": sensitive}
    # Stateful follow-ups stay with GPT unless the user explicitly requests a
    # bounded transformation that is self-contained in the current message.
    if history and (
        not _BOUNDED_TERMS.search(text)
        or (_FOLLOWUP_REFERENCES.search(text) and len(text) < 2_000)
    ):
        return {"route": "gpt", "confidence": 0.86, "reason": "stateful_followup", "sensitive": sensitive}
    if _BOUNDED_TERMS.search(text) and len(text) <= 24_000:
        bounded_mode = mode or (
            "code_triage" if re.search(r"code|bug|error|hata|kod", text, re.I)
            else "classify" if re.search(r"classif|s[iı]n[iı]fland[iı]r", text, re.I)
            else "extract" if re.search(r"extract|parse|[cç][iı]kar|ay[iı]r|json|csv", text, re.I)
            else "summarize"
        )
        # LFM is the terminal route for private work and for sufficiently large
        # bounded payloads.  For tiny non-sensitive transformations, NIM is more
        # economical because the local worker deliberately protects its context
        # budget with an 800-token threshold.
        if sensitive or len(text) // 4 >= 800:
            return {"route": "lfm", "confidence": 0.9, "reason": "bounded_text", "sensitive": sensitive, "mode": bounded_mode}
        if not sensitive:
            return {"route": "nim", "confidence": 0.87, "reason": "small_bounded_text", "sensitive": False, "mode": bounded_mode}
    if sensitive and _BOUNDED_TERMS.search(text) and len(text) <= 24_000:
        return {"route": "lfm", "confidence": 0.87, "reason": "private_bounded_text", "sensitive": True, "mode": mode or "summarize"}
    if _ROUTINE_TERMS.search(text) and len(text) <= 12_000 and not sensitive:
        return {"route": "nim", "confidence": 0.88, "reason": "routine_medium_text", "sensitive": False, "mode": mode or "answer"}
    if sensitive and source_path and len(text) <= 24_000:
        return {"route": "lfm", "confidence": 0.87, "reason": "private_bounded_text", "sensitive": True, "mode": mode or "summarize"}
    return {"route": "gpt", "confidence": 0.7, "reason": "complex_or_uncertain", "sensitive": sensitive}


def _result_text(result: Any) -> str:
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, (dict, list)):
        return json.dumps(result, ensure_ascii=False, indent=2)
    return str(result)


def dispatch(
    message: Any,
    *,
    conversation_history: list[dict[str, Any]] | None = None,
    platform: str = "",
    mode: str = "",
    source_path: str | None = None,
    image_path: str | None = None,
    request_id: str | None = None,
) -> dict[str, Any] | None:
    """Execute an LFM/NIM terminal route, or return None for the GPT path."""
    if not _enabled():
        return None
    if image_path is None:
        image_path = _local_image_path(message)
    decision = decide(
        message,
        conversation_history=conversation_history,
        platform=platform,
        mode=mode,
        source_path=source_path,
        image_path=image_path,
    )
    route = decision["route"]
    if route != "gpt" and float(decision.get("confidence", 0.0)) < _confidence_threshold():
        _telemetry({
            **decision,
            "request_id": request_id,
            "handled": False,
            "fallback": True,
            "reason": "below_confidence_threshold",
        })
        return None
    if route == "gpt":
        _telemetry({**decision, "request_id": request_id, "handled": False})
        return None
    text, _ = _text_and_image(message)
    started = time.perf_counter()
    queue_wait = 0.0
    if route == "lfm":
        enough_vram, free_vram_mb = _gpu_preflight()
        if not enough_vram:
            _telemetry({
                **decision,
                "request_id": request_id,
                "handled": False,
                "fallback": True,
                "reason": "vram_preflight_failed",
                "free_vram_mb": free_vram_mb,
            })
            return None
    try:
        module = _load_lfm_module()
        params: dict[str, Any] = {
            "mode": decision.get("mode", mode or "answer"),
            "task": text,
            "text": text,
            "source_is_task": True,
            "max_tokens": 256,
            "allow_small_source": bool(decision.get("sensitive")),
        }
        if route == "lfm":
            if image_path:
                params.pop("text", None)
                params["image_path"] = image_path
            with _local_inference_slot() as queue_wait:
                raw = module.run_worker(params)
        else:
            raw = module.run_nim_worker(params)
        payload = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(payload, dict) or not payload.get("success"):
            raise RuntimeError(str((payload or {}).get("error", "terminal route failed"))[:300])
        result = _result_text(payload.get("result", payload.get("content", "")))
        if not result:
            raise RuntimeError("terminal route returned an empty result")
        elapsed = round(time.perf_counter() - started, 3)
        metrics = payload.get("metrics") or {}
        metrics["queue_wait_seconds"] = queue_wait
        _telemetry({
            **decision,
            "request_id": request_id,
            "handled": True,
            "elapsed_seconds": elapsed,
            "queue_wait_seconds": queue_wait,
            "input_tokens": metrics.get("prompt_tokens") or metrics.get("local_prompt_tokens"),
            "output_tokens": metrics.get("completion_tokens") or metrics.get("local_output_tokens"),
        })
        return {
            "handled": True,
            "route": route,
            "route_reason": decision.get("reason"),
            "confidence": decision.get("confidence"),
            "sensitive": decision.get("sensitive", False),
            "final_response": result,
            "metrics": metrics,
            "provider": payload.get("provider", "local" if route == "lfm" else "nvidia"),
            "model": payload.get("model", ""),
        }
    except Exception as exc:
        _telemetry({
            **decision,
            "request_id": request_id,
            "handled": False,
            "fallback": True,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "error_type": type(exc).__name__,
        })
        # Fail open to GPT.  The caller continues through the existing agent
        # path; no raw source/error is exposed in telemetry.
        return None


def append_terminal_result(agent: Any, ctx: Any, terminal: dict[str, Any]) -> dict[str, Any]:
    """Persist a terminal response using the normal agent transcript seam."""
    response = str(terminal.get("final_response") or "").strip()
    messages = list(getattr(ctx, "messages", []) or [])
    messages.append({"role": "assistant", "content": response})
    agent._session_messages = messages
    metrics = terminal.get("metrics") or {}
    try:
        agent.session_prompt_tokens = metrics.get("prompt_tokens") or metrics.get("local_prompt_tokens") or 0
        agent.session_completion_tokens = metrics.get("completion_tokens") or metrics.get("local_output_tokens") or 0
        agent.session_input_tokens = agent.session_prompt_tokens
        agent.session_output_tokens = agent.session_completion_tokens
        agent.session_total_tokens = agent.session_prompt_tokens + agent.session_completion_tokens
    except Exception:
        pass
    try:
        agent._persist_session(messages, getattr(ctx, "conversation_history", None))
    except Exception:
        # A response is still safe to return; the normal gateway/API layer may
        # persist its own copy.  Never turn a successful local/NIM answer into a
        # provider failure solely because transcript I/O is unavailable.
        pass
    return {
        "final_response": response,
        "messages": messages,
        "api_calls": 0,
        "completed": True,
        "failed": False,
        "tools": [],
        "history_offset": len(getattr(ctx, "conversation_history", []) or []),
        "route": terminal.get("route"),
        "route_reason": terminal.get("route_reason"),
        "confidence": terminal.get("confidence"),
        "provider": terminal.get("provider"),
        "model": terminal.get("model"),
        "input_tokens": (terminal.get("metrics") or {}).get("prompt_tokens") or (terminal.get("metrics") or {}).get("local_prompt_tokens") or 0,
        "output_tokens": (terminal.get("metrics") or {}).get("completion_tokens") or (terminal.get("metrics") or {}).get("local_output_tokens") or 0,
    }
