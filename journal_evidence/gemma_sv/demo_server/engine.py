"""Engine interface plus an explicitly labeled recorded-replay implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any

from .contract import classify_admission, classify_certificate
from .scenarios import PRESETS, extraction_prompt
from .span import SelectedSpan
from .state import DemoPhase, SessionRecord


class EngineError(RuntimeError):
    """A safe, expected engine failure suitable for an API error code."""


class DemoEngine(ABC):
    name = "abstract"
    live_compute = False

    @abstractmethod
    def ingest(
        self,
        session: SessionRecord,
        selection: SelectedSpan,
        audit_probe: str,
        audit_target: str | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def recall(self, session: SessionRecord) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def forget(self, session: SessionRecord) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def certify(
        self,
        session: SessionRecord,
        progress=None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def attack(
        self,
        session: SessionRecord,
        *,
        method: str,
        kind: str,
        budget: int | None = None,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def retained_probe(
        self,
        session: SessionRecord,
        *,
        prompt: str,
        target: str,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def twin_start(self, session: SessionRecord) -> dict[str, Any]:
        raise NotImplementedError

    def twin_guess(self, session: SessionRecord, pane: str) -> dict[str, Any]:
        pane = pane.upper()
        if pane not in ("A", "B"):
            raise EngineError("invalid_twin_pane")
        deleted_pane = session.engine_state.get("deleted_pane")
        if deleted_pane is None:
            raise EngineError("twin_not_started")
        session.phase = DemoPhase.TWIN_TESTED
        return {
            "correct": pane == deleted_pane,
            "deleted_pane": deleted_pane,
            "scope": "registered_audit_probe",
            "certificate": session.results.get("certificate"),
        }


# Emergency fallback values for the committed 4B patient-field run. Normal
# replay reads the richer profile in ``demo_site/assets/replay.json``.
_REPLAY = {
    "medicine": {
        "answer": "acute intermittent porphyria",
        "recall_text": 'acute intermittent porphyria", "working_diagnosis_',
        "keep_probability": 0.053153588741219435,
        "exact_probability": 0.014648031031031133,
        "floor_probability": 0.010811089856048145,
        "attack_icul_probability": 0.03443056549333213,
        "attack_exact_probability": 0.009659235724346488,
        "attack_floor_probability": 0.009825695450934748,
        "certificate_kl": 2.87158643821609e-06,
        "fallbacks": 6,
        "head_gates": 240,
    },
}


@lru_cache(maxsize=1)
def _recorded_profile() -> dict[str, Any]:
    """The committed replay profile, shared with the static site fallback."""

    path = Path(__file__).resolve().parent.parent / "demo_site" / "assets" / "replay.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _recorded_log(domain: str) -> list[dict[str, Any]] | None:
    run = (_recorded_profile().get("runs") or {}).get(domain) or {}
    return run.get("conversation_log")


def _recorded_run(domain: str) -> dict[str, Any]:
    return (_recorded_profile().get("runs") or {}).get(domain) or {}


class ReplayDemoEngine(DemoEngine):
    """Deterministic fallback backed by the committed float64 transcript.

    Every response carries ``evidence=recorded_replay``.  It never presents a
    visitor-supplied custom fact as if the model had processed it.
    """

    name = "recorded_replay"
    model_id = "google/gemma-3-4b-pt"
    live_compute = False

    def ingest(self, session, selection, audit_probe, audit_target=None):
        session.memory_text = selection.text
        session.secret_start = selection.start
        session.secret_end = selection.end
        session.deletion_ranges = tuple(
            (item.start, item.end) for item in selection.ranges
        )
        session.deletion_scope = selection.deletion_scope
        session.record_id = selection.record_id
        session.audit_probe = audit_probe
        session.phase = DemoPhase.INGESTED
        available = (_recorded_profile().get("runs") or {})
        requested_key = (
            selection.record_id
            if selection.record_id in available
            else session.domain
        )
        replay_domain = (
            requested_key
            if requested_key in available
            else next(iter(available), "medicine")
        )
        custom_not_computed = requested_key not in available
        recorded = _recorded_run(replay_domain)
        target = audit_target or selection.value
        session.engine_state.update(
            {
                "target_value": target,
                "replay_domain": replay_domain,
                "custom_not_computed": custom_not_computed,
            }
        )
        return {
            "evidence": "recorded_replay",
            "custom_not_computed": custom_not_computed,
            "selected_value": selection.value,
            "selected_values": list(selection.values),
            "audit_target": target,
            "deletion_scope": selection.deletion_scope,
            "record_id": selection.record_id,
            "memory_tokens": recorded.get("memory_tokens"),
            "memory_copies": recorded.get("memory_copies", 1),
            "selected_positions": recorded.get("selected_positions"),
            "distance_beyond_window": recorded.get("distance_beyond_window"),
            "local_window": recorded.get("local_window", 1024),
            "field_audits": recorded.get("field_audits"),
            "neighbor": recorded.get("neighbor"),
            "conversation_log": _recorded_log(replay_domain),
            "message": (
                f"Recorded {recorded.get('model_id', self.model_id)} run loaded; "
                "no visitor text was sent to a model."
            ),
        }

    def recall(self, session):
        self._require_phase(session, DemoPhase.INGESTED)
        data = self._data(session)
        target = session.engine_state["target_value"]
        greedy_match = target.casefold() in data["recall_text"].casefold()
        admission = classify_admission(
            _safe_log(data["keep_probability"]),
            _safe_log(data["floor_probability"]),
            greedy_match=greedy_match,
        )
        result = {
            "evidence": "recorded_replay",
            "generated_text": data["recall_text"],
            "target": target,
            "target_probability": data["keep_probability"],
            "floor_probability": data["floor_probability"],
            "admission": {
                "status": admission.status.value,
                "log_lift_nats": admission.log_lift_nats,
                "probability_ratio": admission.probability_ratio,
                "greedy_match": admission.greedy_match,
                "message": admission.message,
            },
        }
        session.results["recall"] = result
        session.phase = DemoPhase.RECALLED
        return result

    def forget(self, session):
        if session.phase not in (DemoPhase.RECALLED, DemoPhase.INGESTED):
            raise EngineError("recall_or_ingest_required")
        data = self._data(session)
        result = {
            "evidence": "recorded_replay",
            "method": "exact",
            "before_probability": data["keep_probability"],
            "after_probability": data["exact_probability"],
            "floor_probability": data["floor_probability"],
            "deletion_ms": None,
            "message": "Recorded behavioral deletion; verification is reported separately.",
        }
        session.results["forget"] = result
        session.phase = DemoPhase.FORGOTTEN
        return result

    def certify(self, session, progress=None):
        if session.phase not in (
            DemoPhase.FORGOTTEN,
            DemoPhase.ATTACKED,
            DemoPhase.TWIN_TESTED,
        ):
            raise EngineError("forget_required")
        data = self._data(session)
        classified = classify_certificate(data["certificate_kl"])
        result = {
            "evidence": "recorded_replay",
            "kl_nats": classified.kl_nats,
            "band": classified.band.value,
            "probe_scoped": True,
            "probe": session.audit_probe,
            "decrement_fallbacks": data["fallbacks"],
            "head_gates": data["head_gates"],
            "message": classified.message,
        }
        session.results["certificate"] = result
        if progress is not None:
            progress(1.0)
        return result

    def attack(self, session, *, method, kind, budget=None, prompt=None):
        if session.phase not in (DemoPhase.FORGOTTEN, DemoPhase.ATTACKED):
            raise EngineError("forget_required")
        if method not in ("exact", "icul", "never"):
            raise EngineError("unsupported_attack_method")
        if kind not in ("extraction", "freeform", "elicitation"):
            raise EngineError("unsupported_attack_kind")
        data = self._data(session)
        key = {
            "icul": "attack_icul_probability",
            "exact": "attack_exact_probability",
            "never": "attack_floor_probability",
        }[method]
        result = {
            "evidence": "recorded_replay",
            "method": method,
            "kind": kind,
            "budget": budget,
            "prompt": prompt or extraction_prompt(session.audit_probe or ""),
            "target_probability": data[key],
            "floor_probability": data["attack_floor_probability"],
            "at_floor": abs(data[key] - data["attack_floor_probability"])
            <= max(1e-6, data["attack_floor_probability"] * 0.5),
            "generated_text": None,
            "message": "Recorded attack outcome; free-form visitor text was not executed.",
        }
        session.results.setdefault("attacks", []).append(result)
        session.phase = DemoPhase.ATTACKED
        return result

    def retained_probe(self, session, *, prompt, target):
        if session.phase not in (DemoPhase.FORGOTTEN, DemoPhase.ATTACKED):
            raise EngineError("forget_required")
        return {
            "evidence": "recorded_replay",
            "prompt": prompt,
            "target": target,
            "generated_text": None,
            "before_probability": None,
            "after_probability": None,
            "never_probability": None,
            "mean_log_probability_drift": None,
            "custom_not_computed": True,
            "message": (
                "The recorded engine does not execute a custom retained probe."
            ),
        }

    def twin_start(self, session):
        if "certificate" not in session.results:
            raise EngineError("certificate_required")
        digest = hashlib.sha256(session.session_id.encode("utf-8")).digest()
        deleted_pane = "A" if digest[0] % 2 == 0 else "B"
        session.engine_state["deleted_pane"] = deleted_pane
        data = self._data(session)
        panes = {
            deleted_pane: {"target_probability": data["exact_probability"]},
            "B" if deleted_pane == "A" else "A": {
                "target_probability": data["floor_probability"]
            },
        }
        return {
            "evidence": "recorded_replay",
            "panes": panes,
            "scope": "registered_audit_probe",
            "prompt": session.audit_probe,
            "message": "Guess which anonymous probe distribution came from exact deletion.",
        }

    @staticmethod
    def _require_phase(session: SessionRecord, phase: DemoPhase) -> None:
        if session.phase is not phase:
            raise EngineError(f"{phase.value}_phase_required")

    @staticmethod
    def _data(session: SessionRecord) -> dict[str, Any]:
        domain = session.engine_state.get("replay_domain", "medicine")
        run = _recorded_run(domain)
        if run:
            return {
                "answer": run["recall"]["target"],
                "recall_text": run["recall"]["generated_text"],
                "keep_probability": run["recall"]["target_probability"],
                "exact_probability": run["forget"]["after_probability"],
                "floor_probability": run["forget"]["floor_probability"],
                "attack_icul_probability": run["attacks"]["icul"][
                    "target_probability"
                ],
                "attack_exact_probability": run["attacks"]["exact"][
                    "target_probability"
                ],
                "attack_floor_probability": run["attacks"]["exact"][
                    "floor_probability"
                ],
                "certificate_kl": run["certificate"]["kl_nats"],
                "fallbacks": run["certificate"]["decrement_fallbacks"],
                "head_gates": run["certificate"]["head_gates"],
            }
        return _REPLAY[domain]


def preset_payloads() -> list[dict[str, object]]:
    """Public preset payloads, annotated with recorded-replay availability.

    The client disables a preset in replay mode unless ``replay_key`` names a
    recorded run; without this annotation the documented recorded-engine flow
    (static site + ``HERO_ENGINE=replay`` API) offers no runnable preset.
    """

    runs = _recorded_profile().get("runs") or {}
    payloads = []
    for preset in PRESETS.values():
        payload = preset.public_dict()
        if preset.slug in runs:
            payload["replay_key"] = preset.slug
        elif preset.domain in runs:
            payload["replay_key"] = preset.domain
        else:
            payload["replay_key"] = None
        payloads.append(payload)
    return payloads


def _safe_log(value: float) -> float:
    import math

    return math.log(max(float(value), 1e-300))
