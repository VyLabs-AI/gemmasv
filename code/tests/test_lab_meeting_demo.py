import json
import os

from fastapi.testclient import TestClient
import pytest

import gemma_sv.lab_meeting_demo as lab_meeting_demo
from gemma_sv.demo_server.resolver import (
    GeminiResolver,
    ProposalStore,
    ResolverUnavailable,
)
from gemma_sv.lab_meeting_demo import (
    CASE_ASSET,
    LAB_MODEL_ID,
    LabCaseError,
    LabStartupError,
    create_lab_app,
    load_lab_case,
    load_lab_environment,
    main,
)


class ResolverClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class FakeGeminiResolver:
    mode = "gemini"
    evidence = "live_resolver"
    network_requests = True
    available = True
    model_id = LAB_MODEL_ID

    def __init__(self, outcome="resolved"):
        self.outcome = outcome
        self.calls = []

    def resolve(self, request_text, candidates):
        self.calls.append((request_text, tuple(candidates)))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        selected = next(
            candidate
            for candidate in candidates
            if candidate.label == "Current number of bicycles"
        )
        selected_id = (
            "memory_not_in_committed_catalog"
            if self.outcome == "unknown"
            else selected.record_id
        )
        return {
            "status": "resolved",
            "selected_record_id": selected_id,
            "alternative_record_ids": [],
            "confidence": 0.99,
            "explanation": "Matches the committed bicycle-count update.",
            "requires_confirmation": True,
            "deletion_executed": False,
        }


def _resolve(http: TestClient):
    request_text = http.app.state.lab_case.request_text
    return http.post(
        "/api/lab/resolve",
        json={"request_text": request_text},
    )


def test_committed_lab_case_loads_two_complete_hash_bound_exchanges():
    case = load_lab_case()

    assert len(case.records) == 2
    assert case.fixture_selected_record_id in {
        record.record_id for record in case.records
    }
    assert case.server_catalog_hash != case.committed_catalog_hash
    assert len(case.payload_sha256) == 64
    assert len(case.method_report_sha256) == 64
    for record in case.records:
        assert record.deletion_scope == "complete_exchange"
        assert [turn.role for turn in record.owned_exchange] == [
            "user",
            "assistant",
        ]
        assert tuple(turn.turn_id for turn in record.owned_exchange) == (
            record.source_turn_ids
        )


def test_lab_case_rejects_any_catalog_tampering(tmp_path):
    payload = json.loads(CASE_ASSET.read_text(encoding="utf-8"))
    payload["case"]["memory_resolution"]["candidate_catalog"][0][
        "summary"
    ] = "Tampered browser-controlled summary."
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LabCaseError, match="integrity"):
        load_lab_case(tampered)


def test_recorded_rehearsal_resolves_and_confirms_without_live_deletion():
    with TestClient(create_lab_app(mode="recorded")) as http:
        config = http.get("/api/lab/config")
        assert config.status_code == 200
        assert config.json()["resolver"] == {
            "model_id": LAB_MODEL_ID,
            "mode": "recorded",
            "evidence": "recorded_resolution",
            "label": "Recorded resolver fixture — no provider call",
            "network_request_possible": False,
            "network_request_performed": False,
            "outside_deletion_certificate": True,
        }
        assert config.json()["deletion_evidence"]["kind"] == "recorded_replay"
        assert (
            config.json()["deletion_evidence"]["live_model_deletion_run"]
            is False
        )

        resolved = _resolve(http)
        assert resolved.status_code == 200
        data = resolved.json()
        assert data["status"] == "resolved"
        assert data["proposal_id"].startswith("proposal_")
        assert data["deletion_executed"] is False
        assert data["resolver"]["network_request_performed"] is False
        assert data["selected_record"]["label"] == "Current number of bicycles"
        assert len(data["selected_record"]["owned_exchange"]) == 2
        assert "HttpOnly" in resolved.headers["set-cookie"]

        confirmed = http.post(
            "/api/lab/confirm",
            json={"proposal_id": data["proposal_id"], "confirmed": True},
        )
        assert confirmed.status_code == 200
        confirmation = confirmed.json()
        assert confirmation["confirmed"] is True
        assert confirmation["selected_record_id"] == (
            data["selected_record"]["record_id"]
        )
        assert confirmation["confirmed_record"] == data["selected_record"]
        assert confirmation["deletion_executed"] is False
        assert confirmation["subsequent_deletion_evidence"] == (
            config.json()["deletion_evidence"]
        )

        reused = http.post(
            "/api/lab/confirm",
            json={"proposal_id": data["proposal_id"], "confirmed": True},
        )
        assert reused.status_code == 409
        assert reused.json()["detail"]["code"] == "proposal_already_used"


def test_live_shaped_fake_resolver_gets_only_server_catalog_and_stays_outside_evidence():
    resolver = FakeGeminiResolver()
    with TestClient(
        create_lab_app(mode="gemini", resolver=resolver)
    ) as http:
        resolved = _resolve(http)

        assert resolved.status_code == 200
        assert len(resolver.calls) == 1
        request_text, candidates = resolver.calls[0]
        assert request_text == http.app.state.lab_case.request_text
        assert candidates == http.app.state.lab_case.records
        data = resolved.json()
        assert data["resolver"]["mode"] == "gemini"
        assert data["resolver"]["evidence"] == "live_resolver"
        assert data["resolver"]["network_request_performed"] is True
        assert data["resolver"]["outside_deletion_certificate"] is True

        confirmed = http.post(
            "/api/lab/confirm",
            json={"proposal_id": data["proposal_id"], "confirmed": True},
        ).json()
        assert confirmed["deletion_executed"] is False
        assert (
            confirmed["subsequent_deletion_evidence"][
                "live_model_deletion_run"
            ]
            is False
        )
        assert confirmed["subsequent_deletion_evidence"]["kind"] == (
            "recorded_replay"
        )


def test_browser_cannot_submit_catalog_candidate_or_confirmation_record_id():
    resolver = FakeGeminiResolver()
    with TestClient(
        create_lab_app(mode="gemini", resolver=resolver)
    ) as http:
        request_text = http.app.state.lab_case.request_text
        injected_catalog = http.post(
            "/api/lab/resolve",
            json={
                "request_text": request_text,
                "candidate_catalog": [
                    {"record_id": "memory_attacker_controlled"}
                ],
            },
        )
        assert injected_catalog.status_code == 422
        assert resolver.calls == []

        wrong_request = http.post(
            "/api/lab/resolve",
            json={"request_text": "Forget an unrelated arbitrary record."},
        )
        assert wrong_request.status_code == 422
        assert wrong_request.json()["detail"]["code"] == (
            "request_must_match_committed_case"
        )
        assert resolver.calls == []

        resolved = _resolve(http).json()
        injected_selection = http.post(
            "/api/lab/confirm",
            json={
                "proposal_id": resolved["proposal_id"],
                "confirmed": True,
                "selected_record_id": "memory_attacker_controlled",
            },
        )
        assert injected_selection.status_code == 422

        confirmed = http.post(
            "/api/lab/confirm",
            json={
                "proposal_id": resolved["proposal_id"],
                "confirmed": True,
            },
        )
        assert confirmed.status_code == 200


def test_proposal_is_browser_bound_short_lived_and_one_use():
    clock = ResolverClock()
    store = ProposalStore(ttl_seconds=5, clock=clock)
    app = create_lab_app(
        mode="gemini",
        resolver=FakeGeminiResolver(),
        proposal_store=store,
    )

    with TestClient(app) as first, TestClient(app) as second:
        proposal_id = _resolve(first).json()["proposal_id"]

        cross_browser = second.post(
            "/api/lab/confirm",
            json={"proposal_id": proposal_id, "confirmed": True},
        )
        assert cross_browser.status_code == 404
        assert cross_browser.json()["detail"]["code"] == "proposal_not_found"

        clock.now = 6
        expired = first.post(
            "/api/lab/confirm",
            json={"proposal_id": proposal_id, "confirmed": True},
        )
        assert expired.status_code == 409
        assert expired.json()["detail"]["code"] == "proposal_expired"


def test_provider_failure_is_clear_and_never_falls_back_to_recorded_fixture():
    resolver = FakeGeminiResolver(
        ResolverUnavailable("resolver_provider_failed")
    )
    with TestClient(
        create_lab_app(mode="gemini", resolver=resolver)
    ) as http:
        response = _resolve(http)

        assert response.status_code == 502
        assert response.json()["detail"] == {
            "code": "resolver_provider_unavailable",
            "reason": "resolver_provider_failed",
        }
        assert http.app.state.proposals.snapshot_count() == 0


def test_gemini_mode_refuses_to_start_without_server_key():
    with pytest.raises(
        LabStartupError,
        match="GEMINI_API_KEY or GOOGLE_API_KEY",
    ):
        create_lab_app(mode="gemini", environ={})


def test_lab_env_does_not_override_a_terminal_key(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GEMINI_API_KEY=file-key\nGOOGLE_API_KEY=file-google-key\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GEMINI_API_KEY", "terminal-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    assert load_lab_environment(env_file) is True
    assert os.environ["GEMINI_API_KEY"] == "terminal-key"
    assert os.environ["GOOGLE_API_KEY"] == "file-google-key"


def test_gemini_mode_refuses_to_start_without_optional_dependency(monkeypatch):
    def dependency_missing(_cls, _api_key, *, model_id):
        del model_id
        raise ResolverUnavailable("resolver_client_not_installed")

    monkeypatch.setattr(
        GeminiResolver,
        "from_api_key",
        classmethod(dependency_missing),
    )
    with pytest.raises(LabStartupError, match="google-genai"):
        create_lab_app(
            mode="gemini",
            environ={"GEMINI_API_KEY": "test-key-that-is-never-used"},
        )


def test_launcher_serves_replay_and_redirects_to_lab_query():
    with TestClient(create_lab_app(mode="recorded")) as http:
        replay = http.get("/longmemeval_forgetting_replay.html")
        assert replay.status_code == 200
        assert 'id="replay-confirm"' in replay.text

        root = http.get("/", follow_redirects=False)
        assert root.status_code == 307
        assert root.headers["location"].endswith(
            "/longmemeval_forgetting_replay.html?resolver=lab"
        )


def test_cli_prints_exact_url_and_runs_single_same_origin_app(
    monkeypatch,
    capsys,
):
    import uvicorn

    called = {}

    def fake_run(app, **kwargs):
        called["app"] = app
        called["kwargs"] = kwargs

    monkeypatch.setattr(uvicorn, "run", fake_run)

    assert main(["--mode", "recorded", "--port", "8765"]) == 0
    output = capsys.readouterr().out
    assert "GemmaSV lab meeting demo (recorded)" in output
    assert (
        "http://127.0.0.1:8765/longmemeval_forgetting_replay.html?resolver=lab"
        in output
    )
    assert called["kwargs"] == {
        "host": "127.0.0.1",
        "port": 8765,
        "log_level": "warning",
    }
    assert called["app"].state.resolver.mode == "recorded"


def test_gemini_cli_loads_the_repository_env_file(monkeypatch, tmp_path):
    import uvicorn

    env_file = tmp_path / ".env"
    env_file.write_text(
        "GEMINI_API_KEY=local-server-key\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(lab_meeting_demo, "LAB_ENV_FILE", env_file)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    captured = {}

    def fake_from_api_key(_cls, api_key, *, model_id):
        captured["api_key"] = api_key
        captured["model_id"] = model_id
        return FakeGeminiResolver()

    monkeypatch.setattr(
        GeminiResolver,
        "from_api_key",
        classmethod(fake_from_api_key),
    )
    monkeypatch.setattr(uvicorn, "run", lambda *_args, **_kwargs: None)

    assert main(["--mode", "gemini", "--port", "8765"]) == 0
    assert captured == {
        "api_key": "local-server-key",
        "model_id": LAB_MODEL_ID,
    }
