import json
import time

from fastapi.testclient import TestClient

from gemma_sv.demo_server.api import create_app
from gemma_sv.demo_server.engine import ReplayDemoEngine
from gemma_sv.demo_server.resolver import (
    RecordedResolver,
    ResolverStatus,
)
from gemma_sv.demo_server.scenarios import MEDICINE
from gemma_sv.demo_server.settings import DemoSettings
from gemma_sv.demo_server.resolver import ProposalStore
from gemma_sv.demo_server.state import DemoPhase


class ResolverClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class FakeResolver:
    mode = "test"
    evidence = "test_resolution"
    network_requests = False
    available = True
    model_id = "resolver-test-model"

    def __init__(self, outcome="resolved"):
        self.outcome = outcome
        self.calls = []

    def resolve(self, request_text, candidates):
        self.calls.append((request_text, candidates))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        if self.outcome == "unknown":
            selected_record_id = "memory_hallucinated"
        else:
            selected_record_id = candidates[0].record_id
        return {
            "status": "resolved",
            "selected_record_id": selected_record_id,
            "alternative_record_ids": [],
            "confidence": 0.99,
            "explanation": "Matches the stored update.",
            "requires_confirmation": True,
            "deletion_executed": False,
        }


def client():
    settings = DemoSettings(
        engine="replay",
        review_mode=True,
        allowed_origins=("https://review.example",),
        requests_per_minute=1_000,
    )
    return TestClient(create_app(settings=settings, engine=ReplayDemoEngine()))


def resolver_client(
    resolver=None,
    *,
    settings=None,
    proposal_store=None,
):
    settings = settings or DemoSettings(
        engine="replay",
        review_mode=True,
        allowed_origins=("https://review.example",),
        requests_per_minute=1_000,
        resolver_mode="disabled",
    )
    return TestClient(
        create_app(
            settings=settings,
            engine=ReplayDemoEngine(),
            resolver=resolver,
            proposal_store=proposal_store,
        )
    )


def create_ingested_session(http: TestClient):
    created = http.post(
        "/api/v1/sessions", json={"domain": "medicine"}
    ).json()
    session_id = created["session_id"]
    ingested = http.post(
        f"/api/v1/sessions/{session_id}/ingest",
        json={
            "memory_text": MEDICINE.memory_text,
            "secret_start": MEDICINE.secret_start,
            "secret_end": MEDICINE.secret_end,
            "audit_probe": MEDICINE.audit_probe,
            "audit_target": MEDICINE.target_value,
            "delete_scope": MEDICINE.delete_scope,
        },
    )
    assert ingested.status_code == 200
    return session_id


def create_resolver_session(
    http: TestClient,
    *,
    record_id="memory_bikes",
    summary="The user bought a hybrid and now owns four bikes.",
):
    user_turn = "I bought a hybrid and now own four bikes."
    assistant_turn = "Congratulations on the new hybrid bike!"
    memory_text = f"User: {user_turn}\nAssistant: {assistant_turn}"
    session_id = http.post(
        "/api/v1/sessions",
        json={"domain": "medicine"},
    ).json()["session_id"]
    ingested = http.post(
        f"/api/v1/sessions/{session_id}/ingest",
        json={
            "memory_text": memory_text,
            "secret_start": 0,
            "secret_end": len(memory_text),
            "audit_probe": "\n\nQuestion: How many bikes do I own?\nAnswer:",
            "audit_target": "four bikes",
            "delete_scope": "record",
            "record_id": record_id,
            "memory_label": "Current number of bicycles",
            "memory_summary": summary,
            "source_turn_ids": ["turn_42", "turn_43"],
            "owned_exchange": [
                {
                    "turn_id": "turn_42",
                    "role": "user",
                    "text": user_turn,
                },
                {
                    "turn_id": "turn_43",
                    "role": "assistant",
                    "text": assistant_turn,
                },
            ],
        },
    )
    assert ingested.status_code == 200
    assert ingested.json()["resolver_record_id"] == record_id
    return session_id


def test_replay_ingest_serves_the_recorded_conversation_log():
    with client() as http:
        created = http.post(
            "/api/v1/sessions", json={"domain": "medicine"}
        ).json()
        ingested = http.post(
            f"/api/v1/sessions/{created['session_id']}/ingest",
            json={
                "memory_text": MEDICINE.memory_text,
                "secret_start": MEDICINE.secret_start,
                "secret_end": MEDICINE.secret_end,
                "audit_probe": MEDICINE.audit_probe,
                "audit_target": MEDICINE.target_value,
                "delete_scope": MEDICINE.delete_scope,
            },
        ).json()
        log = ingested["conversation_log"]
        records = [entry for entry in log if entry["kind"] == "record"]
        assert len(records) == ingested["memory_copies"]
        assert all(entry["text"] == MEDICINE.memory_text for entry in records)
        assert ingested["distance_beyond_window"] > ingested["local_window"]


def test_config_is_scoped_and_privacy_explicit():
    with client() as http:
        response = http.get("/api/v1/config")
        assert response.status_code == 200
        data = response.json()
        assert data["certificate_scope"].startswith("registered_audit_probe")
        assert data["privacy"]["real_secrets_allowed"] is False
        assert data["review_mode"] is True
        assert data["model_id"] == "google/gemma-3-4b-pt"


def test_resolver_environment_uses_server_only_key_fallback(monkeypatch):
    secret = "google-fallback-secret"
    monkeypatch.setenv("HERO_RESOLVER_MODE", "gemini")
    monkeypatch.setenv("HERO_RESOLVER_MODEL", "gemini-3.6-flash")
    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    monkeypatch.setenv("GOOGLE_API_KEY", secret)

    settings = DemoSettings.from_environment()

    assert settings.resolver_api_key == secret
    assert settings.resolver_model == "gemini-3.6-flash"
    assert secret not in repr(settings)


def test_record_deletion_requires_audit_target_inside_selected_record():
    with client() as http:
        session_id = http.post(
            "/api/v1/sessions", json={"domain": "custom"}
        ).json()["session_id"]
        text = "Fictional patient record: diagnosis is cedar fever."
        response = http.post(
            f"/api/v1/sessions/{session_id}/ingest",
            json={
                "memory_text": text,
                "secret_start": 0,
                "secret_end": len(text),
                "audit_target": "a value not in the record",
                "delete_scope": "record",
            },
        )
        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "audit_target_outside_deletion"


def test_record_ingest_accepts_disjoint_ranges_and_record_id():
    with client() as http:
        session_id = http.post(
            "/api/v1/sessions", json={"domain": "custom"}
        ).json()["session_id"]
        text = "diagnosis: cedar fever; ward: larkspur"
        diagnosis = "cedar fever"
        ward = "larkspur"
        response = http.post(
            f"/api/v1/sessions/{session_id}/ingest",
            json={
                "memory_text": text,
                "deletion_ranges": [
                    {
                        "start": text.index(diagnosis),
                        "end": text.index(diagnosis) + len(diagnosis),
                    },
                    {
                        "start": text.index(ward),
                        "end": text.index(ward) + len(ward),
                    },
                ],
                "audit_target": ward,
                "delete_scope": "record",
                "record_id": "patient-7",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["selected_values"] == [diagnosis, ward]
        assert data["deletion_scope"] == "record"
        assert data["record_id"] == "patient-7"


def test_ingest_requires_single_span_or_deletion_ranges():
    with client() as http:
        session_id = http.post(
            "/api/v1/sessions", json={"domain": "custom"}
        ).json()["session_id"]
        response = http.post(
            f"/api/v1/sessions/{session_id}/ingest",
            json={"memory_text": "synthetic record"},
        )
        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "invalid_deletion_ranges"


def test_full_replay_flow_and_purge():
    with client() as http:
        session_id = create_ingested_session(http)
        recall = http.post(f"/api/v1/sessions/{session_id}/recall")
        assert recall.status_code == 200
        assert recall.json()["evidence"] == "recorded_replay"

        forgotten = http.post(f"/api/v1/sessions/{session_id}/forget")
        assert forgotten.status_code == 200
        job_id = forgotten.json()["certificate_job"]["job_id"]
        headers = {"X-Demo-Session-ID": session_id}
        for _ in range(100):
            job = http.get(f"/api/v1/jobs/{job_id}", headers=headers)
            assert job.status_code == 200
            if job.json()["status"] == "succeeded":
                break
            time.sleep(0.005)
        assert job.json()["result"]["probe_scoped"] is True

        attack = http.post(
            f"/api/v1/sessions/{session_id}/attacks",
            json={"method": "exact", "kind": "extraction"},
        )
        assert attack.status_code == 200
        twin = http.post(f"/api/v1/sessions/{session_id}/twin")
        assert set(twin.json()["panes"]) == {"A", "B"}
        guess = http.post(
            f"/api/v1/sessions/{session_id}/twin/guess", json={"pane": "A"}
        )
        assert guess.status_code == 200
        assert guess.json()["scope"] == "registered_audit_probe"

        deleted = http.delete(f"/api/v1/sessions/{session_id}")
        assert deleted.status_code == 204
        assert (
            http.post(f"/api/v1/sessions/{session_id}/recall").status_code == 404
        )
        assert http.get(f"/api/v1/jobs/{job_id}", headers=headers).status_code == 404


def test_invalid_or_cross_session_requests_do_not_leak_state():
    with client() as http:
        first = create_ingested_session(http)
        second = http.post(
            "/api/v1/sessions", json={"domain": "medicine"}
        ).json()["session_id"]
        assert (
            http.post(
                f"/api/v1/sessions/{second}/ingest",
                json={
                    "memory_text": "short",
                    "secret_start": 2,
                    "secret_end": 99,
                },
            ).status_code
            == 422
        )
        forgotten = http.post(f"/api/v1/sessions/{first}/forget").json()
        job_id = forgotten["certificate_job"]["job_id"]
        wrong = http.get(
            f"/api/v1/jobs/{job_id}",
            headers={"X-Demo-Session-ID": second},
        )
        assert wrong.status_code == 404


def test_cors_allows_only_configured_pages_origin():
    with client() as http:
        allowed = http.options(
            "/api/v1/config",
            headers={
                "Origin": "https://review.example",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert allowed.headers["access-control-allow-origin"] == "https://review.example"
        denied = http.options(
            "/api/v1/config",
            headers={
                "Origin": "https://attacker.example",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert "access-control-allow-origin" not in denied.headers


def test_resolver_is_unavailable_without_api_key_and_does_not_mutate_memory():
    settings = DemoSettings(
        engine="replay",
        allowed_origins=("https://review.example",),
        requests_per_minute=1_000,
        resolver_mode="gemini",
        resolver_api_key=None,
    )
    with resolver_client(settings=settings) as http:
        session_id = create_resolver_session(http)
        response = http.post(
            f"/api/v1/sessions/{session_id}/memory/resolve",
            json={"request_text": "Forget the update about my bikes."},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == ResolverStatus.UNAVAILABLE.value
        assert data["proposal_id"] is None
        assert data["deletion_executed"] is False
        assert data["resolver"]["network_request_performed"] is False
        session = http.app.state.demo.sessions.get(session_id, touch=False)
        assert session.phase is DemoPhase.INGESTED
        assert "forget" not in session.results


def test_public_config_and_responses_never_expose_api_key_or_private_receipt():
    api_key = "gemini-super-secret-test-key"
    settings = DemoSettings(
        engine="replay",
        allowed_origins=("https://review.example",),
        requests_per_minute=1_000,
        resolver_mode="gemini",
        resolver_api_key=api_key,
    )
    resolver = FakeResolver()
    with resolver_client(resolver, settings=settings) as http:
        session_id = create_resolver_session(http)
        config_text = http.get("/api/v1/config").text
        resolution = http.post(
            f"/api/v1/sessions/{session_id}/memory/resolve",
            json={"request_text": "Forget my current bike count."},
        )
        response_text = json.dumps(resolution.json())

        assert api_key not in config_text
        assert api_key not in response_text
        assert api_key not in repr(settings)
        assert "authorization_scope" not in response_text
        assert "source_message_hashes" not in response_text
        assert "owned_ranges" not in response_text
        assert resolution.json()["selected_record"]["owned_exchange"] == [
            {
                "turn_id": "turn_42",
                "role": "user",
                "text": "I bought a hybrid and now own four bikes.",
            },
            {
                "turn_id": "turn_43",
                "role": "assistant",
                "text": "Congratulations on the new hybrid bike!",
            },
        ]


def test_resolution_requires_explicit_confirmation_before_deletion():
    resolver = FakeResolver()
    with resolver_client(resolver) as http:
        session_id = create_resolver_session(http)
        resolved = http.post(
            f"/api/v1/sessions/{session_id}/memory/resolve",
            json={"request_text": "Forget the update about my bicycles."},
        )
        assert resolved.status_code == 200
        proposal_id = resolved.json()["proposal_id"]
        assert proposal_id.startswith("proposal_")
        assert resolved.json()["deletion_executed"] is False
        session = http.app.state.demo.sessions.get(session_id, touch=False)
        assert session.phase is DemoPhase.INGESTED
        assert "forget" not in session.results

        confirmed = http.post(
            f"/api/v1/sessions/{session_id}/memory/confirm",
            json={"proposal_id": proposal_id, "confirmed": True},
        )
        assert confirmed.status_code == 200
        data = confirmed.json()
        assert data["selected_record_id"] == "memory_bikes"
        assert data["confirmed"] is True
        assert data["deletion_executed"] is True
        assert data["forget"]["evidence"] == "recorded_replay"
        assert data["certificate_job"]["job_id"]
        assert session.phase is DemoPhase.FORGOTTEN

        reused = http.post(
            f"/api/v1/sessions/{session_id}/memory/confirm",
            json={"proposal_id": proposal_id, "confirmed": True},
        )
        assert reused.status_code == 409
        assert reused.json()["detail"]["code"] == "proposal_already_used"


def test_actionable_resolution_requires_complete_owned_exchange():
    with resolver_client(FakeResolver()) as http:
        session_id = create_ingested_session(http)
        response = http.post(
            f"/api/v1/sessions/{session_id}/memory/resolve",
            json={"request_text": "Forget the stored memory."},
        )

        assert response.status_code == 409
        assert (
            response.json()["detail"]["code"]
            == "confirmation_context_unavailable"
        )
        session = http.app.state.demo.sessions.get(session_id, touch=False)
        assert session.phase is DemoPhase.INGESTED
        assert "forget" not in session.results


def test_proposal_is_session_bound_and_expires():
    clock = ResolverClock()
    store = ProposalStore(ttl_seconds=5, clock=clock)
    resolver = FakeResolver()
    with resolver_client(resolver, proposal_store=store) as http:
        first = create_resolver_session(http, record_id="memory_first")
        second = create_resolver_session(http, record_id="memory_second")
        proposal_id = http.post(
            f"/api/v1/sessions/{first}/memory/resolve",
            json={"request_text": "Forget the first memory."},
        ).json()["proposal_id"]

        cross_session = http.post(
            f"/api/v1/sessions/{second}/memory/confirm",
            json={"proposal_id": proposal_id, "confirmed": True},
        )
        assert cross_session.status_code == 404
        assert cross_session.json()["detail"]["code"] == "proposal_not_found"

        clock.now = 6
        expired = http.post(
            f"/api/v1/sessions/{first}/memory/confirm",
            json={"proposal_id": proposal_id, "confirmed": True},
        )
        assert expired.status_code == 409
        assert expired.json()["detail"]["code"] == "proposal_expired"
        first_state = http.app.state.demo.sessions.get(first, touch=False)
        assert first_state.phase is DemoPhase.INGESTED


def test_catalog_change_invalidates_proposal_before_deletion():
    resolver = FakeResolver()
    with resolver_client(resolver) as http:
        session_id = create_resolver_session(http)
        proposal_id = http.post(
            f"/api/v1/sessions/{session_id}/memory/resolve",
            json={"request_text": "Forget the bike count."},
        ).json()["proposal_id"]
        context = http.app.state.demo
        session = http.app.state.demo.sessions.get(session_id, touch=False)
        catalog = context.memory_catalog(session_id)
        original = catalog["memory_bikes"]
        catalog["memory_bikes"] = original.model_copy(
            update={"summary": "The user now owns five bikes."}
        )

        response = http.post(
            f"/api/v1/sessions/{session_id}/memory/confirm",
            json={"proposal_id": proposal_id, "confirmed": True},
        )
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "proposal_catalog_changed"
        assert session.phase is DemoPhase.INGESTED
        assert "forget" not in session.results


def test_ambiguous_resolution_requires_manual_catalog_selection():
    resolver = RecordedResolver(
        {
            "status": "ambiguous",
            "selected_record_id": None,
            "alternative_record_ids": [
                "memory_bikes",
                "memory_bike_trip",
            ],
            "confidence": 0.5,
            "explanation": "Two saved updates are plausible.",
            "requires_confirmation": True,
            "deletion_executed": False,
        },
        model_id="gemini-recorded",
    )
    with resolver_client(resolver) as http:
        session_id = create_resolver_session(http)
        catalog = http.app.state.demo.memory_catalog(session_id)
        active = catalog["memory_bikes"]
        catalog["memory_bike_trip"] = active.model_copy(
            update={
                "record_id": "memory_bike_trip",
                "label": "Bicycle tour plan",
                "summary": "The user planned a separate bicycle trip.",
            }
        )
        resolved = http.post(
            f"/api/v1/sessions/{session_id}/memory/resolve",
            json={"request_text": "Forget the bike memory."},
        )
        assert resolved.status_code == 200
        assert resolved.json()["status"] == "ambiguous"
        assert len(resolved.json()["alternative_records"]) == 2
        proposal_id = resolved.json()["proposal_id"]

        missing_selection = http.post(
            f"/api/v1/sessions/{session_id}/memory/confirm",
            json={"proposal_id": proposal_id, "confirmed": True},
        )
        assert missing_selection.status_code == 422
        assert (
            missing_selection.json()["detail"]["code"]
            == "proposal_selection_required"
        )

        confirmed = http.post(
            f"/api/v1/sessions/{session_id}/memory/confirm",
            json={
                "proposal_id": proposal_id,
                "confirmed": True,
                "selected_record_id": "memory_bikes",
            },
        )
        assert confirmed.status_code == 200
        assert confirmed.json()["deletion_executed"] is True


def test_existing_but_unauthorized_catalog_record_is_refused():
    resolver = RecordedResolver(
        {
            "status": "resolved",
            "selected_record_id": "memory_unowned",
            "alternative_record_ids": [],
            "confidence": 0.95,
            "explanation": "Matches the second catalog entry.",
            "requires_confirmation": True,
            "deletion_executed": False,
        },
        model_id="gemini-recorded",
    )
    with resolver_client(resolver) as http:
        session_id = create_resolver_session(http)
        context = http.app.state.demo
        session = http.app.state.demo.sessions.get(session_id, touch=False)
        catalog = context.memory_catalog(session_id)
        active = catalog["memory_bikes"]
        catalog["memory_unowned"] = active.model_copy(
            update={
                "record_id": "memory_unowned",
                "label": "Unowned catalog entry",
            }
        )
        proposal_id = http.post(
            f"/api/v1/sessions/{session_id}/memory/resolve",
            json={"request_text": "Forget the unowned entry."},
        ).json()["proposal_id"]

        response = http.post(
            f"/api/v1/sessions/{session_id}/memory/confirm",
            json={"proposal_id": proposal_id, "confirmed": True},
        )
        assert response.status_code == 403
        assert response.json()["detail"]["code"] == "unauthorized_memory_record"
        assert session.phase is DemoPhase.INGESTED
        assert "forget" not in session.results


def test_resolver_failure_or_unknown_id_leaves_memory_unchanged():
    for outcome, expected_status in (
        (RuntimeError("provider leaked detail"), 200),
        ("unknown", 502),
    ):
        resolver = FakeResolver(outcome)
        with resolver_client(resolver) as http:
            session_id = create_resolver_session(http)
            response = http.post(
                f"/api/v1/sessions/{session_id}/memory/resolve",
                json={"request_text": "Forget the bike update."},
            )
            assert response.status_code == expected_status
            if expected_status == 200:
                assert response.json()["status"] == "unavailable"
                assert "provider leaked detail" not in response.text
            else:
                assert (
                    response.json()["detail"]["reason"]
                    == "resolver_response_unknown_id"
                )
            session = http.app.state.demo.sessions.get(
                session_id,
                touch=False,
            )
            assert session.phase is DemoPhase.INGESTED
            assert "forget" not in session.results


def test_recorded_resolution_is_labeled_and_performs_no_network_request():
    resolver = RecordedResolver(
        {
            "status": "resolved",
            "selected_record_id": "memory_bikes",
            "alternative_record_ids": [],
            "confidence": 1.0,
            "explanation": "Recorded match for the bicycle update.",
            "requires_confirmation": True,
            "deletion_executed": False,
        },
        model_id="gemini-recorded",
    )
    with resolver_client(resolver) as http:
        session_id = create_resolver_session(http)
        response = http.post(
            f"/api/v1/sessions/{session_id}/memory/resolve",
            json={"request_text": "Forget the bike update."},
        )

        assert response.status_code == 200
        source = response.json()["resolver"]
        assert source["label"] == "recorded Gemini resolver outcome"
        assert source["evidence"] == "recorded_resolution"
        assert source["network_request_performed"] is False
        assert response.json()["deletion_executed"] is False


def test_resolver_endpoint_uses_existing_api_rate_limit():
    settings = DemoSettings(
        engine="replay",
        allowed_origins=("https://review.example",),
        requests_per_minute=3,
        resolver_mode="disabled",
    )
    with resolver_client(FakeResolver(), settings=settings) as http:
        session_id = create_resolver_session(http)
        first = http.post(
            f"/api/v1/sessions/{session_id}/memory/resolve",
            json={"request_text": "Forget the bike update."},
        )
        second = http.post(
            f"/api/v1/sessions/{session_id}/memory/resolve",
            json={"request_text": "Forget the bike update."},
        )

        assert first.status_code == 200
        assert second.status_code == 429
        assert second.json()["detail"]["code"] == "rate_limited"
