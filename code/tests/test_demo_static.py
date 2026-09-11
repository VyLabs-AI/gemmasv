import json
from pathlib import Path

from gemma_sv.demo_server.gemma_engine import _QA_FILLERS
from gemma_sv.demo_server.scenarios import (
    PRESETS,
    scaffold_memory,
    scaffold_offset,
)


SITE = Path(__file__).parents[1] / "gemma_sv" / "demo_site"


def test_replay_presets_select_record_and_audit_target_separately():
    replay = json.loads((SITE / "assets" / "replay.json").read_text())
    expected = {
        "cybersecurity": "credential replay",
        "medicine": "acute intermittent porphyria",
        "tofu": "civil engineer",
    }
    field_presets = [
        preset
        for preset in replay["config"]["presets"]
        if preset["delete_scope"] == "span"
    ]
    assert {preset["domain"] for preset in field_presets} == set(expected)
    for preset in field_presets:
        selected = preset["memory_text"][preset["secret_start"] : preset["secret_end"]]
        assert selected == expected[preset["domain"]]
        assert preset["selected_value"] == selected
        assert preset["audit_target"] == expected[preset["domain"]]
        assert preset["audit_target"] in selected
        if preset["domain"] != "tofu":  # the TOFU pair is a bare Q/A record
            assert len(preset["conversation"]) >= 4


def test_replay_presets_follow_the_chat_scaffold_recipe():
    """The committed presets must match the client/server scaffolding contract."""

    replay = json.loads((SITE / "assets" / "replay.json").read_text())
    for payload in replay["config"]["presets"]:
        if payload["delete_scope"] != "span":
            continue
        preset = PRESETS[payload["slug"]]
        span_start = preset.fact.index(preset.selected_value)
        assert payload["memory_text"] == scaffold_memory(preset.question, preset.fact)
        assert payload["secret_start"] == scaffold_offset(preset.question) + span_start
        assert payload["audit_probe"] == preset.audit_probe
        # The registered probe must never leak the short audited answer.
        assert preset.target_value not in payload["audit_probe"]


def test_tofu_preset_is_attributed_and_matches_the_benchmark_protocol():
    """The TOFU-derived preset must credit its source and mirror the paper probe."""

    preset = PRESETS["tofu-forget10-2"]
    assert preset.domain == "tofu"
    assert preset.guided is False
    assert "TOFU" in preset.description and "locuslab/TOFU" in preset.description
    assert preset.selected_value == "civil engineer"
    assert preset.selected_value not in preset.question
    # Stem probe ends right before the secret span, never containing it.
    assert preset.audit_probe.endswith("is a")
    assert preset.selected_value not in preset.audit_probe


def test_whole_record_presets_expose_full_scope_without_fake_replay():
    whole = [
        preset for preset in PRESETS.values()
        if preset.delete_scope == "record"
    ]
    assert len(whole) == 8
    for preset in whole:
        payload = preset.public_dict()
        assert payload["guided"] is False
        assert payload["record_id"] == preset.slug
        assert payload["deletion_ranges"] == [
            {"start": 0, "end": len(preset.memory_text)}
        ]
        assert preset.target_value in preset.memory_text
        assert (
            "Passed every 1B" in preset.description
            or "Failed one 1B" in preset.description
        )


def test_replay_runs_are_complete_and_recalled():
    replay = json.loads((SITE / "assets" / "replay.json").read_text())
    assert set(replay["runs"]) == {"medicine", "whole-case-zaffre"}
    for run in replay["runs"].values():
        assert run["recall"]["admission"]["status"] == "recalled"
        assert set(run["attacks"]) == {"icul", "exact"}
        assert set(run["twins"]) == {"exact", "refit"}
        assert run["certificate"]["kl_nats"] >= 0


def test_whole_record_replay_run_carries_all_field_audits_and_neighbor():
    replay = json.loads((SITE / "assets" / "replay.json").read_text())
    run = replay["runs"]["whole-case-zaffre"]
    assert run["deletion_scope"] == "record"
    assert len(run["field_audits"]) == 3
    for field in run["field_audits"]:
        assert field["deleted_probability"] <= field["present_probability"]
        # Deleted field lands at or below its never-stored floor (small slack).
        assert field["deleted_probability"] <= field["never_probability"] + 0.05
    neighbor = run["neighbor"]
    assert neighbor["deleted_probability"] >= neighbor["present_probability"] - 0.05
    assert run["certificate"]["kl_nats"] <= 1e-6


def test_replay_runs_carry_the_visible_conversation_log():
    """The UI shows the packed memory as chat history; it must match the preset."""

    replay = json.loads((SITE / "assets" / "replay.json").read_text())
    presets = {preset["domain"]: preset for preset in replay["config"]["presets"]}
    scaffold_runs = {
        domain: run
        for domain, run in replay["runs"].items()
        if "conversation_log" in run
    }
    assert "medicine" in scaffold_runs
    for domain, run in scaffold_runs.items():
        log = run["conversation_log"]
        records = [entry for entry in log if entry["kind"] == "record"]
        exchanges = [entry for entry in log if entry["kind"] == "exchange"]
        assert len(records) == run["memory_copies"]
        for record in records:
            assert record["text"] == presets[domain]["memory_text"]
        assert run["distance_beyond_window"] > run["local_window"]
        assert len(exchanges) >= 20
        assert all(entry["text"].startswith("User: ") for entry in exchanges)
        # The log must end inside the local window (recent side of the boundary).
        assert log[-1]["beyond_window"] is False
        # Distances decrease monotonically toward the end of the log.
        distances = [entry["tokens_from_end"] for entry in log]
        assert distances == sorted(distances, reverse=True)


def test_conversational_padding_keeps_the_piloted_constraints():
    """The padding constraints were chosen from the filler-format pilots.

    The sample conversations must use the User/Assistant dialogue format (distinct
    from the visitor's Question:/Answer: record), must not use the word
    "fictional" (which the preset questions use), and must not contain any
    preset's protected value.
    """

    values = [preset.target_value for preset in PRESETS.values()]
    assert len(_QA_FILLERS) == 32
    for filler in _QA_FILLERS:
        assert filler.startswith("User: ")
        assert " Assistant: " in filler
        assert "Question:" not in filler and "Answer:" not in filler
        assert "fictional" not in filler.casefold()
        for value in values:
            assert value.casefold() not in filler.casefold()


def test_static_entrypoint_references_only_shipped_assets():
    html = (SITE / "index.html").read_text()
    app = (SITE / "app.js").read_text()
    for relative_path in ("style.css", "config.js", "app.js", "assets/replay.json"):
        assert (SITE / relative_path).exists()
    assert 'src="config.js"' in html
    assert 'src="app.js"' in html
    assert "Use synthetic information only" in html
    assert "Think of the model&rsquo;s memory as a filing cabinet" in html
    assert "Behavioral evidence" in html
    assert "Certificate" in html
    assert "Blind comparison: deleted vs. never stored" in app
    assert "a correct guess is only luck" in app


def test_longmemeval_replay_uses_the_final_hash_bound_case_asset():
    html = (SITE / "longmemeval_forgetting_replay.html").read_text()
    app = (SITE / "longmemeval_forgetting_replay.js").read_text()
    payload = json.loads(
        (SITE / "assets" / "longmemeval_forgetting_final.json").read_text()
    )
    assert "longmemeval_forgetting_final.json" in app
    assert "single-case incomplete preview" not in html
    assert payload["status"] == "final_case_study"
    assert payload["preview_label"] is None
    assert payload["scope"]["final_all16_complete"] is True
    assert payload["scope"]["aggregate_claims"] == []
    assert payload["demo"]["live_compute"] is False
    assert payload["demo"]["network_api_required"] is False
    assert [step["id"] for step in payload["demo"]["sequence"]] == [
        "chat",
        "recall",
        "resolve-request",
        "resolve-proposal",
        "confirm",
        "forget",
        "re-query",
        "retained",
        "certificate",
    ]
    assert payload["provenance"]["source_artifacts"]["method_report"][
        "file_sha256"
    ] == "a7d0582d7d5aa6832320852f0b80e79799621d6520ebef5f7a44eeea0e327fb6"
    certificate = payload["case"]["certificate"]
    assert certificate["decrement_fallbacks"]["value"] == 522
    assert certificate["head_gate_solves"]["value"] == 560
    assert certificate["head_gates"]["value"] == 960


def test_longmemeval_static_default_is_network_free_with_opt_in_lab_api():
    html = (SITE / "longmemeval_forgetting_replay.html").read_text()
    app = (SITE / "longmemeval_forgetting_replay.js").read_text()
    payload = json.loads(
        (SITE / "assets" / "longmemeval_forgetting_final.json").read_text()
    )
    resolution = payload["case"]["memory_resolution"]
    assert resolution["mode"] == "deterministic_recorded_fixture"
    assert resolution["provider"]["call_performed"] is False
    assert resolution["provider"]["artifact_present"] is False
    assert resolution["network_request_performed"] is False
    assert resolution["model_configuration"]["model_produced_fixture"] is False
    assert resolution["decision"]["requires_confirmation"] is True
    assert resolution["decision"]["deletion_executed"] is False
    catalog_ids = {
        candidate["record_id"]
        for candidate in resolution["candidate_catalog"]
    }
    assert resolution["decision"]["selected_record_id"] in catalog_ids
    retained = next(
        candidate
        for candidate in resolution["candidate_catalog"]
        if candidate["owned_exchange_ref"]
        == "/case/dialogue/retained/quotes"
    )
    assert resolution["decision"]["selected_record_id"] != retained["record_id"]
    confirmation = resolution["confirmation"]
    assert confirmation["selected_record_id"] == (
        resolution["decision"]["selected_record_id"]
    )
    assert confirmation["catalog_hash"] == resolution["catalog_hash"]
    assert confirmation["requires_explicit_click"] is True
    assert confirmation["confirmed_in_payload"] is False
    assert confirmation["owned_exchange"] == (
        payload["case"]["dialogue"]["target"]["quotes"]
    )
    assert len(confirmation["owned_exchange"]) == 2

    assert 'id="replay-confirm"' in html
    assert "Confirm and delete this memory" in html
    assert "awaitingConfirmation" in app
    assert "deletionConfirmed" in app
    assert 'STAGE_IDS[nextStage] === "forget"' in app
    assert "deletion requires explicit confirmation" in app
    assert "confirmAndDelete" in app
    assert "deletionConfirmed = false" in app
    assert 'get("resolver") === "lab"' in app
    assert "if (!labResolverEnabled) return;" in app
    assert "fetchJson(PAYLOAD_URL)" in app
    assert 'const LAB_RESOLVE_URL = "/api/lab/resolve"' in app
    assert 'const LAB_CONFIRM_URL = "/api/lab/confirm"' in app
    assert "request_text: resolution.request.text" in app
    assert "proposal_id: labResolution.proposal_id" in app
    assert "candidate_catalog:" not in app
    assert "selected_record_id: selected.record_id" not in app
    assert "live_model_deletion_run === false" in app
    assert "innerHTML" not in app


def test_repository_default_cannot_send_text_to_a_backend():
    config = (SITE / "config.js").read_text()
    assert "forceReplay: !localApi" in config
    assert r"127\.0\.0\.1|localhost" in config
    assert "https://" not in config
