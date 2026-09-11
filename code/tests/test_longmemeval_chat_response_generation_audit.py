from __future__ import annotations

from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gemma_sv import longmemeval_chat_response_generation_audit as audit


class FakeTokenizer:
    is_fast = True
    vocab_size = 262144
    eos_token_id = 1
    bos_token_id = 2
    pad_token_id = 0

    def __len__(self):
        return 262145

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return SimpleNamespace(
            input_ids=list(range(1, max(2, len(str(text).split()) + 1)))
        )

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens,
        clean_up_tokenization_spaces,
    ):
        assert clean_up_tokenization_spaces is False
        values = [int(token_id) for token_id in token_ids]
        pieces = {
            10: "blue",
            11: "green",
            12: "amber",
            106: "<end_of_turn>",
            1: "<eos>",
        }
        rendered = "".join(pieces.get(token_id, f"[{token_id}]") for token_id in values)
        if skip_special_tokens:
            rendered = rendered.replace("<end_of_turn>", "").replace("<eos>", "")
        return rendered


class FakeMemory:
    def __init__(self, token_ids, *, kind):
        self.token_ids = tuple(token_ids)
        self.token_count = len(self.token_ids)
        self.input_digest = audit.geometry.base.token_ids_sha256(self.token_ids)
        self.kind = kind
        self.deletion_kind = kind
        self.deleted_positions = ()
        self.request = SimpleNamespace(gate_floor=0.0)

    def fork(self):
        return copy.deepcopy(self)


class FakeBackend:
    def to_str(self):
        return "fake-backend"


class FakeRuntime:
    def __init__(
        self,
        *,
        nondeterministic=False,
        fail_proxy=False,
        fail_generation_call=None,
    ):
        self.tokenizer = FakeTokenizer()
        self.generation_calls = 0
        self.nondeterministic = nondeterministic
        self.fail_proxy = fail_proxy
        self.fail_generation_call = fail_generation_call

    def ensure_loaded(self):
        return None

    def prefill_persistent(self, token_ids):
        kind = "present" if tuple(token_ids) == (1, 2, 3, 4, 5) else "raw"
        return FakeMemory(token_ids, kind=kind)

    def delete_persistent(self, memory, forget_positions, *, kind):
        del forget_positions
        if self.fail_proxy:
            raise RuntimeError("synthetic proxy failure")
        result = memory.fork()
        result.kind = "proxy"
        result.deletion_kind = kind
        return result

    def persistent_certificate_states(self, memory, forget_positions):
        del forget_positions
        exact = memory.fork()
        exact.kind = "exact"
        exact.deletion_kind = "float64_exact"
        refit = memory.fork()
        refit.kind = "refit"
        refit.deletion_kind = "float64_refit"
        return (
            {
                "fixed_c_feasible": True,
                "used_refit_fallback": True,
                "decrement_fallbacks": 1,
            },
            {"exact": exact, "refit": refit},
        )

    @contextmanager
    def _persistent_branch(self, memory):
        branch = memory.fork()
        branch.step = 0
        yield branch

    def _persistent_prompt(self, branch, prompt):
        self.generation_calls += 1
        if self.generation_calls == self.fail_generation_call:
            raise RuntimeError("synthetic generation failure")
        branch.prompt = prompt
        token = 10 if "current" in prompt.casefold() else 11
        if self.nondeterministic and self.generation_calls % 2 == 0:
            token = 12
        return _logits(token)

    def _persistent_step(self, branch, token_id):
        del branch, token_id
        return _logits(106)


def _logits(token_id):
    values = np.full(128, -100.0)
    values[int(token_id)] = 1.0
    return values


def fixture_record(record_id="record-0"):
    original = (1, 2, 3, 4, 5)
    edited = (1, 3, 4, 5)
    context = SimpleNamespace(
        original_text="The current answer is blue and the retained answer is green.",
        edited_text="The retained answer is green.",
        original_token_ids=original,
        edited_token_ids=edited,
        forget_positions=(1,),
    )
    target_prompt = audit.geometry.chat_v1.render_constrained_query(
        "What is current?", "2026-08-24"
    )
    retained_prompt = audit.geometry.chat_v1.render_constrained_query(
        "What remains?", "2026-08-24"
    )
    return SimpleNamespace(
        record_id=record_id,
        context=context,
        raw_omitted_token_ids=edited,
        probes=(
            SimpleNamespace(
                probe_id="target_current",
                kind="deleted",
                question="What is current?",
                answer="blue",
                prompt_text=target_prompt,
                target_token_ids=(10,),
            ),
            SimpleNamespace(
                probe_id="retained",
                kind="retained",
                question="What remains?",
                answer="green",
                prompt_text=retained_prompt,
                target_token_ids=(11,),
            ),
        ),
    )


def fixture_prompt_binding(record):
    return {
        "record_id": record.record_id,
        "probes": [
            {
                "probe_id": probe.probe_id,
                "kind": probe.kind,
                "prompt_utf8_sha256": audit._text_sha256(probe.prompt_text),
                "question_sha256": audit._text_sha256(probe.question),
                "answer_sha256": audit._text_sha256(probe.answer),
                "target_token_count": len(probe.target_token_ids),
                "target_token_ids_sha256": (
                    audit.geometry.base.token_ids_sha256(
                        probe.target_token_ids
                    )
                ),
            }
            for probe in record.probes
        ],
    }


def fixture_runtime_lock(records):
    ids = [record.record_id for record in records]
    lock = {
        "cohort": {
            "ordered_record_ids": ids,
            "ordered_record_ids_sha256": audit._payload_sha256(ids),
        },
        "prompt_contract": {
            "base_probe_bytes": [
                fixture_prompt_binding(record) for record in records
            ],
        },
        "runtime": {
            "environment": copy.deepcopy(audit.PINNED_RUNTIME_ENVIRONMENT),
        },
        "generation": {
            "stop_token_ids": list(audit.STOP_TOKEN_IDS),
        },
        "privacy_and_scope": {
            "longmemeval": {
                "dataset_id": audit.geometry.base.DATASET_ID,
                "public_provenance": True,
            },
        },
        "conditions": {
            "condition_ids": list(audit.CONDITION_IDS),
            "excluded_condition_ids": list(audit.EXCLUDED_CONDITION_IDS),
        },
        "attempt_policy": {
            "resume_allowed": False,
            "overwrite_allowed": False,
        },
        "output_policy": {
            "canonical_workspace_path": "outputs/audit.json",
        },
        "integrity": {"sha256": "a" * 64},
    }
    return lock


@pytest.fixture(scope="module")
def protocol():
    required = (
        audit.DEFAULT_FINAL_REPORT,
        audit.DEFAULT_MANIFEST,
        audit.DEFAULT_METHOD_LOCK,
        audit.DEFAULT_CORE_MANIFEST,
        audit.DEFAULT_AUTHORIZATION_LOCK,
    )
    if not all(path.is_file() for path in required):
        pytest.skip("standalone release excludes the bound local final report")
    final_report, manifest, method_lock = audit._load_protocol_inputs()
    lock = audit.freeze_authorization_lock(final_report, manifest, method_lock)
    return SimpleNamespace(
        final_report=final_report,
        manifest=manifest,
        method_lock=method_lock,
        lock=lock,
    )


def test_pre_run_lock_binds_final_report_cohort_runtime_and_generation(protocol):
    lock = protocol.lock
    assert lock["artifacts"]["final_report"]["file_sha256"] == (
        "a7d0582d7d5aa6832320852f0b80e79799621d6520ebef5f7a44eeea0e327fb6"
    )
    assert len(lock["cohort"]["ordered_record_ids"]) == 16
    assert len(lock["cohort"]["joint10"]["record_ids"]) == 10
    assert lock["cohort"]["joint10"]["selected_before_generated_outcomes"] is True
    assert lock["conditions"]["condition_ids"] == list(audit.CONDITION_IDS)
    assert audit.methods.CACHE_DELETE_SHIFT_ID not in lock["conditions"][
        "condition_ids"
    ]
    assert lock["conditions"]["token_row_alias"] == {
        "condition_id": audit.methods.TOKEN_ROW_DIAGNOSTIC_ID,
        "alias_of": audit.methods.FRESH_RAW_OMISSION_ID,
        "additional_model_execution": False,
        "identical_registered_turn_token_ids_required": True,
    }
    generation = lock["generation"]
    assert generation["algorithm"] == "direct-step greedy argmax"
    assert generation["do_sample"] is False
    assert generation["max_new_tokens"] == 64
    assert generation["repeat_generations"] == 2
    assert generation["planned_workload"]["generation_calls"] == 448
    assert generation["planned_workload"]["maximum_generated_token_steps"] == 28672
    assert generation["teacher_forced_scores_used_to_infer_text"] is False
    assert lock["runtime"]["environment"] == audit.PINNED_RUNTIME_ENVIRONMENT
    assert lock["runtime"]["model_and_tokenizer"]["network_access"] is False
    assert (
        lock["runtime"]["model_and_tokenizer"]["tokenizer_backend_sha256"]
        == audit.TOKENIZER_BACKEND_SHA256
    )
    assert lock["generation_implementation"]["file_count"] == len(
        audit._IMPLEMENTATION_PATHS
    )
    assert lock["presentation_case"]["selected_before_generated_outcomes"] is True
    assert lock["output_policy"]["release_authorized"] is False
    assert lock["privacy_and_scope"]["decoded_strings_exhaust_extractability"] is False
    audit._assert_source_free_lock(lock)
    audit.validate_authorization_lock(
        lock,
        final_report=protocol.final_report,
        manifest=protocol.manifest,
        method_lock=protocol.method_lock,
    )


def test_committed_lock_reproduces_byte_for_value(protocol):
    committed = audit._load_mapping(
        audit.DEFAULT_AUTHORIZATION_LOCK,
        name="committed response-generation lock",
    )
    assert committed == protocol.lock


def test_resigned_generation_or_case_tamper_fails(protocol):
    for mutate in (
        lambda lock: lock["generation"].__setitem__("max_new_tokens", 63),
        lambda lock: lock["presentation_case"].__setitem__(
            "record_id", lock["cohort"]["ordered_record_ids"][0]
        ),
    ):
        changed = copy.deepcopy(protocol.lock)
        mutate(changed)
        unsigned = dict(changed)
        unsigned.pop("integrity")
        changed["integrity"] = {
            "algorithm": "sha256",
            "sha256": audit._payload_sha256(unsigned),
        }
        with pytest.raises(audit.AuditError, match="differs from frozen"):
            audit.validate_authorization_lock(
                changed,
                final_report=protocol.final_report,
                manifest=protocol.manifest,
                method_lock=protocol.method_lock,
            )


@pytest.mark.parametrize("bad_ack", [False, 1, "true", object()])
def test_run_acknowledgement_is_exact_true(monkeypatch, bad_ack):
    called = False

    def forbidden(**kwargs):
        nonlocal called
        called = True
        raise AssertionError(kwargs)

    monkeypatch.setattr(audit, "_load_protocol_inputs", forbidden)
    with pytest.raises(PermissionError, match="exactly True"):
        audit.run_audit_from_paths(explicit_acknowledgement=bad_ack)
    assert called is False


def test_uncommitted_lock_fails_before_runtime(protocol, monkeypatch):
    called = False

    def reject(path):
        raise PermissionError(f"{path} must be committed at HEAD")

    def forbidden_environment(lock):
        nonlocal called
        called = True
        raise AssertionError(lock)

    monkeypatch.setattr(audit, "_require_head_committed_file", reject)
    monkeypatch.setattr(audit, "verify_runtime_environment", forbidden_environment)
    with pytest.raises(PermissionError, match="committed at HEAD"):
        audit.load_committed_authorization_lock(
            audit.DEFAULT_AUTHORIZATION_LOCK,
            final_report=protocol.final_report,
            manifest=protocol.manifest,
            method_lock=protocol.method_lock,
        )
    assert called is False


def test_direct_greedy_generation_stops_and_hashes_actual_tokens(monkeypatch):
    runtime = FakeRuntime()
    memory = FakeMemory((1, 2), kind="present")
    monkeypatch.setattr(audit, "_inference_context", lambda: audit.nullcontext())
    response = audit.greedy_generate_response(
        runtime,
        memory,
        "What is current?",
    )
    assert response.token_ids == (10, 106)
    assert response.response_text == "blue"
    assert response.raw_content_text == "blue"
    assert response.stop_reason == "terminal_stop_token"
    assert response.stop_token_id == 106
    artifact = audit._generation_artifact(response)
    assert artifact["generated_token_ids_sha256"] == audit._payload_sha256(
        [10, 106]
    )
    assert artifact["response_utf8_sha256"] == audit._text_sha256("blue")


def test_record_executes_exact_matrix_with_alias_and_repeat_check(monkeypatch):
    record = fixture_record()
    runtime = FakeRuntime()
    lock = fixture_runtime_lock([record])
    monkeypatch.setattr(audit.methods.admission, "_validate_record", lambda *args: None)
    monkeypatch.setattr(audit, "_memory_fingerprint", lambda memory: "f" * 64)
    monkeypatch.setattr(audit, "_inference_context", lambda: audit.nullcontext())
    row = audit._audit_record(
        runtime,
        record,
        {"record_id": record.record_id},
        admission_binding={
            "record_id": record.record_id,
            "joint_admitted": True,
        },
        lock=lock,
    )
    assert list(row["conditions"]) == list(audit.CONDITION_IDS)
    assert row["status"] == "completed"
    assert runtime.generation_calls == (
        len(audit.EXECUTED_CONDITION_IDS)
        * len(audit.PROBE_IDS)
        * audit.GENERATION_REPEATS
    )
    alias = row["conditions"][audit.methods.TOKEN_ROW_DIAGNOSTIC_ID]
    assert alias["status"] == "aliased"
    assert alias["additional_generation_executions"] == 0
    assert all(probe["status"] == "aliased" for probe in alias["probes"])
    exact = row["conditions"][audit.methods.EXACT_DECREMENT_ID]
    assert exact["semantics"]["executed_method"] == "fixed_c_refit_fallback"
    for condition_id in audit.EXECUTED_CONDITION_IDS:
        for probe in row["conditions"][condition_id]["probes"]:
            assert probe["status"] == "completed"
            assert isinstance(probe["response_text"], str)
            assert probe["repeat_check"][
                "exact_token_ids_and_response_text_match"
            ] is True
            assert probe["content_audit"][
                "decoded_string_exhausts_extractability"
            ] is False


def test_nondeterminism_keeps_both_attempts_and_fails_probe(monkeypatch):
    record = fixture_record()
    runtime = FakeRuntime(nondeterministic=True)
    lock = fixture_runtime_lock([record])
    probes = audit.method_states.build_probes(record)
    monkeypatch.setattr(audit, "_inference_context", lambda: audit.nullcontext())
    result = audit._generate_probe(
        runtime,
        FakeMemory((1, 2), kind="present"),
        probes[0],
        condition_id=audit.methods.PRESENT_ID,
        record=record,
        lock=lock,
        prompt_is_suppressed=False,
    )
    assert result["status"] == "failed_nondeterministic"
    assert result["response_text"] == "blue"
    assert result["repeat_check"]["repeat_response_text"] == "amber"
    assert result["repeat_check"][
        "exact_token_ids_and_response_text_match"
    ] is False


def test_second_repeat_failure_preserves_first_generated_response(monkeypatch):
    record = fixture_record()
    runtime = FakeRuntime(fail_generation_call=2)
    lock = fixture_runtime_lock([record])
    probes = audit.method_states.build_probes(record)
    monkeypatch.setattr(audit, "_inference_context", lambda: audit.nullcontext())
    result = audit._generate_probe(
        runtime,
        FakeMemory((1, 2), kind="present"),
        probes[0],
        condition_id=audit.methods.PRESENT_ID,
        record=record,
        lock=lock,
        prompt_is_suppressed=False,
    )
    assert result["status"] == "failed"
    assert result["generation_attempts_started"] == 2
    assert result["generation_attempts_completed"] == 1
    assert result["completed_generation_attempts"][0]["response_text"] == "blue"
    audit._validate_probe_result(result, expected_probe_id="target_current")


def test_condition_failure_is_preserved_without_filtering(monkeypatch):
    record = fixture_record()
    runtime = FakeRuntime(fail_proxy=True)
    lock = fixture_runtime_lock([record])
    monkeypatch.setattr(audit.methods.admission, "_validate_record", lambda *args: None)
    monkeypatch.setattr(audit, "_memory_fingerprint", lambda memory: "e" * 64)
    monkeypatch.setattr(audit, "_inference_context", lambda: audit.nullcontext())
    row = audit._audit_record(
        runtime,
        record,
        {"record_id": record.record_id},
        admission_binding={
            "record_id": record.record_id,
            "joint_admitted": False,
        },
        lock=lock,
    )
    assert row["status"] == "completed_with_failures"
    proxy = row["conditions"][audit.methods.FP32_PROXY_ID]
    assert proxy["status"] == "failed"
    assert [probe["probe_id"] for probe in proxy["probes"]] == list(
        audit.PROBE_IDS
    )
    assert all(probe["status"] == "failed" for probe in proxy["probes"])
    assert set(row["conditions"]) == set(audit.CONDITION_IDS)


def _fake_all16(monkeypatch):
    records = tuple(fixture_record(f"record-{index}") for index in range(16))
    lock = fixture_runtime_lock(records)
    manifest = {
        "records": [{"record_id": record.record_id} for record in records],
    }
    method_lock = {
        "authorization": {
            "admission_records": [
                {
                    "record_id": record.record_id,
                    "joint_admitted": index < 10,
                }
                for index, record in enumerate(records)
            ]
        }
    }
    monkeypatch.setattr(audit.methods.admission, "_validate_record", lambda *args: None)
    monkeypatch.setattr(audit, "_memory_fingerprint", lambda memory: "d" * 64)
    monkeypatch.setattr(audit, "_inference_context", lambda: audit.nullcontext())
    return records, lock, manifest, method_lock


def test_all16_report_is_source_bearing_complete_and_validated(monkeypatch):
    records, lock, manifest, method_lock = _fake_all16(monkeypatch)
    report = audit.run_rehydrated_audit(
        FakeRuntime(),
        records,
        manifest,
        method_lock,
        lock,
    )
    assert report["status"] == "completed"
    assert report["contains_source_text"] is True
    assert report["release_authorized"] is False
    assert len(report["records"]) == 16
    assert report["summary"]["attempted_records"] == 16
    assert report["summary"]["not_attempted_records"] == 0
    assert report["summary"]["no_outcome_based_filtering"] is True
    assert report["source_final_report"][
        "generated_text_inferred_from_teacher_forced_scores"
    ] is False
    audit.validate_audit_report(report, lock=lock)


def test_fatal_report_keeps_all_record_condition_probe_slots(monkeypatch):
    records, lock, _manifest, _method_lock = _fake_all16(monkeypatch)
    report = audit._fatal_report(
        lock,
        RuntimeError("synthetic model-load failure"),
        elapsed_seconds=1.0,
    )
    assert report["status"] == "failed"
    assert [row["record_id"] for row in report["records"]] == [
        record.record_id for record in records
    ]
    assert report["summary"]["not_attempted_records"] == 16
    for row in report["records"]:
        assert list(row["conditions"]) == list(audit.CONDITION_IDS)
        for condition in row["conditions"].values():
            assert [probe["probe_id"] for probe in condition["probes"]] == list(
                audit.PROBE_IDS
            )
    audit.validate_audit_report(report, lock=lock)


def test_projection_can_only_use_case_preselected_in_lock(monkeypatch):
    records, lock, manifest, method_lock = _fake_all16(monkeypatch)
    lock["presentation_case"] = {
        "record_id": audit.PRESENTATION_CASE_RECORD_ID,
        "authorized_position": audit.PRESENTATION_CASE_POSITION,
        "selected_in_this_pre_run_lock": True,
        "selected_before_generated_outcomes": True,
    }
    records = list(records)
    records[1] = fixture_record(audit.PRESENTATION_CASE_RECORD_ID)
    lock["cohort"]["ordered_record_ids"][1] = audit.PRESENTATION_CASE_RECORD_ID
    lock["prompt_contract"]["base_probe_bytes"][1] = fixture_prompt_binding(
        records[1]
    )
    manifest["records"][1]["record_id"] = audit.PRESENTATION_CASE_RECORD_ID
    method_lock["authorization"]["admission_records"][1][
        "record_id"
    ] = audit.PRESENTATION_CASE_RECORD_ID
    report = audit.run_rehydrated_audit(
        FakeRuntime(),
        records,
        manifest,
        method_lock,
        lock,
    )
    projection = audit.build_presentation_projection(report, lock=lock)
    assert projection["case"]["record_id"] == audit.PRESENTATION_CASE_RECORD_ID
    assert projection["release_authorized"] is False
    assert projection["part_of_original_final_report"] is False
    assert projection["part_of_original_certificate"] is False
    assert projection["preselection_proof"][
        "selected_before_generated_outcomes"
    ] is True

    changed = copy.deepcopy(lock)
    changed["presentation_case"]["selected_before_generated_outcomes"] = False
    with pytest.raises(audit.AuditError, match="pre-run selection proof"):
        audit.build_presentation_projection(report, lock=changed)


def test_runtime_environment_drift_fails_closed(protocol, monkeypatch):
    changed = copy.deepcopy(audit.PINNED_RUNTIME_ENVIRONMENT)
    changed["packages"]["torch"] = "different"
    monkeypatch.setattr(audit, "_observed_runtime_environment", lambda: changed)
    with pytest.raises(RuntimeError, match="differs from the lock"):
        audit.verify_runtime_environment(protocol.lock)


def test_atomic_first_writer_and_alias_protection(tmp_path):
    output = tmp_path / "report.json"
    audit._atomic_write_new(output, {"status": "first"})
    before = output.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite is forbidden"):
        audit._atomic_write_new(output, {"status": "second"})
    assert output.read_bytes() == before
    assert list(tmp_path.glob("*.tmp-*")) == []

    protected = tmp_path / "protected.json"
    protected.write_text("{}", encoding="utf-8")
    symlink = tmp_path / "protected-link.json"
    symlink.symlink_to(protected)
    hardlink = tmp_path / "protected-hardlink.json"
    os.link(protected, hardlink)
    for candidate in (protected, symlink, hardlink):
        with pytest.raises(audit.AuditError, match="aliases"):
            audit._validate_no_alias(candidate, inputs=(protected,))


def test_persistent_atomic_attempt_claim_prevents_second_run(tmp_path, monkeypatch):
    claim = tmp_path / "audit.claim"
    monkeypatch.setattr(audit, "DEFAULT_ATTEMPT_CLAIM", claim)
    lock = {
        "integrity": {"sha256": "a" * 64},
        "cohort": {"ordered_record_ids_sha256": "b" * 64},
        "output_policy": {"canonical_workspace_path": "outputs/audit.json"},
    }
    assert audit._claim_attempt(lock) == claim.resolve()
    marker = json.loads(
        (claim / audit.ATTEMPT_MARKER_FILENAME).read_text(encoding="utf-8")
    )
    assert marker["attempt_number"] == 1
    assert marker["resume_allowed"] is False
    with pytest.raises(PermissionError, match="already claimed"):
        audit._claim_attempt(lock)


def test_lock_freeze_path_never_loads_model(protocol, tmp_path, monkeypatch):
    destination = tmp_path / "authorization.json"
    monkeypatch.setattr(audit, "DEFAULT_AUTHORIZATION_LOCK", destination)
    monkeypatch.setattr(
        audit,
        "_load_protocol_inputs",
        lambda **kwargs: (
            protocol.final_report,
            protocol.manifest,
            protocol.method_lock,
        ),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(audit.methods.admission, "_make_runtime", forbidden)
    lock = audit.freeze_authorization_lock_file(lock_path=destination)
    assert destination.is_file()
    assert json.loads(destination.read_text(encoding="utf-8")) == lock
    with pytest.raises(FileExistsError, match="already exists"):
        audit.freeze_authorization_lock_file(lock_path=destination)


def test_disk_runner_uses_fake_runtime_and_atomically_writes_all16(
    tmp_path,
    monkeypatch,
):
    records, lock, manifest, method_lock = _fake_all16(monkeypatch)
    output = tmp_path / "audit.json"
    claim = tmp_path / "audit.claim"
    authorization = tmp_path / "authorization.json"
    monkeypatch.setattr(audit, "DEFAULT_OUTPUT", output)
    monkeypatch.setattr(audit, "DEFAULT_ATTEMPT_CLAIM", claim)
    monkeypatch.setattr(audit, "DEFAULT_AUTHORIZATION_LOCK", authorization)
    monkeypatch.setattr(
        audit,
        "_load_protocol_inputs",
        lambda **kwargs: ({}, manifest, method_lock),
    )
    monkeypatch.setattr(
        audit,
        "load_committed_authorization_lock",
        lambda *args, **kwargs: lock,
    )
    monkeypatch.setattr(audit, "verify_runtime_environment", lambda value: None)
    monkeypatch.setattr(
        audit,
        "_validate_no_alias",
        lambda value, **kwargs: Path(value).resolve(),
    )
    monkeypatch.setattr(audit, "_snapshot_files", lambda paths: {"stable": "x"})
    monkeypatch.setattr(audit, "_configure_determinism", lambda: None)
    monkeypatch.setattr(
        audit,
        "_offline_huggingface",
        lambda: audit.nullcontext(),
    )
    monkeypatch.setattr(audit, "_verify_cached_model_metadata", lambda value: None)
    monkeypatch.setattr(
        audit.geometry.base,
        "load_pinned_longmemeval_rows",
        lambda data_path: (),
    )
    runtime = FakeRuntime()
    monkeypatch.setattr(
        audit.methods.admission,
        "_make_runtime",
        lambda device: runtime,
    )
    monkeypatch.setattr(
        audit,
        "verify_loaded_generation_runtime",
        lambda runtime, **kwargs: None,
    )
    monkeypatch.setattr(
        audit.geometry,
        "rehydrate_manifest",
        lambda *args, **kwargs: tuple(
            SimpleNamespace(runtime=record) for record in records
        ),
    )
    report = audit.run_audit_from_paths(
        authorization_lock_path=authorization,
        output_path=output,
        explicit_acknowledgement=True,
    )
    assert report["status"] == "completed"
    assert output.is_file()
    assert claim.is_dir()
    assert json.loads(output.read_text(encoding="utf-8")) == report
    with pytest.raises(FileExistsError, match="already exists"):
        audit.run_audit_from_paths(
            authorization_lock_path=authorization,
            output_path=output,
            explicit_acknowledgement=True,
        )


def test_cli_exposes_no_resume_or_overwrite_modes():
    parser = audit._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--run", "--resume"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--run", "--overwrite"])


def test_credential_fields_are_rejected(protocol):
    changed = copy.deepcopy(protocol.lock)
    changed["hf_token"] = "must-not-serialize"
    with pytest.raises(audit.AuditError, match="credential-bearing"):
        audit._assert_source_free_lock(changed)
