from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest

from gemma_sv import longmemeval_chat_response_generation_audit_v2 as audit


@pytest.fixture(autouse=True)
def _isolate_mock_runtime_from_accelerator_rng(monkeypatch):
    """These fake-runtime contract tests do not exercise device RNG seeding."""
    monkeypatch.setattr(audit.decoded_v1, "_seed_response", lambda seed: None)


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return SimpleNamespace(input_ids=[7, 8])

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens,
        clean_up_tokenization_spaces,
    ):
        assert clean_up_tokenization_spaces is False
        pieces = {10: "blue", 11: "green", 106: "<end_of_turn>"}
        rendered = "".join(pieces.get(int(item), "?") for item in token_ids)
        if skip_special_tokens:
            rendered = rendered.replace("<end_of_turn>", "")
        return rendered


class FakeMemory:
    def __init__(self, token_ids, *, kind):
        self.token_ids = tuple(token_ids)
        self.token_count = len(self.token_ids)
        self.input_digest = audit.cohort_v3.base.token_ids_sha256(
            self.token_ids
        )
        self.kind = kind
        self.request = SimpleNamespace(gate_floor=0.0)

    def fork(self):
        return copy.deepcopy(self)


class FakeRuntime:
    def __init__(self, records):
        self.tokenizer = FakeTokenizer()
        self.generation_calls = 0
        self.prefill_calls = 0
        self.boundaries = []
        self.progress_callback = None
        self.originals = {
            tuple(record.context.original_token_ids) for record in records
        }

    def prefill_persistent(self, token_ids):
        self.prefill_calls += 1
        kind = "present" if tuple(token_ids) in self.originals else "raw"
        return FakeMemory(token_ids, kind=kind)

    def persistent_certificate_states(self, memory, forget_positions):
        del forget_positions
        exact = memory.fork()
        exact.kind = "exact"
        if self.progress_callback is not None:
            self.progress_callback(1, 1)
        return (
            {
                "fixed_c_feasible": True,
                "used_refit_fallback": True,
                "n_fallback": 1,
                "n_solves": 2,
                "n_head_gates": 2,
                "box_C": None,
                "box_C_by_boundary": {"8": 0.2},
                "objective": {
                    "box_C": None,
                    "box_C_by_boundary": {"8": 0.2},
                    "box_source": "frozen_decode_session",
                    "bandwidth_source": "frozen_decode_session",
                    "kpar_by_layer": {0: 1.5},
                },
                "feasibility": [],
                "fallback_details": [
                    {
                        "layer_id": 0,
                        "start": 8,
                        "head": 0,
                        "reason": "synthetic",
                    }
                ],
                "max_functional_deviation": 0.0,
                "max_candidate_deviation": 0.0,
                "functional_tolerance": 1e-5,
                "partition_diagnostics": {
                    "affected_support_fraction": None,
                    "affected_margin_fraction": None,
                    "refit_support_fraction": None,
                    "fallback_support_fraction": None,
                    "successful_support_fraction": None,
                },
                "materialization": {
                    "persistent_states": ["exact_policy"],
                    "refit_coefficients_used_for_solver_conformance": True,
                    "refit_persistent_state_constructed": False,
                    "decay_persistent_state_constructed": False,
                },
            },
            {"exact": exact},
        )

    @contextmanager
    def _persistent_branch(self, memory):
        branch = memory.fork()
        branch.step = 0
        yield branch

    def _persistent_prompt(self, branch, prompt):
        self.generation_calls += 1
        branch.prompt = prompt
        token = 10 if "current" in prompt.casefold() else 11
        return _logits(token)

    def _persistent_step(self, branch, token_id):
        del branch, token_id
        return _logits(106)

    def set_certificate_progress_callback(self, callback):
        self.progress_callback = callback

    @contextmanager
    def record_boundary(self, slot, record_id):
        self.boundaries.append(("start", slot, record_id))
        try:
            yield
        finally:
            self.boundaries.append(("end", slot, record_id))
            self.progress_callback = None


def _logits(token_id):
    values = np.full(128, -100.0)
    values[int(token_id)] = 1.0
    return values


def _record(slot):
    original = (1000 + slot, 2, 3, 4)
    raw = (1000 + slot, 3, 4)
    target_prompt = audit.cohort_v3.chat_v1.render_constrained_query(
        "What is current?",
        "2026-08-25",
    )
    retained_prompt = audit.cohort_v3.chat_v1.render_constrained_query(
        "What remains?",
        "2026-08-25",
    )
    probes = (
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
    )
    return SimpleNamespace(
        record_id=f"history-{slot:03d}",
        context=SimpleNamespace(
            original_token_ids=original,
            edited_token_ids=raw,
            forget_positions=(1,),
        ),
        raw_omitted_token_ids=raw,
        probes=probes,
    )


def _fixtures():
    records = [_record(slot) for slot in range(audit.EXPECTED_HISTORIES)]
    histories = []
    bindings = []
    for slot, record in enumerate(records):
        probe_rows = [
            {
                "probe_id": probe.probe_id,
                "kind": probe.kind,
                "prompt_sha256": audit._text_sha256(probe.prompt_text),
                "question_sha256": audit._text_sha256(probe.question),
                "answer_sha256": audit._text_sha256(probe.answer),
                "target_token_count": len(probe.target_token_ids),
                "target_token_ids_sha256": (
                    audit.cohort_v3.base.token_ids_sha256(
                        probe.target_token_ids
                    )
                ),
            }
            for probe in record.probes
        ]
        integrity = f"{slot:064x}"
        histories.append(
            {
                "record_id": record.record_id,
                "cluster_id": f"cluster-{slot // 3:02d}",
                "variant_index": slot % 3,
                "record_integrity": {"sha256": integrity},
                "probes": probe_rows,
            }
        )
        bindings.append(
            {
                "history_id": record.record_id,
                "cluster_id": f"cluster-{slot // 3:02d}",
                "variant_index": slot % 3,
                "history_integrity_sha256": integrity,
                "probes": copy.deepcopy(probe_rows),
            }
        )
    ids = [record.record_id for record in records]
    lock = {
        "cohort": {
            "ordered_history_ids": ids,
            "ordered_history_ids_sha256": audit._payload_sha256(ids),
        },
        "prompt_answer_bindings": {"histories": bindings},
        "integrity": {"sha256": "a" * 64},
    }
    cohort = {"histories": histories}
    return records, cohort, lock


def _terminal_shard(slot, lock, *, status="completed"):
    record_id = lock["cohort"]["ordered_history_ids"][slot]
    record = _record(slot)
    binding = audit._binding(lock, record_id)
    public = {
        "record_id": record_id,
        "cluster_id": binding["cluster_id"],
        "variant_index": binding["variant_index"],
        "record_integrity": {
            "sha256": binding["history_integrity_sha256"]
        },
        "probes": binding["probes"],
    }
    if status == "completed":
        result = audit.audit_history(
            FakeRuntime([record]),
            record,
            public,
            lock=lock,
        )
    elif status == "failed":
        result = audit._terminal_failure(
            record,
            public,
            RuntimeError("synthetic terminal failure"),
        )
    else:
        raise AssertionError(status)
    return audit._seal(
        {
            "schema": audit.SHARD_SCHEMA,
            "schema_version": audit.SCHEMA_VERSION,
            "slot": slot,
            "record_id": record_id,
            "terminal": True,
            "authorization_lock_integrity_sha256": lock["integrity"]["sha256"],
            "result": result,
        }
    )


def _initialized_root(tmp_path, lock):
    root = tmp_path / "run"
    root.parent.mkdir(parents=True, exist_ok=True)
    audit._initialize_root(
        root,
        lock,
        certificate_workers=8,
        resume=False,
    )
    return root


def _publish_terminal(root, lock, slot, *, status="completed"):
    shard = _terminal_shard(slot, lock, status=status)
    record_id = lock["cohort"]["ordered_history_ids"][slot]
    audit._attempt_started(
        root,
        slot=slot,
        attempt=1,
        record_id=record_id,
        lock=lock,
    )
    audit._atomic_write_new(
        root / "records" / f"{slot:03d}.json",
        shard,
    )
    audit._attempt_terminal(
        root,
        slot=slot,
        attempt=1,
        record_id=record_id,
        shard=shard,
        lock=lock,
    )
    return shard


def test_frozen_lock_binds_exact_96_by_1536_without_model_loading(
    monkeypatch,
):
    cohort, cluster_lock, census = audit.load_frozen_inputs()

    def forbidden(*args, **kwargs):
        raise AssertionError("model loading is forbidden during lock freeze")

    monkeypatch.setattr(audit.runtime_v2, "_make_runtime", forbidden)
    monkeypatch.setattr(
        audit.decoded_v1,
        "_verify_cached_model_metadata",
        forbidden,
    )
    lock = audit.freeze_authorization_lock(
        cohort,
        cluster_lock,
        census,
    )

    assert len(lock["cohort"]["ordered_history_ids"]) == 96
    assert len(lock["cohort"]["ordered_cluster_ids"]) == 32
    workload = lock["generation"]["planned_workload"]
    assert workload == {
        "histories": 96,
        "conditions": 4,
        "probes": 2,
        "repeats": 2,
        "generation_calls": 1536,
        "maximum_generated_token_steps": 98304,
    }
    assert lock["matrix"]["condition_ids"] == list(audit.CONDITION_IDS)
    assert lock["certificate_workers"]["maximum"] == 16
    environment = lock["runtime"]["environment"]
    assert set(environment["packages"]) == set(
        audit._RUNTIME_DISTRIBUTIONS
    )
    assert environment["platform"]["python_version"]
    assert environment["torch"]["mps_built"] is True
    assert environment["controls"]["certificate_child_blas_threads"] == 1
    assert lock["matrix"]["persistent_state_materialization"] == {
        "states_constructed": ["exact_policy"],
        "refit_coefficients_used_inside_solver": True,
        "refit_persistent_state_constructed": False,
        "decay_persistent_state_constructed": False,
    }
    audit._assert_source_free_lock(lock)


def test_frozen_authorization_artifact_reproduces_exactly(monkeypatch):
    # Replay the recorded environment input; this machine may have a newer OS
    # or lack a visible accelerator. Production environment checks are unchanged.
    recorded = audit._load_json(audit.DEFAULT_AUTHORIZATION_LOCK, name="lock")
    monkeypatch.setattr(
        audit, "_observed_runtime_environment",
        lambda: copy.deepcopy(recorded["runtime"]["environment"]),
    )
    cohort, cluster_lock, census = audit.load_frozen_inputs()
    expected = audit.freeze_authorization_lock(
        cohort,
        cluster_lock,
        census,
    )
    committed_value = audit._load_json(
        audit.DEFAULT_AUTHORIZATION_LOCK,
        name="authorization lock",
    )
    assert committed_value == expected
    audit.validate_authorization_lock(
        committed_value,
        cohort=cohort,
        cluster_lock=cluster_lock,
        census=census,
    )


@pytest.mark.parametrize("acknowledgement", ["", True, "yes", None])
def test_live_run_requires_exact_acknowledgement_before_loading(
    monkeypatch,
    acknowledgement,
):
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("lock or model access occurred")

    monkeypatch.setattr(audit, "load_authorization_lock", forbidden)
    monkeypatch.setattr(audit, "_make_runtime", forbidden)
    with pytest.raises(PermissionError, match="exact explicit"):
        audit.run_from_paths(
            output_root=audit.DEFAULT_OUTPUT_ROOT,
            data_path=None,
            certificate_workers=8,
            resume=False,
            acknowledgement=acknowledgement,
        )
    assert called is False


def test_uncommitted_lock_refusal_precedes_model_loading(monkeypatch):
    called = False

    def reject(*, require_committed):
        assert require_committed is True
        raise PermissionError("authorization lock must be committed at HEAD")

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("model loading occurred")

    monkeypatch.setattr(audit, "_safe_cli_output_root", lambda path: Path(path))
    monkeypatch.setattr(audit, "load_authorization_lock", reject)
    monkeypatch.setattr(audit, "_make_runtime", forbidden)
    with pytest.raises(PermissionError, match="committed at HEAD"):
        audit.run_from_paths(
            output_root=audit.DEFAULT_OUTPUT_ROOT,
            data_path=None,
            certificate_workers=8,
            resume=False,
            acknowledgement=audit.EXPLICIT_ACKNOWLEDGEMENT,
        )
    assert called is False


def test_history_serializes_real_text_tokens_checks_and_four_by_two_by_two():
    records, cohort, lock = _fixtures()
    runtime = FakeRuntime(records)
    row = audit.audit_history(
        runtime,
        records[0],
        cohort["histories"][0],
        lock=lock,
    )

    assert list(row["conditions"]) == list(audit.CONDITION_IDS)
    assert runtime.generation_calls == 16
    assert row["generation_calls_started"] == 16
    assert row["status"] == "completed"
    for condition in row["conditions"].values():
        assert len(condition["probes"]) == 2
        for probe in condition["probes"]:
            assert len(probe["generation_attempts"]) == 2
            assert probe["repeat_check"][
                "exact_token_ids_and_response_text_match"
            ] is True
            first = probe["generation_attempts"][0]
            assert isinstance(first["response_text"], str)
            assert first["generated_token_ids"][-1] == 106
            assert first["response_utf8_sha256"] == audit._text_sha256(
                first["response_text"]
            )
            assert probe["response_checks"][
                "decoded_strings_exhaust_extractability"
            ] is False
    policy = row["conditions"]["exact_decrement_or_refit_policy"]
    assert policy["semantics"]["used_any_refit_fallback"] is True
    assert policy["semantics"]["separate_refit_response_decoded"] is False
    assert "exact" not in policy["fixed_c_diagnostics"]


def test_full_sharded_run_resume_permissions_heartbeat_and_canonical_final(
    tmp_path,
):
    records, cohort, lock = _fixtures()
    runtime = FakeRuntime(records)
    root = tmp_path / "audit"

    final = audit.run_sharded_audit(
        runtime,
        records,
        cohort,
        lock,
        output_root=root,
        certificate_workers=8,
    )

    assert final is not None
    assert runtime.generation_calls == audit.EXPECTED_GENERATION_CALLS
    assert [row["record_id"] for row in final["records"]] == (
        lock["cohort"]["ordered_history_ids"]
    )
    assert final["summary"]["generation_calls_started"] == 1536
    assert oct(os.stat(root).st_mode & 0o777) == "0o700"
    evidence = [
        root / "run.json",
        root / "records" / "000.json",
        root / "attempts" / "000" / "001-started.json",
        root / "attempts" / "000" / "001-terminal.json",
        root / "final.json",
    ]
    assert all(oct(os.stat(path).st_mode & 0o777) == "0o600" for path in evidence)
    heartbeat = json.loads((root / "heartbeat.json").read_text())
    keys = set(audit._walk_keys(heartbeat))
    assert not any(
        fragment in key
        for key in keys
        for fragment in audit._HEARTBEAT_FORBIDDEN_FRAGMENTS
    )

    resumed = FakeRuntime(records)
    existing = audit.run_sharded_audit(
        resumed,
        records,
        cohort,
        lock,
        output_root=root,
        certificate_workers=8,
        resume=True,
        assemble=False,
    )
    assert existing == final
    assert resumed.generation_calls == 0
    assert resumed.prefill_calls == 0


def test_unsealed_started_attempt_is_retained_then_retried(tmp_path):
    records, cohort, lock = _fixtures()
    root = _initialized_root(tmp_path, lock)
    audit._attempt_started(
        root,
        slot=0,
        attempt=1,
        record_id=records[0].record_id,
        lock=lock,
    )
    runtime = FakeRuntime(records)

    audit.run_sharded_audit(
        runtime,
        records,
        cohort,
        lock,
        output_root=root,
        certificate_workers=8,
        resume=True,
    )

    ledger = root / "attempts" / "000"
    assert (ledger / "001-started.json").exists()
    assert not (ledger / "001-terminal.json").exists()
    assert (ledger / "002-started.json").exists()
    assert (ledger / "002-terminal.json").exists()


def test_terminal_failure_shard_is_reused_without_reexecution(tmp_path):
    records, cohort, lock = _fixtures()
    root = _initialized_root(tmp_path, lock)
    failed = _publish_terminal(root, lock, 0, status="failed")
    runtime = FakeRuntime(records)

    audit.run_sharded_audit(
        runtime,
        records,
        cohort,
        lock,
        output_root=root,
        certificate_workers=8,
        resume=True,
    )

    assert runtime.generation_calls == (95 * 4 * 2 * 2)
    preserved = audit._load_json(
        root / "records" / "000.json",
        name="preserved failure",
    )
    assert preserved == failed


def test_corrupt_extra_duplicate_and_wrong_lock_shards_are_rejected(
    tmp_path,
):
    _records, _cohort, lock = _fixtures()

    corrupt_root = _initialized_root(tmp_path / "corrupt", lock)
    corrupt = _terminal_shard(0, lock)
    corrupt["result"]["status"] = "tampered"
    audit._atomic_write_new(corrupt_root / "records" / "000.json", corrupt)
    with pytest.raises(audit.AuditV2Error, match="integrity"):
        audit._scan_shards(corrupt_root, lock)

    extra_root = _initialized_root(tmp_path / "extra", lock)
    audit._atomic_write_new(extra_root / "records" / "extra.json", {})
    with pytest.raises(audit.AuditV2Error, match="extra shard"):
        audit._scan_shards(extra_root, lock)

    duplicate_root = _initialized_root(tmp_path / "duplicate", lock)
    duplicate = _terminal_shard(1, lock)
    duplicate["record_id"] = lock["cohort"]["ordered_history_ids"][0]
    duplicate["result"]["record_id"] = duplicate["record_id"]
    duplicate = audit._seal(duplicate)
    audit._atomic_write_new(
        duplicate_root / "records" / "001.json",
        duplicate,
    )
    with pytest.raises(audit.AuditV2Error, match="binding differs"):
        audit._scan_shards(duplicate_root, lock)

    wrong_root = _initialized_root(tmp_path / "wrong", lock)
    wrong = _terminal_shard(0, lock)
    wrong["authorization_lock_integrity_sha256"] = "b" * 64
    wrong = audit._seal(wrong)
    audit._atomic_write_new(wrong_root / "records" / "000.json", wrong)
    with pytest.raises(audit.AuditV2Error, match="binding differs"):
        audit._scan_shards(wrong_root, lock)


def test_no_overwrite_and_reverse_publication_assembles_canonically(tmp_path):
    _records, _cohort, lock = _fixtures()
    root = _initialized_root(tmp_path, lock)
    shard = _publish_terminal(root, lock, 0)
    existing = root / "records" / "000.json"
    with pytest.raises(FileExistsError, match="overwrite is forbidden"):
        audit._atomic_write_new(existing, shard)

    for slot in reversed(range(1, audit.EXPECTED_HISTORIES)):
        _publish_terminal(root, lock, slot)
    final = audit.assemble_final(root, lock)
    assert [row["record_id"] for row in final["records"]] == (
        lock["cohort"]["ordered_history_ids"]
    )
    assert audit.assemble_final(root, lock) == final


def test_resealed_semantic_shard_corruption_is_rejected():
    _records, _cohort, lock = _fixtures()
    base = _terminal_shard(0, lock)

    mutations = []

    def extra_condition(shard):
        shard["result"]["conditions"]["unsupported"] = copy.deepcopy(
            shard["result"]["conditions"]["present"]
        )

    mutations.append(extra_condition)
    mutations.append(
        lambda shard: shard["result"]["conditions"]["present"]["probes"][
            0
        ]["generation_attempts"].pop()
    )
    mutations.append(
        lambda shard: shard["result"]["conditions"]["present"]["probes"][
            0
        ]["generation_attempts"][0].__setitem__(
            "response_utf8_sha256",
            "0" * 64,
        )
    )
    mutations.append(
        lambda shard: shard["result"].__setitem__(
            "generation_calls_started",
            15,
        )
    )
    mutations.append(
        lambda shard: shard["result"].__setitem__(
            "cluster_id",
            "wrong-cluster",
        )
    )
    mutations.append(
        lambda shard: shard["result"].__setitem__("unsupported", True)
    )

    for mutate in mutations:
        changed = copy.deepcopy(base)
        mutate(changed)
        changed = audit._seal(changed)
        with pytest.raises(audit.AuditV2Error):
            audit._validate_shard(
                changed,
                slot=0,
                record_id=lock["cohort"]["ordered_history_ids"][0],
                lock=lock,
            )


def test_existing_final_validation_rejects_tamper_lock_order_and_content(
    tmp_path,
):
    records, cohort, lock = _fixtures()
    root = tmp_path / "audit"
    final = audit.run_sharded_audit(
        FakeRuntime(records),
        records,
        cohort,
        lock,
        output_root=root,
        certificate_workers=8,
    )
    assert final is not None
    audit.validate_final(final, lock=lock)

    tampered = copy.deepcopy(final)
    tampered["summary"]["generation_calls_started"] = 0
    with pytest.raises(audit.AuditV2Error, match="integrity"):
        audit.validate_final(tampered, lock=lock)

    wrong_lock = copy.deepcopy(final)
    wrong_lock["authorization_lock_integrity_sha256"] = "b" * 64
    wrong_lock = audit._seal(wrong_lock)
    with pytest.raises(audit.AuditV2Error, match="binding"):
        audit.validate_final(wrong_lock, lock=lock)

    wrong_order = copy.deepcopy(final)
    wrong_order["records"][0], wrong_order["records"][1] = (
        wrong_order["records"][1],
        wrong_order["records"][0],
    )
    wrong_order = audit._seal(wrong_order)
    with pytest.raises(audit.AuditV2Error, match="order"):
        audit.validate_final(wrong_order, lock=lock)

    wrong_content = copy.deepcopy(final)
    attempt = wrong_content["records"][0]["conditions"]["present"][
        "probes"
    ][0]["generation_attempts"][0]
    attempt["response_text"] = "tampered"
    wrong_content = audit._seal(wrong_content)
    with pytest.raises(audit.AuditV2Error, match="payload"):
        audit.validate_final(wrong_content, lock=lock)


def test_heartbeat_ticker_writes_during_long_operation_without_content(
    tmp_path,
):
    root = tmp_path / "heartbeat"
    root.mkdir(mode=0o700)
    emissions = []

    def emit():
        emissions.append(time.monotonic())
        audit._heartbeat(
            root,
            phase="record_heartbeat",
            slot=0,
            record_id="history-000",
            started=emissions[0],
            completed=0,
            terminal=0,
            certificate_progress=None,
            last_durable_shard=None,
        )

    with audit._heartbeat_ticker(emit, interval_seconds=0.01):
        time.sleep(0.04)
        emit()

    assert len(emissions) >= 3
    heartbeat = json.loads((root / "heartbeat.json").read_text())
    assert heartbeat["phase"] == "record_heartbeat"
    assert not any(
        fragment in key
        for key in audit._walk_keys(heartbeat)
        for fragment in audit._HEARTBEAT_FORBIDDEN_FRAGMENTS
    )


def test_resume_scan_rejects_symlinked_children_and_evidence(tmp_path):
    _records, _cohort, lock = _fixtures()

    child_root = _initialized_root(tmp_path / "child", lock)
    child_target = tmp_path / "child-target"
    child_target.mkdir()
    (child_root / "records").rmdir()
    (child_root / "records").symlink_to(child_target, target_is_directory=True)
    with pytest.raises(audit.AuditV2Error, match="symlink"):
        audit._scan_shards(child_root, lock)

    file_root = _initialized_root(tmp_path / "file", lock)
    external = tmp_path / "external.json"
    external.write_text("{}")
    (file_root / "records" / "000.json").symlink_to(external)
    with pytest.raises(audit.AuditV2Error, match="extra shard"):
        audit._scan_shards(file_root, lock)

    run_root = _initialized_root(tmp_path / "run-link", lock)
    run_copy = tmp_path / "run-copy.json"
    run_copy.write_bytes((run_root / "run.json").read_bytes())
    (run_root / "run.json").unlink()
    (run_root / "run.json").symlink_to(run_copy)
    with pytest.raises(audit.AuditV2Error, match="symlink"):
        audit._initialize_root(
            run_root,
            lock,
            certificate_workers=8,
            resume=True,
        )
