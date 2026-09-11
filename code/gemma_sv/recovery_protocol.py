"""Reproducibility helpers for matched Gemma recovery/control experiments."""
from __future__ import annotations

import hashlib
import json
import os
import random
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch


MODEL_REVISION = "fcf18a2a879aab110ca39f8bffbccd5d49d8eb29"
FINEWEB_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def seed_everything(seed: int) -> dict[str, Any]:
    """Seed every RNG used by the recovery path.

    MPS does not promise bitwise reproducibility for every kernel, so reports
    record the deterministic-algorithm status rather than claiming it.
    """

    resolved = int(seed)
    os.environ["PYTHONHASHSEED"] = str(resolved)
    random.seed(resolved)
    np.random.seed(resolved)
    torch.manual_seed(resolved)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved)
    mps = getattr(torch, "mps", None)
    if mps is not None and hasattr(mps, "manual_seed"):
        mps.manual_seed(resolved)
    return {
        "seed": resolved,
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": bool(
            getattr(getattr(torch.backends, "cudnn", None), "deterministic", False)
        ),
    }


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor values, dtype, and shape independently of its device."""

    value = tensor.detach().to("cpu").contiguous()
    digest = hashlib.sha256()
    dtype = str(value.dtype).encode("utf-8")
    digest.update(struct.pack(">I", len(dtype)))
    digest.update(dtype)
    digest.update(struct.pack(">I", value.ndim))
    for size in value.shape:
        digest.update(struct.pack(">Q", int(size)))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


class BatchFingerprint:
    """Streaming SHA-256 over an ordered sequence of training batches."""

    def __init__(self) -> None:
        self._digest = hashlib.sha256(b"gemma-sv-stage2-stream-v1\0")
        self.count = 0
        self.first_batch_sha256: str | None = None
        self.last_batch_sha256: str | None = None

    def update(self, batch: torch.Tensor) -> None:
        batch_digest = tensor_sha256(batch)
        if self.first_batch_sha256 is None:
            self.first_batch_sha256 = batch_digest
        self.last_batch_sha256 = batch_digest
        self._digest.update(struct.pack(">Q", self.count))
        self._digest.update(bytes.fromhex(batch_digest))
        self.count += 1

    def summary(self) -> dict[str, Any]:
        return {
            "algorithm": "sha256",
            "schema": "gemma-sv-stage2-stream-v1",
            "batches": self.count,
            "sha256": self._digest.hexdigest(),
            "first_batch_sha256": self.first_batch_sha256,
            "last_batch_sha256": self.last_batch_sha256,
        }


def path_sha256(path: str | Path) -> str:
    """Hash a file or directory with stable relative-path framing."""

    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(root)
    files = [root] if root.is_file() else sorted(item for item in root.rglob("*") if item.is_file())
    digest = hashlib.sha256(b"gemma-sv-path-v1\0")
    for item in files:
        relative = item.name if root.is_file() else item.relative_to(root).as_posix()
        encoded = relative.encode("utf-8")
        digest.update(struct.pack(">I", len(encoded)))
        digest.update(encoded)
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON without exposing a partially written result artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)


def validate_stage2_pair(recovered: dict[str, Any], control: dict[str, Any]) -> None:
    """Reject a control that did not train on the recovered arm's exact stream."""

    recovery_stream = recovered.get("stage2_stream") or {}
    control_stream = control.get("stage2_stream") or {}
    failures = []
    for key in ("batches", "sha256", "first_batch_sha256", "last_batch_sha256"):
        if recovery_stream.get(key) != control_stream.get(key):
            failures.append(
                f"stage2_stream.{key}: "
                f"{recovery_stream.get(key)!r} != {control_stream.get(key)!r}"
            )
    recovery_config = recovered.get("config") or {}
    control_config = control.get("config") or {}
    for key in (
        "model",
        "model_revision",
        "seq_len",
        "batch",
        "stage2_steps",
        "lr2",
        "rank",
        "preserve_prefix_mass",
        "per_boundary_box",
        "data_seed",
        "init_seed",
    ):
        if recovery_config.get(key) != control_config.get(key):
            failures.append(
                f"config.{key}: "
                f"{recovery_config.get(key)!r} != {control_config.get(key)!r}"
            )
    if recovered.get("stage2_offset_batches") != control.get(
        "stage2_offset_batches"
    ):
        failures.append(
            "stage2_offset_batches: "
            f"{recovered.get('stage2_offset_batches')!r} != "
            f"{control.get('stage2_offset_batches')!r}"
        )
    recovery_eval = recovered.get("eval") or {}
    control_eval = control.get("eval") or {}
    for key in ("revision", "blocks", "seq_len", "token_ids_sha256"):
        if recovery_eval.get(key) != control_eval.get(key):
            failures.append(
                f"eval.{key}: "
                f"{recovery_eval.get(key)!r} != {control_eval.get(key)!r}"
            )
    if failures:
        raise ValueError("unmatched recovery/control pair: " + "; ".join(failures))
