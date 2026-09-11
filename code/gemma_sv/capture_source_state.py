"""Capture the exact tracked diff and untracked-source hashes for a run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from gemma_sv.recovery_protocol import write_json_atomic


def _run(repo: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(repo: Path, out_dir: Path) -> dict:
    repo = repo.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    head = _run(repo, "rev-parse", "HEAD").decode().strip()
    diff = _run(repo, "diff", "--binary", "--no-ext-diff", "HEAD")
    status = _run(repo, "status", "--porcelain=v1", "--untracked-files=all")
    untracked_raw = _run(
        repo,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    untracked = []
    for raw in untracked_raw.split(b"\0"):
        if not raw:
            continue
        relative = raw.decode()
        path = repo / relative
        if path.is_file():
            untracked.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    untracked.sort(key=lambda row: row["path"])
    diff_path = out_dir / "source_tracked.diff"
    diff_path.write_bytes(diff)
    report = {
        "schema": "gemma-sv-source-state-v1",
        "git_head": head,
        "tracked_diff": {
            "path": diff_path.name,
            "bytes": len(diff),
            "sha256": _sha256_bytes(diff),
        },
        "status_sha256": _sha256_bytes(status),
        "untracked_files": untracked,
    }
    write_json_atomic(out_dir / "source_state.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    report = capture(args.repo, args.out_dir)
    print(json.dumps(report["tracked_diff"], sort_keys=True))
    print(f"saved -> {args.out_dir / 'source_state.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
