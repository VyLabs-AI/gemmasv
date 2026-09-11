"""Publish a source-free summary of the predeclared second-checkpoint run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1] if ROOT.parent.name == "code" else ROOT.parent
DEFAULT_RUN_ROOT = PROJECT_ROOT / "outputs/gemma_sv_second_checkpoint_12b_pt_v1"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _admission_rows(report: dict) -> list[dict]:
    whole = report["whole_record"]
    rows = [*whole["records"], *whole["rejected_records"]]
    return sorted(rows, key=lambda row: int(row["manifest_index"]))


def _admission_summary(report: dict) -> dict:
    rows = _admission_rows(report)
    admitted = {
        int(row["manifest_index"]) for row in report["whole_record"]["records"]
    }
    paired_rows = []
    feasible_records = 0
    for row in rows:
        fields = row["admission"]["fields"]
        feasibility = [
            boundary["feasible"]
            for field in fields
            for boundary in field.get("fixed_c_feasibility", [])
        ]
        fixed_c_feasible = bool(feasibility) and all(feasibility)
        feasible_records += int(fixed_c_feasible)
        paired_rows.append(
            {
                "manifest_index": int(row["manifest_index"]),
                "admitted": int(row["manifest_index"]) in admitted,
                "rejection_reasons": list(row.get("reasons", [])),
                "fixed_c_feasible": fixed_c_feasible,
            }
        )
    whole = report["whole_record"]
    return {
        "attempted": int(whole["attempted"]),
        "admitted": int(whole["admitted"]),
        "fixed_c_feasible": feasible_records,
        "records": paired_rows,
    }


def summarize(
    predeclaration_path: Path,
    ungrafted_path: Path,
    graft_path: Path,
    quality_path: Path,
) -> dict:
    predeclaration = _load(predeclaration_path)
    ungrafted = _load(ungrafted_path)
    graft = _load(graft_path)
    quality = _load(quality_path)

    model = predeclaration["primary_model"]
    confirmation = predeclaration["confirmation"]
    configuration = predeclaration["configuration"]
    for label, report in (("ungrafted", ungrafted), ("graft", graft)):
        if report["model"] != model["id"]:
            raise ValueError(f"{label} model differs from predeclaration")
        if report["model_revision"] != model["revision"]:
            raise ValueError(f"{label} revision differs from predeclaration")
        manifest = report["provenance"]["manifest"]
        if manifest["sha256"] != confirmation["sha256"]:
            raise ValueError(f"{label} confirmation manifest hash mismatch")
        if int(report["whole_record"]["attempted"]) != int(
            confirmation["attempted_records"]
        ):
            raise ValueError(f"{label} attempted-record count mismatch")

    if ungrafted["ungrafted"] is not True or graft["ungrafted"] is not False:
        raise ValueError("admission arms are mislabeled")
    if graft["preserve_prefix_mass"] is not True:
        raise ValueError("graft arm did not preserve prefix mass")
    if graft["per_boundary_box"] is not True:
        raise ValueError("graft arm did not use per-boundary boxes")
    if float(graft["nu"]) != float(configuration["nu"]):
        raise ValueError("graft nu differs from predeclaration")

    if quality["model"] != model["id"]:
        raise ValueError("quality model differs from predeclaration")
    if quality["model_revision"] != model["revision"]:
        raise ValueError("quality revision differs from predeclaration")
    if int(quality["evaluation"]["blocks"]) != int(
        predeclaration["quality"]["blocks"]
    ):
        raise ValueError("quality block count differs from predeclaration")

    expected_token_hash = predeclaration["quality"]["expected_token_sha256"]
    observed_token_hash = quality["evaluation"]["token_ids_sha256"]
    return {
        "schema": "gemma-sv-training-free-second-checkpoint-result-v1",
        "status": "complete-predeclared-primary-outcomes",
        "predeclaration": {
            "path": predeclaration_path.name,
            "sha256": _sha256(predeclaration_path),
        },
        "model": model,
        "configuration": configuration,
        "confirmation": {
            "manifest": confirmation["manifest"],
            "sha256": confirmation["sha256"],
            "ungrafted": _admission_summary(ungrafted),
            "graft": _admission_summary(graft),
        },
        "quality": {
            "evaluation": quality["evaluation"],
            "token_hash_matches_predeclaration": (
                observed_token_hash == expected_token_hash
            ),
            "summary": quality["summary"],
        },
        "source_reports": {
            "ungrafted_admission_sha256": _sha256(ungrafted_path),
            "graft_admission_sha256": _sha256(graft_path),
            "quality_sha256": _sha256(quality_path),
        },
        "reporting_policy": predeclaration["reporting_policy"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predeclaration",
        type=Path,
        default=ROOT / "benchmarks/training_free_second_checkpoint_v1.json",
    )
    parser.add_argument(
        "--ungrafted",
        type=Path,
        default=DEFAULT_RUN_ROOT / "12b-pt-ungrafted-admission.json",
    )
    parser.add_argument(
        "--graft",
        type=Path,
        default=DEFAULT_RUN_ROOT / "12b-pt-graft-admission.json",
    )
    parser.add_argument(
        "--quality",
        type=Path,
        default=DEFAULT_RUN_ROOT / "12b-pt-quality-400.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "benchmarks/training_free_second_checkpoint_result_v1.json",
    )
    args = parser.parse_args()
    result = summarize(
        args.predeclaration,
        args.ungrafted,
        args.graft,
        args.quality,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
