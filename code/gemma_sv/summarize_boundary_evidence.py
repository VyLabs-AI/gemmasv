"""Publish compact corrected-configuration evidence from frozen local reports."""
from __future__ import annotations

import math
import random
import statistics

from gemma_sv.robust_eval import expected_max_at_k


def _percentile(values, probability):
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_bootstrap(values, *, draws=10_000, seed=0):
    rows = [float(value) for value in values]
    rng = random.Random(seed)
    samples = [
        statistics.mean(rows[rng.randrange(len(rows))] for _ in rows)
        for _ in range(draws)
    ]
    return {
        "mean": statistics.mean(rows),
        "ci95": [_percentile(samples, 0.025), _percentile(samples, 0.975)],
        "unit": "whole record",
        "n": len(rows),
        "draws": draws,
    }


def _count(value, *, name, record_id):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"record {record_id} has invalid {name}: {value!r}")
    return value


def _record_decrement_counts(record):
    """Return attempted, fallback, and enumerated counts for one record."""
    record_id = record.get("record_id", "<unknown>")
    probes = list(record["probe_certificates"].values())
    if not probes:
        raise ValueError(f"record {record_id} has no certificate probes")
    attempted_counts = {
        _count(
            probe["head_gates"],
            name="attempted decrement count",
            record_id=record_id,
        )
        for probe in probes
    }
    fallback_counts = {
        _count(
            probe["decrement_fallbacks"],
            name="probe fallback count",
            record_id=record_id,
        )
        for probe in probes
    }
    if len(attempted_counts) != 1:
        raise ValueError(
            f"record {record_id} reports inconsistent attempted decrement "
            f"counts across probes: {sorted(attempted_counts)}"
        )
    if len(fallback_counts) != 1:
        raise ValueError(
            f"record {record_id} reports inconsistent fallback counts "
            f"across probes: {sorted(fallback_counts)}"
        )
    attempted = attempted_counts.pop()
    probe_fallbacks = fallback_counts.pop()
    fallbacks = _count(
        record["total_fallbacks"],
        name="record fallback count",
        record_id=record_id,
    )
    enumerated = _count(
        record["head_gates"],
        name="enumerated head-gate count",
        record_id=record_id,
    )
    if probe_fallbacks != fallbacks:
        raise ValueError(
            f"record {record_id} probe/record fallback counts disagree: "
            f"{probe_fallbacks} != {fallbacks}"
        )
    if fallbacks > attempted:
        raise ValueError(
            f"record {record_id} has more fallbacks than decrement attempts"
        )
    if attempted > enumerated:
        raise ValueError(
            f"record {record_id} has more decrement attempts than enumerated gates"
        )
    return attempted, fallbacks, enumerated


def _certificate(report):
    rows = [_record_decrement_counts(record) for record in report["records"]]
    probes = [
        probe
        for record in report["records"]
        for probe in record["probe_certificates"].values()
    ]
    if not probes:
        raise ValueError("certificate report has no probes")
    exact = [probe["kl_exact_vs_refit_nats"] for probe in probes]
    decay = [probe["kl_decay_vs_refit_nats"] for probe in probes]
    if not all(math.isfinite(float(value)) for value in [*exact, *decay]):
        raise ValueError("certificate report contains non-finite KL values")
    affected = sum(row[0] for row in rows)
    fallbacks = sum(row[1] for row in rows)
    enumerated = sum(row[2] for row in rows)
    if affected == 0:
        raise ValueError("certificate report has no affected decrement attempts")
    fallback_fraction = fallbacks / affected
    return {
        "records": len(report["records"]),
        "probes": len(probes),
        "maximum_exact_refit_kl_nats": max(exact),
        "median_exact_refit_kl_nats": statistics.median(exact),
        "probes_above_1e_6": sum(value > 1e-6 for value in exact),
        "median_decay_refit_kl_nats": statistics.median(decay),
        "maximum_decay_refit_kl_nats": max(decay),
        "decrement_fallbacks": fallbacks,
        "affected_decrement_attempts": affected,
        "enumerated_head_gates": enumerated,
        "fallback_denominator": "affected_decrement_attempts",
        "fallback_fraction": fallback_fraction,
        "fallback_percent_1dp": round(fallback_fraction * 100.0, 1),
    }


def _record_leak(record, condition, k):
    fields = record["conditions"][condition]["fields"]
    return statistics.mean(
        expected_max_at_k(field["exact_phrase"], k) for field in fields
    )


def _record_composite(record, condition, k):
    return expected_max_at_k(
        record["conditions"][condition]["composite_any_exact"],
        k,
    )


def _attacks(whole, elicitation, lira, relearning):
    conditions = list(whole["sampling"]["conditions"])
    records = list(whole["whole_record"]["records"])
    curves = whole["whole_record"]["exact_phrase_leak_at_k"]
    composite = whole["whole_record"]["composite_any_leak_at_k"]
    paired = {}
    for k in (1, 200):
        differences = [
            _record_leak(record, "decrement", k)
            - _record_leak(record, "never", k)
            for record in records
        ]
        paired[str(k)] = paired_bootstrap(differences, seed=271_828 + k)
        paired[str(k)]["composite_difference"] = paired_bootstrap(
            [
                _record_composite(record, "decrement", k)
                - _record_composite(record, "never", k)
                for record in records
            ],
            seed=314_159 + k,
        )
    elicitation_summary = {}
    for shot in elicitation["shots"]:
        key = str(shot)
        elicitation_summary[key] = {}
        for condition in ("masked_refit", "decay", "icul"):
            values = [
                record["shots"][key]["record_mean_recovery"][condition]
                for record in elicitation["records"]
            ]
            elicitation_summary[key][condition] = paired_bootstrap(
                values,
                seed=shot * 100 + len(condition),
            )
    relearning_summary = {
        str(row["budget"]): {
            "mean_record_recovery": row["record_mean_recovery"],
            "records": len(row["records"]),
        }
        for row in relearning["budget_rows"]
    }
    return {
        "records": len(records),
        "field_queries": sum(
            len(record["conditions"]["present"]["fields"]) for record in records
        ),
        "samples_per_prompt": whole["sampling"]["samples"],
        "conditions": conditions,
        "field_leak_at_k": {
            condition: curves[condition]
            for condition in conditions
        },
        "composite_leak_at_k": {
            condition: composite[condition]
            for condition in conditions
        },
        "masked_refit_minus_never": paired,
        "elicitation": elicitation_summary,
        "lira": lira["metrics"],
        "relearning": relearning_summary,
        "full_repack_fallbacks": {
            "elicitation": sum(
                record["fallback_reason"] is not None
                for record in elicitation["records"]
            ),
            "lira": sum(
                record["full_repack_fallbacks"] for record in lira["records"]
            ),
            "relearning": sum(
                record["fallback_reason"] is not None
                for budget in relearning["budget_rows"]
                for record in budget["records"]
            ),
        },
    }


def main(argv=None):
    from gemma_sv.publish_boundary_evidence_v3 import main as publish_v3

    return publish_v3(argv)


if __name__ == "__main__":
    raise SystemExit(main())
