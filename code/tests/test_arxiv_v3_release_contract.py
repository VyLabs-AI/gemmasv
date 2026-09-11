from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import build_arxiv_v3 as release  # noqa: E402
from gemma_sv import build_figure_metadata  # noqa: E402


FIGURE_IDS = tuple(row["id"] for row in build_figure_metadata.FIGURES)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("figure_id", FIGURE_IDS)
def test_canonical_png_has_one_srgb_declaration_and_release_resolution(
    figure_id: str,
) -> None:
    path = ROOT / f"paper/figs/{figure_id}.png"
    dimensions = build_figure_metadata._png_dimensions(path)
    data = path.read_bytes()
    assert data.count(b"sRGB") == 1
    assert b"iCCP" not in data
    assert b"/Users/" not in data
    assert b"/Volumes/" not in data
    assert dimensions["effective_ppi_at_tex_width"] >= 300


def test_figure_metadata_is_current_and_covers_every_tex_inclusion() -> None:
    recorded = json.loads(
        (ROOT / "reproducibility/FIGURE_METADATA.json").read_text()
    )
    expected = build_figure_metadata.build_manifest()
    assert recorded == expected
    assert recorded["record_count"] == len(FIGURE_IDS)
    assert {row["figure_id"] for row in recorded["records"]} == set(FIGURE_IDS)
    assert len({row["tex_label"] for row in recorded["records"]}) == len(FIGURE_IDS)
    for row in recorded["records"]:
        assert len(row["takeaway"].split()) <= 18
        assert row["alt_text"]
        assert row["long_description"]
        assert row["canonical_renderer"]["sha256"] == _sha256(
            ROOT / row["canonical_renderer"]["path"]
        )
        for source in row["source_data_or_payloads"]:
            assert source["sha256"] == _sha256(ROOT / source["path"])
        for output in row["outputs"].values():
            assert output["sha256"] == _sha256(ROOT / output["path"])

    qualitative = next(
        row
        for row in recorded["records"]
        if row["figure_id"] == "gemmasv_leak_verbatim"
    )
    assert "registered exact-phrase endpoint" in qualitative["long_description"]
    svg = (ROOT / qualitative["outputs"]["svg"]["path"]).read_text()
    assert 'y="924"' not in svg
    body = (ROOT / "paper/body_gemma.tex").read_text()
    assert "registered exact-phrase endpoint" in body
    assert "different unrelated completions" not in body


def test_human_adjudicated_artifacts_and_macros_are_current() -> None:
    release.validate_human_adjudicated_artifacts()
    human = json.loads(release.HUMAN_RESULTS.read_text(encoding="utf-8"))
    census = json.loads(release.ADJUDICATED_CENSUS.read_text(encoding="utf-8"))
    assert human["primary"]["final_label_counts"] == {
        "ambiguous": 1,
        "leak": 15,
        "no_leak": 22,
    }
    assert census["geometry"]["matcher_clean_outcome_counts"] == {
        "ambiguous": 3,
        "leak": 15,
        "missing": 0,
        "no_leak": 235,
        "parse_error": 0,
        "transport_error": 0,
    }


def test_source_archive_exactly_matches_current_release_inputs() -> None:
    archive = (
        ROOT
        / "releases/arxiv_v3/ai_assistant_forget_v3_arxiv_source.tar.gz"
    )
    expected = {
        name: ROOT / "paper" / name for name in release.SOURCE_FILES
    }
    expected.update(
        {
            f"figs/{name}": ROOT / "paper/figs" / name
            for name in release.FIGURES
        }
    )
    expected[release.ARCHIVE_FIGURE_METADATA_NAME] = release.FIGURE_METADATA

    with tarfile.open(archive, "r:gz") as handle:
        members = {member.name: member for member in handle if member.isfile()}
        assert set(members) == set(expected)
        for name, source in expected.items():
            extracted = handle.extractfile(members[name])
            assert extracted is not None
            assert extracted.read() == source.read_bytes()


def test_release_manifest_handoff_pages_and_hashes_are_current() -> None:
    manifest_path = ROOT / "reproducibility/RELEASE_MANIFEST.json"
    handoff_path = ROOT / "reproducibility/UPLOAD_HANDOFF.md"
    manifest = json.loads(manifest_path.read_text())
    artifacts = manifest["artifacts"]

    for key in ("preflight_pdf", "source_archive", "replacement_record"):
        record = artifacts[key]
        assert record["sha256"] == _sha256(ROOT / record["path"])
    assert artifacts["preflight_pdf"]["pages"] == release.pdf_pages(
        ROOT / artifacts["preflight_pdf"]["path"]
    )
    assert artifacts["figure_metadata"]["sha256"] == _sha256(
        ROOT / artifacts["figure_metadata"]["path"]
    )
    assert artifacts["figure_metadata"]["records"] == len(FIGURE_IDS)
    suites = manifest["validation"]["test_suites"]
    assert suites["complete_python"] == {
        "collected": 383,
        "passed": 368,
        "skipped_for_excluded_local_inputs": 15,
        "failed": 0,
    }
    assert suites["python_renderer"]["passed"] == 17
    assert suites["javascript_renderer"]["passed"] == 11
    assert suites["provenance_regression"]["passed"] == 2
    assert suites["release_contract"]["passed"] == 11

    handoff = handoff_path.read_text()
    assert _sha256(manifest_path) in handoff
    assert str(artifacts["preflight_pdf"]["pages"]) in handoff
    for record in artifacts.values():
        if "sha256" in record:
            assert record["sha256"] in handoff
    assert "PENDING" in handoff

    checksum_lines = (
        ROOT / "releases/arxiv_v3/SHA256SUMS"
    ).read_text().splitlines()
    checksums = {
        line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
        for line in checksum_lines
    }
    for path in (
        manifest_path,
        handoff_path,
        ROOT / "reproducibility/FIGURE_METADATA.json",
    ):
        relative = path.relative_to(ROOT).as_posix()
        assert checksums[relative] == _sha256(path)
