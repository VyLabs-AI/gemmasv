"""Build and validate the active Gemma/LongMemEval figure manifest."""
from __future__ import annotations

import argparse
import binascii
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
DEFAULT_OUTPUT = ROOT / "reproducibility/FIGURE_METADATA.json"
INCLUSION_WIDTH_PT = 396.0


FIGURES = ({'id': 'audit_boundaries',
  'label': 'fig:audit-boundaries',
  'takeaway': 'Only Gemma 4B preserves binary admission with the frozen recipe; all scale outcomes '
              'remain reported.',
  'evidence_class': 'measured aggregate',
  'status': 'measured',
  'renderer': '../render_supplied_manuscript_figures.py',
  'data': ['code/gemma_sv/benchmarks/iclr_audit_boundaries_v2.json']},
 {'id': 'boundary_attacks',
  'label': 'fig:boundary-attacks',
  'takeaway': 'Extraction approaches the never-stored reference; pooled membership AUC is near '
              'chance, but within-record signal remains.',
  'evidence_class': 'measured behavioral audit',
  'status': 'measured',
  'renderer': '../render_supplied_manuscript_figures.py',
  'data': ['code/gemma_sv/benchmarks/iclr_mass_preserving_boundary_v2.json',
           'code/gemma_sv/benchmarks/boundary_lira_record_bootstrap_v1.json']},
 {'id': 'gemmasv_leak_verbatim',
  'label': 'fig:leak-verbatim',
  'takeaway': 'Prompt-only suppression discloses directly; edited policy has no exact hit on the '
              'shown query.',
  'evidence_class': 'measured qualitative behavioral audit',
  'status': 'measured',
  'renderer': '../render_supplied_manuscript_figures.py',
  'data': ['code/gemma_sv/benchmarks/boundary_attack_suite_v1.json',
           'code/gemma_sv/benchmarks/boundary_qualitative_generation_replay_v1.json']},
 {'id': 'longmemeval_summary',
  'label': 'fig:longmemeval-summary',
  'takeaway': 'The 2.15-nat fresh-omission gap remains distinct from a refit-path implementation '
              'check.',
  'evidence_class': 'measured conversational-memory summary',
  'status': 'measured',
  'renderer': '../render_supplied_manuscript_figures.py',
  'data': ['code/gemma_sv/benchmarks/longmemeval_chat_geometry_methods_compact16_v1.json']})


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path_record(relative: str) -> dict[str, str]:
    path = ROOT / relative
    if not path.is_file():
        raise ValueError(f"missing metadata input: {relative}")
    return {"path": relative, "sha256": sha256(path)}


def _tex_inclusions() -> dict[str, dict[str, object]]:
    pattern = re.compile(
        r"\\includegraphics\[alt=\{(?P<alt>.*?)\},width=(?P<width>[^]]+)\]"
        r"\{figs/(?P<asset>[^}]+)\}(?P<after>.*?\\label\{(?P<label>fig:[^}]+)\})",
        re.DOTALL,
    )
    observed: dict[str, dict[str, object]] = {}
    for name in ("body_gemma.tex", "appendix_gemma.tex"):
        path = PAPER / name
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            figure_id = Path(match.group("asset")).stem
            start_line = text.count("\n", 0, match.start()) + 1
            caption_match = re.search(r"\\caption\{", match.group("after"))
            if caption_match is None:
                raise ValueError(f"missing caption after {figure_id}")
            caption_line = (
                text.count(
                    "\n",
                    0,
                    match.start("after") + caption_match.start(),
                )
                + 1
            )
            observed[figure_id] = {
                "tex_label": match.group("label"),
                "alt_text": " ".join(match.group("alt").split()),
                "inclusion_width": match.group("width"),
                "caption": {
                    "path": path.relative_to(ROOT).as_posix(),
                    "includegraphics_line": start_line,
                    "caption_start_line": caption_line,
                },
            }
    return observed


def _svg_record(path: Path) -> tuple[dict[str, object], str]:
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    namespace = "{http://www.w3.org/2000/svg}"
    descriptions = root.findall(f"{namespace}desc")
    description = " ".join((descriptions[0].text or "").split()) if descriptions else ""
    if not description:
        raise ValueError(f"missing SVG description: {path}")
    view_box = [float(value) for value in root.attrib["viewBox"].split()]
    return (
        {
            "width": root.attrib["width"],
            "height": root.attrib["height"],
            "view_box": view_box,
        },
        description,
    )


def _pdf_dimensions(path: Path) -> dict[str, float]:
    result = subprocess.run(
        ["pdfinfo", str(path)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    match = re.search(r"^Page size:\s+([0-9.]+) x ([0-9.]+) pts", result, re.MULTILINE)
    if match is None:
        raise ValueError(f"cannot parse PDF dimensions: {path}")
    return {"width_pt": float(match.group(1)), "height_pt": float(match.group(2))}


def _png_dimensions(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG: {path}")
    offset = 8
    chunks: list[tuple[bytes, bytes]] = []
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        end = offset + 8 + length
        payload = data[offset + 8 : end]
        expected = struct.unpack(">I", data[end : end + 4])[0]
        if expected != (binascii.crc32(kind + payload) & 0xFFFFFFFF):
            raise ValueError(f"invalid PNG CRC: {path}")
        chunks.append((kind, payload))
        offset = end + 4
        if kind == b"IEND":
            break
    ihdr = next(payload for kind, payload in chunks if kind == b"IHDR")
    width, height = struct.unpack(">II", ihdr[:8])
    srgb = [payload for kind, payload in chunks if kind == b"sRGB"]
    iccp = [payload for kind, payload in chunks if kind == b"iCCP"]
    if srgb != [b"\x00"] or iccp:
        raise ValueError(f"PNG lacks one canonical sRGB chunk: {path}")
    effective_ppi = width / (INCLUSION_WIDTH_PT / 72.0)
    if effective_ppi < 300:
        raise ValueError(f"PNG effective resolution below 300 ppi: {path}")
    return {
        "width_px": width,
        "height_px": height,
        "color_space": "sRGB chunk; perceptual rendering intent",
        "effective_ppi_at_tex_width": round(effective_ppi, 3),
    }


def build_manifest() -> dict[str, object]:
    inclusions = _tex_inclusions()
    expected_ids = {row["id"] for row in FIGURES}
    if set(inclusions) != expected_ids:
        raise ValueError(
            f"figure inclusion mismatch: missing={expected_ids - set(inclusions)}, "
            f"extra={set(inclusions) - expected_ids}"
        )

    records = []
    for spec in FIGURES:
        figure_id = str(spec["id"])
        inclusion = inclusions[figure_id]
        if inclusion["tex_label"] != spec["label"]:
            raise ValueError(f"label mismatch for {figure_id}")
        takeaway = str(spec["takeaway"])
        if len(takeaway.split()) > 18:
            raise ValueError(f"takeaway exceeds 18 words: {figure_id}")

        outputs = {}
        long_description = ""
        for extension in ("svg", "pdf", "png"):
            relative = f"paper/figs/{figure_id}.{extension}"
            path = ROOT / relative
            record: dict[str, object] = _path_record(relative)
            if extension == "svg":
                record["dimensions"], long_description = _svg_record(path)
            elif extension == "pdf":
                record["dimensions"] = _pdf_dimensions(path)
            else:
                record["dimensions"] = _png_dimensions(path)
            outputs[extension] = record

        png_dimensions = outputs["png"]["dimensions"]
        final_height = INCLUSION_WIDTH_PT * (
            png_dimensions["height_px"] / png_dimensions["width_px"]
        )
        records.append(
            {
                "figure_id": figure_id,
                "tex_label": spec["label"],
                "alt_text": inclusion["alt_text"],
                "long_description": long_description,
                "takeaway": takeaway,
                "evidence_class": spec["evidence_class"],
                "measured_fictional_status": spec["status"],
                "source_data_or_payloads": [
                    _path_record(str(path)) for path in spec["data"]
                ],
                "canonical_renderer": _path_record(str(spec["renderer"])),
                "outputs": outputs,
                "final_dimensions": {
                    "intended_tex_inclusion_width": inclusion["inclusion_width"],
                    "compiled_width_pt": INCLUSION_WIDTH_PT,
                    "aspect_preserving_height_pt": round(final_height, 3),
                },
                "caption": inclusion["caption"],
                "release_variants": ["arxiv_v3_public"],
            }
        )

    return {
        "schema": "gemmasv-figure-metadata-v1",
        "release": "arxiv_v3_public",
        "record_count": len(records),
        "records": records,
    }


def write_manifest(path: Path = DEFAULT_OUTPUT) -> None:
    payload = build_manifest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = json.dumps(build_manifest(), indent=2, sort_keys=True) + "\n"
    if args.check:
        if args.output.read_text(encoding="utf-8") != payload:
            raise SystemExit(f"stale figure metadata: {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
