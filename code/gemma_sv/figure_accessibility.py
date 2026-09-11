"""Deterministic accessibility and color metadata for generated figures."""
from __future__ import annotations

import binascii
from html import escape
from pathlib import Path
import struct


def add_svg_accessibility(
    path: Path,
    *,
    figure_id: str,
    title: str,
    description: str,
) -> None:
    """Add canonical title/description IDs and ARIA references to an SVG."""
    text = path.read_text(encoding="utf-8")
    marker = "<svg "
    start = text.index(marker)
    end = text.index(">", start) + 1
    title_id = f"{figure_id}-title"
    description_id = f"{figure_id}-desc"
    labelled = text[start:end].replace(
        marker,
        (
            f'<svg role="img" aria-labelledby="{title_id}" '
            f'aria-describedby="{description_id}" '
        ),
        1,
    )
    metadata = (
        f'\n <title id="{title_id}">{escape(title)}</title>'
        f'\n <desc id="{description_id}">{escape(description)}</desc>'
    )
    output = text[:start] + labelled + metadata + text[end:]
    output = "\n".join(line.rstrip() for line in output.splitlines()) + "\n"
    path.write_text(output, encoding="utf-8")


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_COLOR_CHUNKS = frozenset({b"sRGB", b"iCCP", b"gAMA", b"cHRM"})


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = binascii.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def normalize_png_srgb(path: Path) -> None:
    """Install one canonical perceptual-intent sRGB chunk in a PNG.

    Existing color-space declarations are removed first so the output cannot
    contain conflicting iCCP/sRGB or nonstandard gamma/chromaticity metadata.
    Pixel data, resolution, and public text metadata remain unchanged.
    """
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError(f"not a PNG: {path}")

    chunks: list[tuple[bytes, bytes]] = []
    offset = len(PNG_SIGNATURE)
    while offset < len(data):
        if offset + 12 > len(data):
            raise ValueError(f"truncated PNG chunk: {path}")
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        payload_end = offset + 8 + length
        payload = data[offset + 8 : payload_end]
        expected = struct.unpack(">I", data[payload_end : payload_end + 4])[0]
        actual = binascii.crc32(kind + payload) & 0xFFFFFFFF
        if expected != actual:
            raise ValueError(f"invalid PNG CRC for {kind!r}: {path}")
        if kind not in _COLOR_CHUNKS:
            chunks.append((kind, payload))
        offset = payload_end + 4
        if kind == b"IEND":
            break

    if not chunks or chunks[0][0] != b"IHDR" or chunks[-1][0] != b"IEND":
        raise ValueError(f"invalid PNG chunk order: {path}")
    chunks.insert(1, (b"sRGB", b"\x00"))
    path.write_bytes(
        PNG_SIGNATURE + b"".join(_png_chunk(kind, payload) for kind, payload in chunks)
    )
