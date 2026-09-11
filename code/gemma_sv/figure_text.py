"""Text placement helpers that keep figure labels inside their containers.

Matplotlib does not clip or reflow text to a patch, so a label that grows
past its box silently overprints neighbouring artwork. These helpers measure
the rendered extent and shrink the label until it fits, which keeps every
figure legible at single-column print size without hand-tuned font sizes.
"""
from __future__ import annotations

from typing import Sequence

Rect = Sequence[float]

_SHRINK_STEP = 0.2
_MIN_FONTSIZE = 5.5


def _renderer(figure):
    canvas = figure.canvas
    if hasattr(canvas, "get_renderer"):
        return canvas.get_renderer()
    canvas.draw()
    return canvas.get_renderer()


def box_extent(ax, rect: Rect, *, transform=None, margin: float = 0.88):
    """Return the usable display width/height of a container rectangle."""
    x, y, width, height = rect
    transform = ax.transData if transform is None else transform
    (x0, y0), (x1, y1) = transform.transform([(x, y), (x + width, y + height)])
    return abs(x1 - x0) * margin, abs(y1 - y0) * margin


def fit_text(
    ax,
    rect: Rect,
    label: str,
    *,
    color: str,
    fontsize: float,
    weight: str = "bold",
    linespacing: float = 1.06,
    transform=None,
    margin: float = 0.88,
    align: str = "center",
):
    """Draw ``label`` inside ``rect``, shrinking it until it fits.

    ``rect`` is ``(x, y, width, height)`` in the given transform's units and
    is the container the text may not leave. The default transform is
    ``ax.transData`` so a rectangle can be given in the same coordinates as
    the patch it labels.
    """
    x, y, width, height = rect
    transform = ax.transData if transform is None else transform
    limit_width, limit_height = box_extent(
        ax, rect, transform=transform, margin=margin
    )
    if align == "left":
        anchor_x, horizontal = x + 0.04 * width, "left"
    else:
        anchor_x, horizontal = x + width / 2, "center"
    text = ax.text(
        anchor_x,
        y + height / 2,
        label,
        ha=horizontal,
        va="center",
        color=color,
        fontsize=fontsize,
        weight=weight,
        linespacing=linespacing,
        transform=transform,
    )
    renderer = _renderer(ax.figure)
    size = fontsize
    while size > _MIN_FONTSIZE:
        extent = text.get_window_extent(renderer=renderer)
        if extent.width <= limit_width and extent.height <= limit_height:
            break
        size -= _SHRINK_STEP
        text.set_fontsize(size)
    return text


def overflowing_labels(figure, containers) -> list[str]:
    """Return labels whose rendered extent leaves their declared container."""
    renderer = _renderer(figure)
    escaped = []
    for text, rect, transform in containers:
        (x0, y0), (x1, y1) = transform.transform(
            [(rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3])]
        )
        extent = text.get_window_extent(renderer=renderer)
        if (
            extent.x0 < min(x0, x1) - 0.5
            or extent.x1 > max(x0, x1) + 0.5
            or extent.y0 < min(y0, y1) - 0.5
            or extent.y1 > max(y0, y1) + 0.5
        ):
            escaped.append(text.get_text())
    return escaped


def overlapping_text(figure) -> list[tuple[str, str]]:
    """Return pairs of text artists whose rendered extents intersect."""
    renderer = _renderer(figure)
    items = []
    for axis in figure.axes:
        items.extend(axis.texts)
    items.extend(figure.texts)
    measured = [
        (text.get_text(), text.get_window_extent(renderer=renderer))
        for text in items
        if text.get_text().strip()
    ]
    clashes = []
    for index, (label, extent) in enumerate(measured):
        for other_label, other_extent in measured[index + 1 :]:
            overlap_x = min(extent.x1, other_extent.x1) - max(
                extent.x0, other_extent.x0
            )
            overlap_y = min(extent.y1, other_extent.y1) - max(
                extent.y0, other_extent.y0
            )
            if overlap_x > 0.5 and overlap_y > 0.5:
                clashes.append((label, other_label))
    return clashes
