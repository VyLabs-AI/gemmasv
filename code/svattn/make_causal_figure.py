"""Schematic for Section 3.3 (the causal, chunk-frozen, multi-head layer).

A reproducible diagram (no data; pure schematic) in the paper's figure style:

  Left  -- one head's causal chunk-frozen stream. A single gate is fit on the
           causal prefix (earlier chunks); its reserve tokens (alpha = 0) are
           evicted, the margin/error tokens are the bounded working set. The
           queries in the current chunk read that frozen gate. (One such gate per
           head; one head shown.)
  Right -- bounded working state: as the context grows, full attention's state
           grows with it, while the SV layer's working set is capped at the
           budget B by certified-reserve eviction.

Run (matplotlib; ferg venv):
    PYTHONPATH=. \
        ./ferg/.venv/bin/python -m svattn.make_causal_figure
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from svattn import figstyle as fs

OUT = Path("./outputs")

# prefix partition (8 tokens): r = reserve (evicted), m = margin S, e = error E
PREFIX = list("rmrr" "mrme")
COLMAP = {"m": fs.BLUE, "e": fs.RED, "r": fs.GRAY}


def token(ax, cx, cy, fc, faded=False, query=False):
    w = h = 0.78
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.01,rounding_size=0.12",
                 fc=fc, ec=(fs.INK if query else "white"),
                 lw=(2.0 if query else 0.8), alpha=(0.3 if faded else 1.0),
                 zorder=3, mutation_aspect=1))


def span_bar(ax, x0, x1, y, label, above=True):
    tick = 0.12 if above else -0.12
    ax.plot([x0, x0, x1, x1], [y - tick, y, y, y - tick],
            color=fs.DARKGRAY, lw=1.3, zorder=2)
    ax.text((x0 + x1) / 2, y + (0.18 if above else -0.18), label, ha="center",
            va=("bottom" if above else "top"), fontsize=8.6, color=fs.INK,
            weight="bold")


def draw_stream(ax):
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-0.55, 3.25)
    ax.axis("off")
    cy = 1.75

    for i, p in enumerate(PREFIX):                       # prefix: chunks < t
        token(ax, i + 0.5, cy, COLMAP[p], faded=(p == "r"))
    for i in range(8, 12):                               # current chunk: queries
        token(ax, i + 0.5, cy, "#eef1ff", query=True)

    ax.plot([8, 8], [cy - 0.62, cy + 0.62], ls=(0, (4, 3)), color="#b8b8b8",
            lw=1.1, zorder=1)

    # bracket over the current chunk; bracket under the prefix
    span_bar(ax, 8.1, 11.9, cy + 0.5, "chunk $t$: queries read the frozen gate",
             above=True)
    span_bar(ax, 0.1, 7.9, cy - 0.5,
             "causal prefix (chunks $<t$): one gate fit here\n"
             "faded $=$ reserve ($\\alpha\\!=\\!0$), evicted; solid $=$ working set",
             above=False)

    # the read direction: queries (right) read the gate fit on the prefix (left)
    ax.add_patch(FancyArrowPatch((8.4, cy + 0.92), (3.9, cy + 0.92),
                 connectionstyle="arc3,rad=0.28", arrowstyle="-|>",
                 mutation_scale=15, lw=1.8, color=fs.INK, zorder=5))

    # boundary operation, placed in the white space below the stream
    ax.text(6, 0.45, "at each boundary:  add keys $\\to$ re-solve $\\to$ evict "
            "new reserve  ($\\Rightarrow$ working set $\\leq$ budget $B$)",
            ha="center", va="center", fontsize=8.6, color=fs.DARKGRAY)

    # legend
    handles = [plt.Line2D([0], [0], marker="s", ls="", markersize=9,
               markerfacecolor=fs.BLUE, markeredgecolor="white", label="margin $S$ (kept)"),
               plt.Line2D([0], [0], marker="s", ls="", markersize=9,
               markerfacecolor=fs.RED, markeredgecolor="white", label="error $E$ (kept)"),
               plt.Line2D([0], [0], marker="s", ls="", markersize=9, alpha=0.3,
               markerfacecolor=fs.GRAY, markeredgecolor="white", label="reserve (evicted)"),
               plt.Line2D([0], [0], marker="s", ls="", markersize=9,
               markerfacecolor="#eef1ff", markeredgecolor=fs.INK, label="query")]
    ax.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.4,
              frameon=False, bbox_to_anchor=(0.5, -0.16), handletextpad=.3,
              columnspacing=1.1)
    fs.caps_title(ax, "causal chunk-frozen streaming (one gate per head)")


def draw_budget(ax):
    pos = np.array([4, 8, 12, 16])
    full = pos.astype(float)                       # full attention: state = context len
    sv = np.array([3.0, 5.0, 6.0, 6.0])            # SV working set, capped at B
    B = 6.0
    ax.plot(pos, full, "o-", color=fs.RED, lw=1.9, label="full attention (grows)")
    ax.step(pos, sv, where="mid", color=fs.BLUE, lw=2.3,
            label="SV working set", marker="o")
    ax.axhline(B, ls="--", color=fs.DARKGRAY, lw=1.2)
    ax.text(4.2, B + 0.25, "budget $B$", fontsize=8.4, color=fs.DARKGRAY)
    ax.set_xlabel("tokens processed", fontsize=9)
    ax.set_ylabel("working-state size", fontsize=9)
    ax.set_xticks(pos)
    ax.set_ylim(0, 17)
    ax.grid(alpha=.3)
    ax.legend(fontsize=7.6, loc="upper left")
    fs.caps_title(ax, "bounded state")


def main():
    fs.apply()
    fig = plt.figure(figsize=(12.6, 3.9))
    gs = fig.add_gridspec(1, 3, wspace=0.28, left=0.045, right=0.985,
                          top=0.90, bottom=0.10)
    draw_stream(fig.add_subplot(gs[0, :2]))
    draw_budget(fig.add_subplot(gs[0, 2]))
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "svattn_causal.png", bbox_inches="tight")
    print(f"saved {OUT / 'svattn_causal.png'}")


if __name__ == "__main__":
    main()
