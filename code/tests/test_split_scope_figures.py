from __future__ import annotations

import matplotlib.pyplot as plt

from gemma_sv import make_split_scope_figures as figures
from gemma_sv.figure_text import overlapping_text


def _text(figure) -> str:
    values = list(figure.texts)
    for axis in figure.axes:
        values.extend(axis.texts)
    return "\n".join(value.get_text() for value in values)


def test_claim_ladder_is_gemma_only_and_reference_explicit() -> None:
    figure = figures.build_claim_ladder()
    text = _text(figure)
    assert "Match retained-key refit" in text
    assert "max KL 6.48×10⁻¹¹" in text
    assert "2.15-nat pilot gap" in text
    assert "KDA" not in text
    assert overlapping_text(figure) == []
    plt.close(figure)


def test_taxonomy_separates_refit_repack_and_regeneration() -> None:
    figure = figures.build_deletion_taxonomy()
    text = _text(figure)
    assert "fixed-C retained-key refit" in text
    assert "history rebuilt without record" in text
    assert "causal counterfactual" in text
    assert "implementation check" in text
    assert "KDA" not in text
    assert overlapping_text(figure) == []
    plt.close(figure)
