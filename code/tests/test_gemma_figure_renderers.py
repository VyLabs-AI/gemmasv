from __future__ import annotations

import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import matplotlib.pyplot as plt
import numpy as np

from gemma_sv import make_boundary_attack_figure
from gemma_sv import make_concept_hero
from gemma_sv import make_audit_boundaries_figure
from gemma_sv import make_forget_figure
from gemma_sv import make_hero_figure
from gemma_sv import make_longmemeval_summary_figure
from gemma_sv import make_method_figure
from gemma_sv import make_robust_figure
from gemma_sv.figure_text import overlapping_text


def _figure_text(figure) -> str:
    items = list(figure.texts)
    for axis in figure.axes:
        items.extend(axis.texts)
        items.extend((axis.title, axis._left_title, axis._right_title))
        items.append(axis.xaxis.label)
        items.append(axis.yaxis.label)
    return "\n".join(item.get_text() for item in items)


def _render_twice(
    case: unittest.TestCase,
    *,
    builder,
    writer,
    root: Path,
    stem: str,
) -> None:
    figure = builder()
    first = writer(figure, root / f"{stem}-a", preview=None)
    plt.close(figure)
    figure = builder()
    second = writer(figure, root / f"{stem}-b", preview=None)
    plt.close(figure)

    case.assertEqual(
        {path.suffix for path in first},
        {".svg", ".pdf", ".png"},
    )
    for left, right in zip(first, second):
        case.assertEqual(left.read_bytes(), right.read_bytes())

    svg = first[0].read_text(encoding="utf-8")
    case.assertIn('role="img"', svg)
    case.assertIn("aria-labelledby=", svg)
    case.assertIn("<title id=", svg)
    case.assertIn("<desc id=", svg)
    png = first[2].read_bytes()
    case.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
    case.assertEqual(png.count(b"sRGB"), 1)
    case.assertNotIn(b"iCCP", png)


class FigureLegibilityTests(unittest.TestCase):
    """Every published figure must keep its labels apart and readable."""

    def _assert_no_overlap(self, figure) -> None:
        clashes = overlapping_text(figure)
        plt.close(figure)
        self.assertEqual(clashes, [], f"overlapping labels: {clashes}")

    def test_published_python_figures_have_no_overlapping_labels(self) -> None:
        for builder in (
            make_concept_hero.build_figure,
            make_longmemeval_summary_figure.build_figure,
            make_audit_boundaries_figure.build_figure,
            make_boundary_attack_figure.build_figure,
            make_method_figure.build_figure,
        ):
            with self.subTest(builder=builder.__module__):
                self._assert_no_overlap(builder())

    def test_published_figures_use_readable_type(self) -> None:
        for builder in (
            make_concept_hero.build_figure,
            make_longmemeval_summary_figure.build_figure,
        ):
            figure = builder()
            sizes = [
                text.get_fontsize()
                for axis in figure.axes
                for text in axis.texts
                if text.get_text().strip()
            ]
            plt.close(figure)
            with self.subTest(builder=builder.__module__):
                self.assertTrue(sizes)
                self.assertGreaterEqual(min(sizes), 7.5)


class GemmaFigureRendererTests(unittest.TestCase):
    def test_concept_is_a_complementary_reference_taxonomy(self) -> None:
        figure = make_concept_hero.build_figure()
        text = _figure_text(figure)
        plt.close(figure)

        self.assertIn("Where memory lives", text)
        self.assertIn("REPRESENTATION", text)
        self.assertIn("addressable rows\nGemma", text)
        self.assertIn("refit what remains", text)
        self.assertIn("carry effect\nforward", text)
        self.assertIn("checkpoint\n+ replay", text)
        self.assertIn("rebuild without\nrecord", text)
        self.assertIn("RECOMPUTE LATER CONSEQUENCES", text)
        self.assertIn("NOT CHANGED", text)
        self.assertNotIn("privacy complete", text.lower())
        self.assertNotIn("permanently deleted", text.lower())

    def test_method_keeps_reference_and_nonclaim_visible(self) -> None:
        figure = make_method_figure.build_figure()
        text = _figure_text(figure)
        plt.close(figure)

        self.assertNotIn("edit one address", text)
        self.assertIn("STORE", text)
        self.assertIn("addressable\nmemory rows", text)
        self.assertIn("remove rows", text)
        self.assertIn("exact-refit fallback", text)
        self.assertIn("retained-key refit", text)
        self.assertIn("SAME REGISTERED PROMPT", text)
        self.assertIn("matches the exact refit", text)
        self.assertIn("fresh-history equality", text)
        self.assertNotIn("never ingested", text.lower())
        self.assertNotIn("faster", text.lower())

    def test_boundary_summary_preserves_values_and_denominators(self) -> None:
        summary = make_boundary_attack_figure.summarize_data(
            make_boundary_attack_figure.load_data()
        )

        np.testing.assert_array_equal(
            summary["leak"]["budgets"],
            [1, 2, 4, 8, 16, 32, 64, 128, 200],
        )
        curves = summary["leak"]["curves"]
        self.assertAlmostEqual(curves["decrement"][-1], 100.0 / 6.0)
        self.assertAlmostEqual(curves["never"][-1], 100.0 / 6.0)
        self.assertAlmostEqual(curves["decay"][-1], 100.0 * 5.0 / 18.0)
        self.assertAlmostEqual(curves["present"][-1], 100.0)
        self.assertAlmostEqual(curves["icul"][-1], 100.0)
        self.assertEqual(summary["records"], 6)
        self.assertEqual(summary["field_queries"], 18)
        self.assertEqual(summary["samples_per_prompt"], 200)

        elicitation = summary["elicitation"]
        np.testing.assert_array_equal(elicitation["shots"], [0, 1, 2, 4, 8])
        self.assertAlmostEqual(
            elicitation["conditions"]["decrement"]["mean"][-1],
            0.011102896501893868,
        )
        np.testing.assert_array_equal(
            summary["relearning"]["budgets"],
            [0, 4, 16, 64, 256],
        )
        self.assertAlmostEqual(
            summary["relearning"]["mean_record_recovery"][1],
            -0.08766293894102324,
        )

        lira = summary["lira"]
        self.assertEqual((lira["admitted"], lira["attempted"]), (16, 20))
        self.assertEqual(lira["full_repack_fallbacks"], 16)
        edited = lira["conditions"]["masked_refit"]
        self.assertAlmostEqual(edited["auc"], 0.517303466796875)
        self.assertAlmostEqual(edited["tpr_at_1pct_fpr"], 0.013671875)
        self.assertEqual(edited["positive_tests"], 512)
        self.assertEqual(edited["negative_tests"], 512)

    def test_boundary_artwork_marks_behavioral_scope(self) -> None:
        figure = make_boundary_attack_figure.build_figure()
        text = _figure_text(figure)
        legend_labels = {
            entry.get_text()
            for legend in figure.legends
            for entry in legend.get_texts()
        }
        plt.close(figure)

        self.assertEqual(
            legend_labels,
            {
                "record present",
                "prompt-only",
                "decay",
                "edited memory",
                "never stored",
            },
        )

        self.assertIn("separate from numerical certificate", text)
        self.assertIn("chance", text)
        self.assertIn("edited 0.517", text)
        self.assertNotIn("Policy LiRA is near chance", text)
        self.assertIn("(d) Membership inference", text)
        self.assertNotIn("privacy", text.lower())
        self.assertNotIn("certified deletion", text.lower())

    def test_longmemeval_summary_preserves_headline_values(self) -> None:
        summary = make_longmemeval_summary_figure.summarize_data(
            make_longmemeval_summary_figure.load_data()
        )
        self.assertAlmostEqual(summary["suppression"]["edited memory"], 8.327004143363855)
        self.assertAlmostEqual(summary["suppression"]["prompt-only"], 0.5342220649363307)
        self.assertAlmostEqual(summary["retained_abs_drift"], 8.054104423996529e-05)
        self.assertAlmostEqual(summary["retained_kl"], 1.8951008830790114e-06)
        self.assertAlmostEqual(summary["target_kl"], 2.1517010661192484)
        self.assertAlmostEqual(summary["certificate_kl"], 1.7528497686921997e-16)
        self.assertEqual((summary["records"], summary["certificate_probes"]), (10, 32))

    def test_longmemeval_summary_separates_references(self) -> None:
        figure = make_longmemeval_summary_figure.build_figure()
        text = _figure_text(figure)
        plt.close(figure)
        self.assertIn("POLICY vs FRESH OMISSION", text)
        self.assertIn("IMPLEMENTATION CHECK", text)
        self.assertIn("executed vs independent refit", text)
        self.assertIn("16/16 refit fallback", text)
        self.assertIn("target KL = 2.15 nats", text)
        self.assertIn("maximum KL = 1.75e-16", text)

    def test_longmemeval_summary_rejects_scope_or_value_tampering(self) -> None:
        source = make_longmemeval_summary_figure.load_data()
        tampered_scope = copy.deepcopy(source)
        tampered_scope["scope"]["no_raw_history_state_equality_claim"] = False
        with self.assertRaises(ValueError):
            make_longmemeval_summary_figure.summarize_data(tampered_scope)

        tampered_value = copy.deepcopy(source)
        tampered_value["efficacy"]["conditions"]["exact_decrement"][
            "mean_target_suppression_vs_present_nats"
        ] = 8.34
        with self.assertRaises(ValueError):
            make_longmemeval_summary_figure.summarize_data(tampered_value)

    def test_concept_renderer_is_byte_deterministic(self) -> None:
        with TemporaryDirectory() as directory:
            _render_twice(
                self,
                builder=make_concept_hero.build_figure,
                writer=make_concept_hero.write_figure,
                root=Path(directory),
                stem="hero",
            )

    def test_method_renderer_is_byte_deterministic(self) -> None:
        with TemporaryDirectory() as directory:
            _render_twice(
                self,
                builder=make_method_figure.build_figure,
                writer=make_method_figure.write_figure,
                root=Path(directory),
                stem="method",
            )

    def test_boundary_attack_renderer_is_byte_deterministic(self) -> None:
        with TemporaryDirectory() as directory:
            _render_twice(
                self,
                builder=make_boundary_attack_figure.build_figure,
                writer=make_boundary_attack_figure.write_figure,
                root=Path(directory),
                stem="boundary",
            )

    def test_longmemeval_summary_renderer_is_byte_deterministic(self) -> None:
        with TemporaryDirectory() as directory:
            _render_twice(
                self,
                builder=make_longmemeval_summary_figure.build_figure,
                writer=lambda figure, path, preview=None: (
                    make_longmemeval_summary_figure.write_figure(figure, path)
                ),
                root=Path(directory),
                stem="longmemeval-summary",
            )

    def test_audit_boundaries_renderer_is_byte_deterministic(self) -> None:
        with TemporaryDirectory() as directory:
            _render_twice(
                self,
                builder=make_audit_boundaries_figure.build_figure,
                writer=lambda figure, path, preview=None: (
                    make_audit_boundaries_figure.write_figure(figure, path)
                ),
                root=Path(directory),
                stem="audit-boundaries",
            )

    def test_forgetting_renderer_is_byte_deterministic(self) -> None:
        with TemporaryDirectory() as directory:
            _render_twice(
                self,
                builder=make_forget_figure.build_public_figure,
                writer=lambda figure, path, preview=None: (
                    make_forget_figure.write_public_figure(figure, path)
                ),
                root=Path(directory),
                stem="forgetting",
            )

    def test_robust_renderer_is_byte_deterministic(self) -> None:
        with TemporaryDirectory() as directory:
            _render_twice(
                self,
                builder=make_robust_figure.build_public_figure,
                writer=lambda figure, path, preview=None: (
                    make_robust_figure.write_public_figure(figure, path)
                ),
                root=Path(directory),
                stem="robust",
            )

    def test_whole_record_renderer_is_byte_deterministic(self) -> None:
        with TemporaryDirectory() as directory:
            _render_twice(
                self,
                builder=make_hero_figure.build_public_figure,
                writer=lambda figure, path, preview=None: (
                    make_hero_figure.write_public_figure(figure, path)
                ),
                root=Path(directory),
                stem="whole-record",
            )


if __name__ == "__main__":
    unittest.main()
