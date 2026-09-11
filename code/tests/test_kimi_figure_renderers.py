from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import matplotlib.pyplot as plt

from kimi_sv import make_arch_figure, make_mimic_figure


def _report(
    *,
    admitted: int,
    attempted: int,
    mean: float,
    minimum: float,
    maximum: float,
    positions: list[dict] | None = None,
) -> dict:
    return {
        "admission": {"admitted": admitted, "attempted": attempted},
        "efficacy": {
            "field_lift_present_nats": {
                "mean": mean,
                "min": minimum,
                "max": maximum,
            },
            "field_max_abs_lift_after_deletion_nats": {"max": 0.0},
        },
        "exactness": {
            "logit_residual": {"max": 0.0},
            "state_residual": {"max": 0.0},
        },
        "per_position": positions or [],
    }


def mimic_reports() -> dict[str, dict]:
    positions = [
        {
            "victim_position": 127,
            "replayed_tokens": 0,
            "replay_seconds": 0.0,
            "rebuild_seconds": 6.75,
        },
        {
            "victim_position": 64,
            "replayed_tokens": 1579,
            "replay_seconds": 3.53,
            "rebuild_seconds": 6.76,
        },
        {
            "victim_position": 0,
            "replayed_tokens": 3202,
            "replay_seconds": 6.70,
            "rebuild_seconds": 6.79,
        },
    ]
    return {
        "cds_small": _report(
            admitted=8,
            attempted=8,
            mean=1.48,
            minimum=0.73,
            maximum=2.50,
        ),
        "cds_large": _report(
            admitted=9,
            attempted=9,
            mean=2.02,
            minimum=0.42,
            maximum=4.06,
            positions=positions,
        ),
        "notes": _report(
            admitted=4,
            attempted=5,
            mean=1.86,
            minimum=1.25,
            maximum=2.35,
        ),
    }


def _render_twice(
    case: unittest.TestCase,
    builder,
    writer,
    first: Path,
    second: Path,
) -> None:
    figure = builder()
    first_paths = writer(figure, first, preview=None)
    plt.close(figure)
    figure = builder()
    second_paths = writer(figure, second, preview=None)
    plt.close(figure)
    for left, right in zip(first_paths, second_paths):
        case.assertEqual(left.read_bytes(), right.read_bytes())
    png = first_paths[2].read_bytes()
    case.assertEqual(png.count(b"sRGB"), 1)
    case.assertNotIn(b"iCCP", png)


class KimiFigureRendererTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mimic_reports = mimic_reports()

    def test_mimic_summary_preserves_zero_and_denominators(self) -> None:
        summary = make_mimic_figure.summarize_reports(self.mimic_reports)
        self.assertEqual(summary["admitted"], 21)
        self.assertEqual(summary["attempted"], 22)
        self.assertEqual(summary["tokens"].tolist(), [0.0, 1579.0, 3202.0])
        self.assertEqual(summary["replay"].tolist(), [0.0, 3.53, 6.70])
        self.assertGreater(summary["r_squared"], 0.998)

        changed = {
            name: {
                **report,
                "exactness": {
                    **report["exactness"],
                    "state_residual": {"max": 1e-7},
                },
            }
            for name, report in self.mimic_reports.items()
        }
        with self.assertRaisesRegex(ValueError, "exact-zero replay"):
            make_mimic_figure.summarize_reports(changed)

    def test_architecture_artwork_keeps_reader_facing_labels(self) -> None:
        figure = make_arch_figure.build_figure()
        text = "\n".join(
            item.get_text()
            for axis in figure.axes
            for item in (*axis.texts, axis.title)
        )
        plt.close(figure)
        self.assertIn("ATTENTION", text)
        self.assertIn("KDA", text)
        self.assertIn("MASK ONE\n\u2260 DELETE", text)
        self.assertIn("checkpoint", text)
        self.assertIn("RESTORE", text)
        self.assertIn("REPLAY SUFFIX", text)
        self.assertNotIn("masking here", text)
        self.assertNotIn("clearing here", text)
        self.assertNotIn("20 recurrent", text)
        self.assertNotIn("seven global", text)

    def test_mimic_uses_separate_facets_and_clear_legend(self) -> None:
        figure = make_mimic_figure.build_figure(self.mimic_reports)
        self.assertEqual(len(figure.axes), 3)
        present, replay, timing = figure.axes
        self.assertEqual(present.get_title(), "PRESENT")
        self.assertEqual(replay.get_title(), "REPLAY = 0")
        self.assertEqual(len(present.texts), 0)
        self.assertEqual(len(replay.texts), 0)
        self.assertEqual(len(timing.texts), 0)
        self.assertEqual(
            [item.get_text() for item in timing.get_legend().get_texts()],
            ["replay timing", "linear fit", "full rebuild"],
        )
        plt.close(figure)

    def test_arch_renderer_is_byte_deterministic(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _render_twice(
                self,
                make_arch_figure.build_figure,
                make_arch_figure.write_figure,
                root / "arch-a",
                root / "arch-b",
            )

    def test_mimic_renderer_is_byte_deterministic(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _render_twice(
                self,
                lambda: make_mimic_figure.build_figure(self.mimic_reports),
                make_mimic_figure.write_figure,
                root / "mimic-a",
                root / "mimic-b",
            )


if __name__ == "__main__":
    unittest.main()
