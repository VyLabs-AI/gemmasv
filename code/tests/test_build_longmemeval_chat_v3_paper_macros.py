from __future__ import annotations

from pathlib import Path
import re

import pytest

from gemma_sv import build_longmemeval_chat_v3_paper_macros as macros


REPOSITORY = Path(__file__).resolve().parents[1]
SUMMARY = (
    REPOSITORY
    / "gemma_sv"
    / "benchmarks"
    / "longmemeval_chat_v3_summary_v1.json"
)
CENSUS = (
    REPOSITORY
    / "gemma_sv"
    / "benchmarks"
    / "longmemeval_chat_leakage_recall_census_statistics_v3.json"
)
ADJUDICATED = (
    REPOSITORY
    / "gemma_sv"
    / "benchmarks"
    / "longmemeval_chat_leakage_recall_census_human_adjudicated_v4.json"
)
COMMITTED_MACROS = (
    REPOSITORY.parent
    / "paper/longmemeval_chat_v3_decoded_macros.tex"
)
MACRO_RE = re.compile(
    r"\\providecommand\{\\(?P<name>[A-Za-z]+)\}"
    r"\{(?P<value>[^{}]+)\}"
)
SAFE_VALUE_RE = re.compile(
    r"(?:-?(?:0|[1-9][0-9]*)(?:\.[0-9])?|[0-9a-f]{64})\Z"
)


@pytest.fixture(scope="module")
def rendered() -> str:
    return macros.build_paper_macros(
        decoded_summary_path=SUMMARY,
        census_statistics_path=CENSUS,
        adjudicated_census_path=ADJUDICATED,
    )


@pytest.fixture(scope="module")
def values(rendered: str) -> dict[str, str]:
    return {
        match.group("name"): match.group("value")
        for match in MACRO_RE.finditer(rendered)
    }


def test_current_real_values_are_rendered(values: dict[str, str]):
    assert values["LMChatVThreeK"] == "32"
    assert values["LMChatVThreeN"] == "96"
    assert values["LMChatVThreeCompletedN"] == "96"
    assert values["LMChatVThreeMatcherCleanN"] == "253"
    assert values["LMChatVThreeJudgeMissN"] == "19"
    assert values["LMChatVThreeJudgeAmbiguousN"] == "2"
    assert values["LMChatVThreeJudgeFailureN"] == "0"
    assert values["LMChatVThreeJudgeMissingN"] == "0"
    assert values["LMChatVThreeJudgeParseErrorN"] == "0"
    assert values["LMChatVThreeJudgeTransportErrorN"] == "0"
    assert values["LMChatVThreeEffectiveMatcherCleanLeakN"] == "15"
    assert values["LMChatVThreeEffectiveMatcherCleanNoLeakN"] == "235"
    assert values["LMChatVThreeEffectiveMatcherCleanAmbiguousN"] == "3"
    assert values["LMChatVThreeHumanReviewN"] == "38"
    assert values["LMChatVThreeHumanUniqueTripleN"] == "21"
    assert values["LMChatVThreeHumanFlagLeakN"] == "15"
    assert values["LMChatVThreeHumanFlagNoLeakN"] == "3"
    assert values["LMChatVThreeHumanFlagAmbiguousN"] == "1"
    assert values["LMChatVThreeHumanControlLeakN"] == "0"
    assert values["LMChatVThreeHumanControlNoLeakN"] == "19"
    assert values["LMChatVThreeHumanNormalizationN"] == "7"
    assert values["LMChatVThreeHumanAliasMorphologyN"] == "8"
    assert values["LMChatVThreeHumanParaphraseN"] == "0"
    assert values["LMChatVThreeHumanSecondaryTripleN"] == "3"
    assert values["LMChatVThreeHumanThirdRaterN"] == "0"

    expected_conditions = {
        "Present": ("64", "66", "66.7", "68.8", "53.1", "79.2", "55.2", "82.3"),
        "Rebuild": ("13", "13", "13.5", "13.5", "3.1", "26.0", "3.1", "26.0"),
        "Policy": ("15", "15", "15.6", "15.6", "3.1", "28.1", "3.1", "28.1"),
        "Prompt": ("54", "55", "56.3", "57.3", "41.7", "70.8", "42.7", "71.9"),
    }
    for stem, expected in expected_conditions.items():
        prefix = f"LMChatVThree{stem}Leak"
        observed = (
            values[f"{prefix}LowerN"],
            values[f"{prefix}UpperN"],
            values[f"{prefix}LowerPct"],
            values[f"{prefix}UpperPct"],
            values[f"{prefix}LowerCILowerPct"],
            values[f"{prefix}LowerCIUpperPct"],
            values[f"{prefix}UpperCILowerPct"],
            values[f"{prefix}UpperCIUpperPct"],
        )
        assert observed == expected

    assert values["LMChatVThreePresentMatcherCleanN"] == "40"
    assert values["LMChatVThreePresentJudgeMissN"] == "11"
    assert values["LMChatVThreeRebuildMatcherCleanN"] == "83"
    assert values["LMChatVThreeRebuildJudgeMissN"] == "0"
    assert values["LMChatVThreePolicyMatcherCleanN"] == "82"
    assert values["LMChatVThreePolicyJudgeMissN"] == "1"
    assert values["LMChatVThreePromptMatcherCleanN"] == "48"
    assert values["LMChatVThreePromptJudgeMissN"] == "7"

    assert values["LMChatVThreeEditedMinusFreshLowerN"] == "2"
    assert values["LMChatVThreeEditedMinusFreshUpperN"] == "2"
    assert values["LMChatVThreeEditedMinusFreshLowerPoints"] == "2.1"
    assert values["LMChatVThreeEditedMinusFreshUpperPoints"] == "2.1"
    assert values["LMChatVThreeEditedMinusFreshLowerCILowerPoints"] == "0.0"
    assert values["LMChatVThreeEditedMinusFreshLowerCIUpperPoints"] == "5.2"
    assert values["LMChatVThreeEditedMinusFreshUpperCILowerPoints"] == "0.0"
    assert values["LMChatVThreeEditedMinusFreshUpperCIUpperPoints"] == "5.2"

    assert values["LMChatVThreePromptMinusEditedLowerN"] == "39"
    assert values["LMChatVThreePromptMinusEditedUpperN"] == "40"
    assert values["LMChatVThreePromptMinusEditedLowerPoints"] == "40.6"
    assert values["LMChatVThreePromptMinusEditedUpperPoints"] == "41.7"
    assert values["LMChatVThreePromptMinusEditedLowerCILowerPoints"] == "17.7"
    assert values["LMChatVThreePromptMinusEditedLowerCIUpperPoints"] == "62.5"
    assert values["LMChatVThreePromptMinusEditedUpperCILowerPoints"] == "18.8"
    assert values["LMChatVThreePromptMinusEditedUpperCIUpperPoints"] == "62.5"

    assert values["LMChatVThreeTargetPolicyRebuildExactN"] == "77"
    assert values["LMChatVThreeTargetPolicyRebuildNormalizedN"] == "87"
    assert values["LMChatVThreeRetainedPresentCorrectN"] == "46"
    assert values["LMChatVThreeRetainedRebuildCorrectN"] == "46"
    assert values["LMChatVThreeRetainedPolicyCorrectN"] == "44"
    assert values["LMChatVThreeRetainedPolicyPresentExactN"] == "74"
    assert values["LMChatVThreeRetainedPolicyPresentNormalizedN"] == "82"
    assert values["LMChatVThreeRetainedPolicyRebuildExactN"] == "75"
    assert values["LMChatVThreeRetainedPolicyRebuildNormalizedN"] == "77"
    assert values["LMChatVThreeSuffixContaminatedHistoryN"] == "24"
    assert values["LMChatVThreeSuffixContaminatedClusterN"] == "10"

    assert values["LMChatVThreeSummaryFileSHA"] == (
        "04e7292977e7a8762cbe8e076e9a1f15"
        "e8db679e7edf33a26fc604ddb14ac185"
    )
    assert values["LMChatVThreeSummaryPayloadSHA"] == (
        "29dace92767a0a274bd852b1fcf5acf3"
        "86285a837d94cf669e8bb7f8188c1fca"
    )
    assert values["LMChatVThreeSummaryIntegritySHA"] == (
        "780365f81f4ffa8603d65f7b520c0286"
        "313d51ea733468d9af9650bf11be9777"
    )
    assert values["LMChatVThreeCensusFileSHA"] == (
        "226c00beeba29e7849cb0612e8dc8f8a"
        "789616a0813919798aab4de845c7008a"
    )
    assert values["LMChatVThreeCensusPayloadSHA"] == (
        "004f4c0b97216ef2b0cb62c09a401465"
        "84a9f67c84c480d23214f512f3a58f5f"
    )
    assert values["LMChatVThreeCensusIntegritySHA"] == (
        "ed81e4a61c2693de82dfee30f6e2e422"
        "7d07976d53ede48f455ba87dd2d1b577"
    )
    assert values["LMChatVThreeAdjudicatedCensusFileSHA"] == (
        "a12f12f8d0012ce3ce171fcc87a6fe530"
        "fdac2c21654f8e100d2d7d7a4610355"
    )
    assert values["LMChatVThreeAdjudicatedCensusPayloadSHA"] == (
        "84ed9fe8bab07eabffa52858de9f58fee"
        "b214dd86db12da6f8f2b845c03f77a6"
    )
    assert values["LMChatVThreeAdjudicatedCensusIntegritySHA"] == (
        "007c241ac530eb7620ae0675ecd45ea1e"
        "c384d3a1294433a51ff8792ed893e56"
    )


def test_half_up_formatting_is_explicit():
    assert macros.format_percentage(0.5625) == "56.3"
    assert macros.format_percentage(-0.5625) == "-56.3"
    assert macros.format_points(0.1875) == "18.8"


def test_rendering_is_deterministic(rendered: str):
    second = macros.build_paper_macros(
        decoded_summary_path=SUMMARY,
        census_statistics_path=CENSUS,
        adjudicated_census_path=ADJUDICATED,
    )
    assert second == rendered
    assert second.encode("ascii") == rendered.encode("ascii")
    assert COMMITTED_MACROS.read_text(encoding="ascii") == rendered


def test_output_contains_only_source_free_values(
    rendered: str,
    values: dict[str, str],
):
    macros.assert_source_free_latex(rendered)
    assert rendered.startswith("% GENERATED FILE -- DO NOT EDIT.\n")
    assert '"' not in rendered
    assert "under my bed" not in rendered
    assert "Under the bed" not in rendered
    assert "sneakers" not in rendered
    assert len(values) == len(MACRO_RE.findall(rendered))
    assert all(SAFE_VALUE_RE.fullmatch(value) for value in values.values())


@pytest.mark.parametrize(
    ("source", "argument"),
    (
        (SUMMARY, "decoded_summary_path"),
        (CENSUS, "census_statistics_path"),
        (ADJUDICATED, "adjudicated_census_path"),
    ),
)
def test_exact_input_tampering_is_rejected(
    tmp_path: Path,
    source: Path,
    argument: str,
):
    tampered = tmp_path / source.name
    tampered.write_bytes(source.read_bytes() + b" ")
    paths = {
        "decoded_summary_path": SUMMARY,
        "census_statistics_path": CENSUS,
        "adjudicated_census_path": ADJUDICATED,
    }
    paths[argument] = tampered
    with pytest.raises(macros.PaperMacroError, match="binding differs"):
        macros.build_paper_macros(**paths)


def test_atomic_write_requires_explicit_overwrite(
    tmp_path: Path,
    rendered: str,
):
    output = tmp_path / "macros.tex"
    macros.write_macros(output, rendered)
    assert output.read_text(encoding="ascii") == rendered

    output.write_text("sentinel", encoding="ascii")
    with pytest.raises(FileExistsError, match="--overwrite"):
        macros.write_macros(output, rendered)
    assert output.read_text(encoding="ascii") == "sentinel"

    macros.write_macros(output, rendered, overwrite=True)
    assert output.read_text(encoding="ascii") == rendered


def test_validate_only_does_not_write(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    output = tmp_path / "must-not-exist.tex"
    assert (
        macros.main(
            [
                "--decoded-summary",
                str(SUMMARY),
                "--census-statistics",
                str(CENSUS),
                "--adjudicated-census",
                str(ADJUDICATED),
                "--output",
                str(output),
                "--validate-only",
            ]
        )
        == 0
    )
    assert not output.exists()
    assert "validated" in capsys.readouterr().out
