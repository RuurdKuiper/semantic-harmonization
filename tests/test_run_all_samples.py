"""Tests for the all-samples batch runner."""

from __future__ import annotations

from scripts.run_all_samples import _console_progress, _ground_truth_vocabularies
from src.utils.config import load_config


def test_ground_truth_vocabulary_selection():
    config = load_config("configs/default.yaml")
    expected_five = ["ICD10CM", "ICD9CM", "ICPC", "MDR", "SNOMEDCT_US"]
    for phenotype in [
        "acute_disseminated_encephalomyelitis",
        "acute_myocardial_infarction",
        "guillain_barre",
        "kidney_disease",
        "myocarditis",
        "type_1_diabetes",
    ]:
        assert _ground_truth_vocabularies(config, phenotype) == expected_five

    expected_four = ["ICD10CM", "ICD9CM", "MDR", "SNOMEDCT_US"]
    assert _ground_truth_vocabularies(config, "eczema_vaccinatum") == expected_four
    assert _ground_truth_vocabularies(config, "erythema_multiforme") == expected_four


def test_console_progress_displays_stage_and_bar(capsys):
    report = _console_progress("[sample]")
    report("Step 1/8: Loading EDF", None, None)
    report("Step 6/8: LLM classification", 10, 20)
    output = capsys.readouterr().out
    assert "Step 1/8" in output
    assert "10/20 (50%)" in output
