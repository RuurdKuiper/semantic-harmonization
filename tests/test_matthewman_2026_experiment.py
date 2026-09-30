from pathlib import Path

import pandas as pd

from experiments.matthewman_2026.experiment import (
    load_cprd_browser,
    scope_reference_to_corpus,
    score_whole_codelist,
)


def test_load_cprd_browser_builds_full_and_read_subsets(tmp_path: Path):
    source = tmp_path / "browser.csv"
    pd.DataFrame(
        [
            {"OriginalReadCode": "A1", "Term": "Alpha", "CleansedReadCode": "A1"},
            {"OriginalReadCode": "B2", "Term": "Beta", "CleansedReadCode": ""},
            {"OriginalReadCode": "A1", "Term": "Alpha duplicate", "CleansedReadCode": "A1"},
        ]
    ).to_csv(source, index=False)

    full, read = load_cprd_browser(source)

    assert full["code"].tolist() == ["A1", "B2"]
    assert read["code"].tolist() == ["A1"]
    assert set(full.columns) == {"code", "description", "vocabulary"}


def test_scope_reference_to_selected_corpus():
    reference = pd.DataFrame(
        [
            {"code": "A1", "required": 1},
            {"code": "B2", "required": 0},
        ]
    )
    corpus = pd.DataFrame([{"code": "A1"}])
    assert scope_reference_to_corpus(reference, corpus)["code"].tolist() == ["A1"]


def test_whole_codelist_grades_match_paper_rule():
    reference = pd.DataFrame(
        [
            {"code": "R1", "required": 1},
            {"code": "O1", "required": 0},
        ]
    )

    correct = score_whole_codelist(reference, ["R1", "O1"], ["R1", "O1"])
    partial = score_whole_codelist(reference, ["R1", "X1"], ["R1", "X1"])
    incorrect = score_whole_codelist(reference, ["R1"], [])

    assert (correct.grade, correct.grade_value) == ("C", 1.0)
    assert (partial.grade, partial.grade_value) == ("P", 0.5)
    assert (incorrect.grade, incorrect.grade_value) == ("I", 0.0)
    assert incorrect.retrieved_but_excluded == ["R1"]
    assert incorrect.required_not_retrieved_count == 0
    assert incorrect.retrieved_but_excluded_count == 1
    assert partial.irrelevant_predicted_count == 1
    assert partial.hallucinated_count == 0
