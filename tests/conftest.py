"""Shared pytest fixtures for the semantic-harmonization test suite."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sample_codes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"code": "I40.0", "description": "Infective myocarditis", "vocabulary": "ICD10"},
            {"code": "I40.9", "description": "Acute myocarditis unspecified", "vocabulary": "ICD10"},
            {"code": "I30.9", "description": "Acute pericarditis unspecified", "vocabulary": "ICD10"},
            {"code": "I25.10", "description": "Atherosclerotic heart disease", "vocabulary": "ICD10CM"},
            {"code": "50920009", "description": "Myocarditis", "vocabulary": "SNOMED"},
        ]
    )


@pytest.fixture
def sample_gold() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"code": "I40.0", "vocabulary": "ICD10", "label": "Narrow"},
            {"code": "I40.9", "vocabulary": "ICD10", "label": "Narrow"},
            {"code": "I30.9", "vocabulary": "ICD10", "label": "Exclude"},
            {"code": "50920009", "vocabulary": "SNOMED", "label": "Narrow"},
        ]
    )
