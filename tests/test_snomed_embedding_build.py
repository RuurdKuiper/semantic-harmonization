"""Safe reuse in the resumable SNOMED embedding build."""
import pandas as pd

from scripts.build_snomed_embeddings import matching_positions


def test_reuse_requires_exact_wording_code_and_vocabulary():
    previous = pd.DataFrame([
        ('1', 'SNOMEDCT_US', 'Original term'),
        ('2', 'SNOMEDCT_US', 'Unchanged term'),
        ('3', 'OTHER', 'Same words'),
    ], columns=['code', 'vocabulary', 'description'])
    current = pd.DataFrame([
        ('2', 'SNOMEDCT_US', 'Unchanged term'),
        ('1', 'SNOMEDCT_US', 'Changed term'),
        ('3', 'SNOMEDCT_US', 'Same words'),
        ('4', 'SNOMEDCT_US', 'Unchanged term'),
    ], columns=previous.columns)

    assert matching_positions(current, previous).tolist() == [1, -1, -1, -1]
