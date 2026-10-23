"""Submitted identifiers, rather than descriptions or retrieval traces, define predictions."""
import pytest
import pandas as pd
from experiments.pipeline_design.published_baselines import extract_codes,format_aliases,normalize_codes


def test_markup_escaped_identifiers_unknown_codes_and_duplicates():
    text='''| Code | Description |
|---|---|
| `M11` | Eczema |
| **EMISICD10\\|F0100** | Dementia |
| UNKNOWN123 | Unknown code must remain in the prediction denominator |
| M11 | Duplicate |
'''
    codes,audit=extract_codes(text)
    assert codes==['EMISICD10|F0100','M11','UNKNOWN123']
    assert audit['duplicate_rows']==1


def test_explicit_code_column_and_excluded_notes():
    text='''| Code Source | Code | Description |
|---|---|---|
| Read | M11 | Eczema |

**Excluded codes**
| Code | Description |
|---|---|
| M12 | Not selected |
'''
    assert extract_codes(text)[0]==['M11']


def test_group_headings_are_not_identifiers_and_ranges_are_not_expanded():
    text='''| Code | Description |
|---|---|
| **Read / CTV3 Codes** | |
| M11* | A literal wildcard is not expanded |
| ^ESCT123 | A literal identifier |
'''
    codes,audit=extract_codes(text)
    assert codes==['M11*','^ESCT123']
    assert audit['ignored_groups']==['Read / CTV3 Codes']


def test_empty_submission_is_distinct_from_an_unrecognized_answer():
    assert extract_codes('')[1]['empty_submission']
    with pytest.raises(ValueError):extract_codes('Codes are M11 and M12.')


def test_compound_cells_separate_predictions_from_annotations_without_identifier_repair():
    text='''| Code | Description |
|---|---|
| **BADCODE** (via ^ESCT123) | Keep the submitted code; do not rescue from its citation |
| **^ESCT1** / **ESCT2** | Explicit multiple codes; missing prefix remains missing |
| **^ESCT3** (score, excluded) | Not a prediction |
'''
    codes,_=extract_codes(text)
    assert codes==['BADCODE','ESCT2','^ESCT1']


def test_verified_source_record_pairs_do_not_count_retrieval_ids_as_predictions():
    text='| Code | Description |\n|---|---|\n| 123456 / M11 | Eczema |'
    assert extract_codes(text,source_id_pairs=True)[0]==['M11']
    assert extract_codes(text)[0]==['123456','M11']


def test_format_normalization_accepts_verified_carets_and_read_padding_and_deduplicates():
    full=pd.DataFrame({'code':['^ESCT123','M11','M11z','M111-98']})
    read=full.iloc[1:]
    codes,audit=normalize_codes(['ESCT123','^ESCT123','M11..','M11z.','M111-98'],set(full.code),format_aliases(full,read))
    assert codes==['M11','M111-98','M11z','^ESCT123']
    assert len(audit)==3


def test_normalization_preserves_case_spelling_unknowns_ranges_and_native_identifiers():
    full=pd.DataFrame({'code':['^ESCT123','M11z','M11']})
    submitted=['esct123','M11Z.','M11*','M11-M12','123456789','UNKNOWN']
    codes,audit=normalize_codes(submitted,set(full.code),format_aliases(full,full.iloc[1:]))
    assert codes==sorted(submitted)
    assert not audit


def test_alias_resolution_preserves_scope_and_full_browser_ambiguity():
    full=pd.DataFrame({'code':['^ESCT123','M11','M11.']})
    aliases=format_aliases(full,full.iloc[1:])
    codes,audit=normalize_codes(['ESCT123','M11..','M11.'],{'M11'},aliases)
    assert codes==['ESCT123','M11.','M11..']
    assert not audit
    assert normalize_codes(['M11.'],set(full.code),aliases)==(['M11.'],[])
