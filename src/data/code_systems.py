"""Loaders for full clinical coding-system reference lists (ICD10, SNOMED CT,
MedDRA, ...), used as the retrieval universe for concept retrieval.

Unlike the phenotype-specific AESI review exports (`load_aesi_dataset`), these
represent the *complete* set of codes in a vocabulary, independent of any
phenotype. Since they can be large (tens of thousands of rows) and slow to
load/parse from their native format, the processed corpus for each vocabulary
is cached to disk as Parquet so the source file only needs to be parsed once.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_icd10_full(xlsx_path: str | Path = "data/codes/icd10.xlsx") -> pd.DataFrame:
    """Load the full ICD-10-CM code list from the CMS-style reference workbook.

    Parameters
    ----------
    xlsx_path : str or Path
        Path to the ICD-10 reference workbook. Expects a "Valid ICD10 ..."
        sheet with ``CODE`` and a ``LONG DESCRIPTION (...)`` column.

    Returns
    -------
    pd.DataFrame
        Columns: ``code``, ``description``, ``vocabulary`` (``"ICD10CM"``).
    """
    path = Path(xlsx_path)
    if not path.exists():
        raise FileNotFoundError(f"Full ICD10 code list not found: {path}")

    xls = pd.ExcelFile(path)
    sheet_name = next((s for s in xls.sheet_names if s.lower().startswith("valid icd10")), xls.sheet_names[0])
    raw = xls.parse(sheet_name, dtype=str)

    desc_col = next((c for c in raw.columns if c.upper().startswith("LONG DESCRIPTION")), None)
    if desc_col is None or "CODE" not in raw.columns:
        raise ValueError(
            f"Unexpected ICD10 workbook schema in sheet '{sheet_name}': columns={list(raw.columns)}"
        )

    df = raw[["CODE", desc_col]].rename(columns={"CODE": "code", desc_col: "description"})
    df = df.dropna(subset=["code", "description"]).copy()
    df["code"] = df["code"].str.strip()
    df["description"] = df["description"].str.strip()
    df["vocabulary"] = "ICD10CM"
    df = df.drop_duplicates(subset=["code", "vocabulary"], keep="first").reset_index(drop=True)
    return df


# Registry of loaders for each supported coding system. SNOMED CT (US) and
# MedDRA are planned but not yet wired up — a source file/loader needs to be
# added here before those vocabularies can be used for retrieval.
CODE_SYSTEM_LOADERS = {
    "ICD10CM": load_icd10_full,
}


def load_code_system_corpus(
    vocabularies: list[str],
    source_paths: dict[str, str] | None = None,
    cache_dir: str | Path = "data/processed",
) -> pd.DataFrame:
    """Load (and cache) the full reference code corpus for the given vocabularies.

    Each vocabulary is parsed from its source file at most once; the processed
    result (``code``, ``description``, ``vocabulary``) is cached to
    ``<cache_dir>/codes_<vocabulary>.parquet`` and reused on subsequent calls,
    so the (potentially slow) source loader only runs once per vocabulary.

    Parameters
    ----------
    vocabularies : list[str]
        Coding systems to include, e.g. ``["ICD10CM"]``.
    source_paths : dict[str, str], optional
        Override source file path per vocabulary (defaults to each loader's
        built-in default path).
    cache_dir : str or Path
        Directory to store/read the cached per-vocabulary Parquet files.

    Returns
    -------
    pd.DataFrame
        Concatenated corpus across all requested vocabularies with columns
        ``code``, ``description``, ``vocabulary``.
    """
    if not vocabularies:
        raise ValueError("At least one vocabulary must be specified.")

    source_paths = source_paths or {}
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for vocab in vocabularies:
        if vocab not in CODE_SYSTEM_LOADERS:
            raise ValueError(
                f"No loader registered for coding system '{vocab}'. "
                f"Available: {sorted(CODE_SYSTEM_LOADERS)}"
            )

        cache_path = cache_dir / f"codes_{vocab}.parquet"
        if cache_path.exists():
            frames.append(pd.read_parquet(cache_path))
            continue

        loader = CODE_SYSTEM_LOADERS[vocab]
        df = loader(source_paths[vocab]) if vocab in source_paths else loader()
        df.to_parquet(cache_path, index=False)
        frames.append(df)

    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
