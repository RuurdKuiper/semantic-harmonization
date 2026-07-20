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


def _load_codelist_csv(
    path: str | Path,
    *,
    vocabulary: str,
    delimiter: str = ",",
) -> pd.DataFrame:
    """Load a code list CSV into the canonical retrieval schema.

    The source files in ``data/codes`` are plain text CSV exports, but the
    description field may contain commas wrapped in quotes. We therefore rely
    on the CSV parser rather than manual splitting so those commas stay inside
    the description instead of being treated as extra columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Full {vocabulary} code list not found: {path}")

    raw = pd.read_csv(
        path,
        dtype=str,
        sep=delimiter,
        engine="python",
        quotechar='"',
        escapechar="\\",
        on_bad_lines="skip",
    )

    if raw.empty:
        raise ValueError(f"No rows loaded from {path}")

    # Normalize the first two columns into code/description regardless of the
    # exact header names used in the export.
    columns = list(raw.columns)
    if len(columns) < 2:
        raise ValueError(f"Unexpected {vocabulary} CSV schema in {path}: columns={columns}")

    code_col = columns[0]
    desc_col = columns[1]
    df = raw[[code_col, desc_col]].rename(columns={code_col: "code", desc_col: "description"})
    df = df.dropna(subset=["code", "description"]).copy()
    df["code"] = df["code"].astype(str).str.strip().str.strip('"')
    df["description"] = df["description"].astype(str).str.strip().str.strip('"')
    df["vocabulary"] = vocabulary
    df = df[df["code"] != ""]
    df = df[df["description"] != ""]
    df = df.drop_duplicates(subset=["code", "vocabulary"], keep="first").reset_index(drop=True)
    return df


def _load_two_column_text(
    path: str | Path,
    *,
    vocabulary: str,
    delimiter: str,
) -> pd.DataFrame:
    """Load a very loose delimited export by splitting only on the first separator.

    Some vendor exports include trailing separators or extra empty columns. This
    parser keeps the first field as the code and the remainder of the line as the
    description so embedded punctuation in the description is preserved.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Full {vocabulary} code list not found: {path}")

    rows: list[tuple[str, str]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.lower().startswith("code"):
            continue
        if delimiter not in line:
            continue
        code, description = line.split(delimiter, 1)
        code = code.strip().strip('"')
        description = description.strip().strip(";").strip(delimiter).strip().strip('"')
        if code and description:
            rows.append((code, description))

    if not rows:
        raise ValueError(f"No rows loaded from {path}")

    df = pd.DataFrame(rows, columns=["code", "description"])
    df["vocabulary"] = vocabulary
    df = df.drop_duplicates(subset=["code", "vocabulary"], keep="first").reset_index(drop=True)
    return df


def load_icd10cm_full(csv_path: str | Path = "data/codes/ICD10CM@2026-codes.csv") -> pd.DataFrame:
    return _load_codelist_csv(csv_path, vocabulary="ICD10CM", delimiter=",")


def load_icpc_full(csv_path: str | Path = "data/codes/ICPC@1993-codes.csv") -> pd.DataFrame:
    return _load_codelist_csv(csv_path, vocabulary="ICPC", delimiter=",")


def load_rcd2_full(csv_path: str | Path = "data/codes/RCD2@20200401-codes.csv") -> pd.DataFrame:
    return _load_two_column_text(csv_path, vocabulary="RCD2", delimiter=",")


def load_snomedct_full(csv_path: str | Path = "data/codes/SNOMEDCT_US@2025_09_01-codes.csv") -> pd.DataFrame:
    return _load_two_column_text(csv_path, vocabulary="SNOMEDCT_US", delimiter=",")


CODE_SYSTEM_LOADERS = {
    "ICD10CM": load_icd10cm_full,
    "ICPC": load_icpc_full,
    "RCD2": load_rcd2_full,
    "SNOMEDCT_US": load_snomedct_full,
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
