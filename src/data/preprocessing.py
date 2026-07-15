from __future__ import annotations

import re
import unicodedata

import pandas as pd

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize_text(text: str) -> str:
    """Lowercase, strip accents/punctuation, and collapse whitespace for
    lexical matching and deduplication."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def standardize_code(code: str, vocabulary: str) -> str:
    """Standardize a code's surface form per vocabulary conventions
    (uppercase, no internal whitespace; ICD-style codes keep their dot)."""
    if not isinstance(code, str):
        return ""
    code = code.strip().upper()
    code = re.sub(r"\s+", "", code)
    vocabulary = (vocabulary or "").strip().upper()
    if vocabulary in {"ICD10", "ICD10CM"} and "." not in code and len(code) > 3:
        code = f"{code[:3]}.{code[3:]}"
    return code


def deduplicate_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate (code, vocabulary) pairs, keeping the first occurrence."""
    if df.empty:
        return df
    working = df.copy()
    working["_code_key"] = working.apply(
        lambda r: standardize_code(r["code"], r.get("vocabulary", "")), axis=1
    )
    working = working.drop_duplicates(subset=["_code_key", "vocabulary"], keep="first")
    return working.drop(columns=["_code_key"]).reset_index(drop=True)


def preprocess_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize codes, normalize descriptions, and drop duplicates from a
    raw code corpus DataFrame."""
    df = deduplicate_codes(df)
    df = df.copy()
    df["code"] = [standardize_code(c, v) for c, v in zip(df["code"], df["vocabulary"])]
    df["normalized_description"] = df["description"].map(normalize_text)
    return df
