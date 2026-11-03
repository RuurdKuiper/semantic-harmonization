# Validation of the new SNOMED CT US export

29 October 2026. Input: `data/codes/csv/SNOMEDCT_US@2025_09_01-codes new.csv`.
This comparison is limited to SNOMED CT US. MedDRA has not been altered.

## Import works, but reference coverage is not repaired

| Check | Existing source/corpus | New export |
|---|---:|---:|
| Identifiers with at least one description | 384,620 | 382,170 |
| Identifiers present only as blank-description rows | 147,667 | 0 |
| New identifiers added relative to the searchable corpus | — | 0 |
| Previously searchable identifiers removed | — | 2,450 |
| Unavailable SNOMED Narrow reference entries, across eight phenotypes | 446 | 447 |
| Previously unavailable reference entries recovered | — | 0 |

Every code identifier in the new export already had wording in the old export.
The former blank-description codes are absent from the new file; their
descriptions have not been filled in. This explains how the new file can look
complete when opened while still failing to recover the missing reference codes.
No conclusion about active/inactive status or exact release changes is inferred
without source metadata.

One previously available Narrow reference entry is also absent:

- Phenotype: kidney disease.
- Code: `295121000119101`.
- Reference description: “Nephrosis co-occurrent and due to systemic lupus
  erythematosus”.

No other phenotype loses a previously available SNOMED Narrow reference entry.
The per-phenotype counts are in `reference_coverage.csv`. They are corpus
availability counts; no new model performance scores have been computed.

## Formatting issue handled

The new file is an Excel-style semicolon export whose first column contains an
original comma-delimited record. Its header is `code,term;;;`. Quoted descriptions
can therefore have two layers of CSV quoting. In 99 source rows, genuine
semicolons inside descriptions also occupy multiple outer columns.

The SNOMED loader now detects this format, decodes both layers, reconstructs
internal semicolons, preserves embedded quotes/commas and rejects malformed inner
records rather than dropping them. The normal-format loading path is unchanged.
The full old-file import was checked against the frozen Parquet cache and is
identical. Loader tests cover the new quoting/punctuation and malformed-record
cases; the complete loader test file passed (12 tests).

All 788,036 new source records decode into a code and nonempty description.
Keeping the first description per code, as in the existing pipeline, yields
382,170 entries. The imported corpus is saved separately:

- `SNOMEDCT_US_new_imported.csv`.
- `parquet/codes_SNOMEDCT_US.parquet`.

The import selected a different first description for 199,267 retained codes,
including cases where the fully specified name appears before another description
in the new file. A future run using these selected descriptions needs an isolated
or rebuilt embedding index; changing file formatting is not evidence that the
existing embeddings represent the new selected text.

## Decision and next step

The active corpus and frozen study settings were **not switched** to this export:
it is importable, but supplies no missing wording and loses some coverage.
Original CSVs, caches, configurations, reference lists and manuscript PDFs are
preserved. No embedding or model calls were made.

To repair the original gap, request an export that actually includes descriptions
for previously blank identifiers (for example `194488006` or `192934005`), together
with status and edition metadata. It remains necessary to distinguish a missing
description from an intentionally excluded historical or out-of-edition code.
The 31 UK-namespace codes absent from the US export are a separate edition issue.

Reproduce the read-only comparison and isolated import from the repository root:

```sh
PYTHONPATH=. .venv-study/bin/python artifacts/reference_availability/20261029/snomed_new_export/compare_export.py
```

`validation_summary.json` retains hashes, import counts, coverage results and
preservation checks. Supporting CSVs list previously missing entries, removed
codes, changed first descriptions and the newly unavailable reference entry.
