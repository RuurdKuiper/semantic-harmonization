# Why reference codes were unavailable

Audit date: 29 October 2026. This audit reproduces all eight rows of manuscript
Table 1: 3,945 Narrow reference entries, 3,051 available and 894 unavailable.
Counts below are phenotype-specific reference entries, as in Table 1, rather
than unique identifiers pooled across phenotypes. There are 893 distinct
unavailable code–terminology pairs because one code occurs in two phenotypes.

## Confirmed immediate causes

| Terminology | Reference entries | Evidence and interpretation |
|---|---:|---|
| MedDRA | 443 | The code occurs in the original CSV, but every corresponding description is empty. The loader omits entries without descriptions. |
| SNOMED CT | 415 | The code occurs in the original CSV, but every corresponding description is empty. The loader omits these entries. |
| SNOMED CT | 31 | The codes are absent from the US export. All have NHS Digital UK namespace `1000000` and occur in the kidney-disease reference. This is an edition/scope mismatch. |
| ICPC | 3 | `N94.01`, `U99.01` and `T90.01` are Dutch subcodes absent from the base `ICPC@1993` export. |
| ICPC | 1 | `T89` is an ICPC-2 insulin-dependent-diabetes code, absent from the supplied older ICPC export, which has `T90` as diabetes mellitus. |
| ICD-9-CM | 1 | The custom reference entry `260.61`, labelled “Diabetes type 1”, is absent from both the local export and the official CMS v32 diagnosis list. It needs reference-author adjudication. Do not assume a replacement. |
| **Total** | **894** | **858 (96.0%) have empty source descriptions; 36 are absent from the source exports.** |

Examples: MedDRA `10000920,` and SNOMED CT `194488006,` are literal blank-term
rows in the source files. No more permissive CSV parser or identifier
normalisation can recover text that is absent. No unavailable entry was found
to have a usable description discarded by parsing or range filtering.

The blank fields do **not** establish that all affected codes are inactive,
that all MedDRA entries have a particular hierarchy level, or that all are
eligible in the intended release. The two-column exports contain no metadata
to resolve these questions. An active-term or hierarchy-level export policy
could be involved, but that underlying reason remains unverified.

## Check of the colleague's version list

| Colleague's statement | What the evaluated local resources establish |
|---|---|
| ICPC: 1993 | Matches the local `ICPC@1993` identifier. Exact edition/package metadata is still needed; the references mix base, Dutch subcodes and an ICPC-2 code. |
| ICD-10-CM: 2025 | The source is labelled `ICD10CM@2026`. Independent comparison with the CDC order files confirms the complete FY2026 identifier set. FY2026 became effective on 1 October 2025, so a calendar-year reference could explain the discrepancy. It is not the FY2025 set. |
| ICD-9-CM: 2014 | Matches the local `ICD9CM@2014` identifier. The independent CMS v32 archive is effective 1 October 2014; it excludes `26061`. The date label alone does not prove the complete historical provenance of the local export. |
| MedDRA: 28.0 | Matches `MDR@28.0`; the official MedDRA 28.0 guide is dated March 2025. This does not establish completeness or hierarchy-level coverage of the export. |
| SNOMED: 2025-03-01 | Conflicts with the configured/export identifier `SNOMEDCT_US@2025_09_01`. NLM lists March and September 2025 as distinct US releases. The original release-package/export manifest is needed to verify which release the file contents represent; a filename cannot settle this independently. |

Read v2 (`RCD2@20200401`, a local export identifier) is also used by the pipeline,
although omitted from the colleague's list. It did not contribute to the
unavailable Narrow entries in Table 1. The local identifier should not be
described as an official 2020 Read v2 release date.

All six Parquet cache hashes match the frozen study manifest. Fresh parsing of
all six original CSVs also reproduces their caches exactly. The findings
therefore concern the actual evaluated inputs, rather than a different current
configuration or stale-cache mismatch.

The ICD-10-CM comparison is particularly strong: after excluding source grouping
ranges and one non-code document header, the local identifiers equal the official
98,186-entry FY2026 order-file set. The local source is missing 12 identifiers
from the FY2025 set and contains the FY2026 additions. The 2025 calendar date and
2026 fiscal-year name must be distinguished.

## Repair plan

1. Obtain independent, complete terminology exports, with descriptions,
   release/package metadata and relevant active/inactive and hierarchy-level
   information. For MedDRA, document which term levels are eligible; for SNOMED,
   document whether historical concepts are included. Restore missing descriptions
   only where those entries meet the agreed corpus scope.
2. Align the editions/national extensions of the reference and corpus. UK SNOMED
   codes should not be silently labelled as US concepts. Dutch ICPC subcodes and
   ICPC-2 entries likewise need the matching terminology resources or an explicit,
   expert-reviewed reference-scoping policy. Broader parent codes are not
   interchangeable with these more specific entries.
3. Have the reference author resolve `260.61`. The superficially similar `250.61`
   exists in CMS but denotes diabetes with neurological manifestations, so
   automatically substituting it would be unjustified.
4. Add corpus-completeness diagnostics and remove non-code document-header rows
   in a separate corrected corpus build. The current ICD-10-CM corpus contains
   one such entry: `ICD-10-CM`, described as “ICD-10-CM TABULAR LIST of DISEASES
   and INJURIES”. This is unrelated to the 894 missing entries but needs cleanup.
5. Version and hash the corrected corpus/reference set separately, rebuild the
   affected retrieval indexes and embeddings, and rerun configuration selection
   and evaluation. Retain the original results as results for the original inputs.
   Adding identifiers to the denominator without generating predictions for them
   would not be a valid corrected evaluation.

Do not fill the blank descriptions from the evaluation reference files. Those
files contain the phenotype-specific evaluation material and could leak both
reference membership and curated descriptions into the retrieval corpus. An
independent terminology source is required. The count of potentially affected
entries is known; their future recoverability and effect on recall/precision/F1
cannot be established without eligible independent descriptions and new runs.

No production corpus, reference file, embedding, prediction, configuration,
manuscript text or result value was modified by this audit.

## Files

- `unavailable_reference_entries_final.csv`: every missing entry, its original
  source file/line numbers, observed cause and appropriate repair.
- `counts_by_cause.csv` and `counts_by_phenotype.csv`: complete reconciliations.
- `frozen_corpus_verification.csv`: matches to the saved study input hashes.
- `source_cache_verification.json`: equality of fresh CSV parsing and saved caches.
- `audit_summary.json`: structured findings and unresolved version/status limits.
- `external/`: public CDC/CMS comparison archives and a download manifest.
- `audit_unavailable.py`: read-only reproduction script. Run from the repository
  root with `PYTHONPATH=. .venv-study/bin/python artifacts/reference_availability/20261029/audit_unavailable.py`.

## Primary evidence consulted

- [CDC ICD-10-CM release files and fiscal-year dates](https://www.cdc.gov/nchs/icd/icd-10-cm/files.html)
- [CDC FY2026 code descriptions](https://ftp.cdc.gov/pub/health_statistics/nchs/publications/ICD10CM/2026/)
- [CDC FY2025 comparison archive](https://ftp.cdc.gov/pub/health_statistics/nchs/publications/ICD10CM/2025/)
- [CMS ICD-9-CM full and abbreviated code titles](https://www.cms.gov/medicare/coding-billing/icd-10-codes/icd-9-cm-diagnosis-procedure-codes-abbreviated-and-full-code-titles)
- [NLM March/September 2025 SNOMED CT US release archive](https://www.nlm.nih.gov/healthit/snomedct/archive.html)
- [SNOMED International: namespace 1000000 owned by NHS Digital UK](https://docs.snomed.org/snomed-ct-specifications/snomed-ct-release-file-specification/reference-set-release-file-specification/5.2-reference-set-types/5.2.4-metadata-reference-sets/5.2.4.8-component-annotation-string-value-reference-set)
- [Nictiz ICPC-1-2000NL subcode list](https://decor.nictiz.nl/pub/ketenzorg/kz-html-20141013T173536/voc-2.16.840.1.113883.2.4.3.11.60.103.11.12-2011-10-12T000000.html)
- [NHG kidney disease registration advice: U99.01](https://richtlijnen.nhg.org/files/2020-09/NHG-registratie%20advies%20bij%20Standaard%20Chronische%20nierschade_2020.pdf)
- [Norwegian Directorate of Health: English ICPC-2, including T89](https://www.helsedirektoratet.no/digitalisering-og-e-helse/helsefaglige-kodeverk/icpc/icpc-2e--english-version/ICPC-2e-English.pdf?download=false)
- [Official MedDRA 28.0 introductory guide, March 2025](https://alt.meddra.org/files_acrobat/intguide_28_0_English.pdf)

Public SNOMED browser requests were unavailable (HTTP 405/423), so no individual
active-status or exact-release assertions were inferred from those attempts.
