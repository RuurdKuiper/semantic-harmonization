# Blank descriptions across the full terminology exports

29 October 2026. Counts are distinct identifiers with **no populated description
in any row** of their source file. Repeated term rows are not double-counted.

| Terminology/export | Distinct identifiers | Without any description | Percentage |
|---|---:|---:|---:|
| MedDRA 28.0 | 92,107 | 63,218 | 68.64% |
| SNOMED CT US, labelled September 2025 | 532,287 | 147,667 | 27.74% |
| Read v2 local export | 89,676 | 60 | 0.07% |
| ICPC 1993 local export | 747 | 7 | 0.94% |
| ICD-10-CM FY2026 local export | 98,506 | 0 | 0% |
| ICD-9-CM 2014 local export | 22,406 | 0 | 0% |

These are source-export identifiers, not official counts of clinically codable
terms. For example, the seven ICPC identifiers are the structural component IDs
`1`–`7`. Read counts preserve identifier case; preprocessing in the frozen
pipeline is a separate operation. None of the six exports contains a blank
description for a code that also has a populated description elsewhere in that
same file. The gaps therefore cannot be dismissed as redundant empty synonym
rows where another row already supplies the label.

## An independent label exists upstream for a blank CSV entry

The local MedDRA row is `10064220,`. An earlier saved CodeMapper API response
contains this same-release source object:

```json
{
  "codingSystem": "MDR@28.0",
  "id": "10064220",
  "preferredTerm": "Eosinophilic esophagitis",
  "cui": "C0341106"
}
```

Evidence: `results/codemapper/eosinophilic_esophagitis.json`, also documented in
`experiments/codemapper_probe.md` from 6 October 2026. This wording was obtained
through a terminology API, not copied from an evaluation reference. It shows
that a CSV blank does not universally mean the terminology service lacks a term.
It does not establish that every missing label can be recovered or that every
such identifier is eligible in the intended corpus.

## A strong MedDRA export-pattern clue, still an inference

The official [MedDRA 28.0 What's New](https://alt.meddra.org/files_acrobat/whatsnew_28_0_English.pdf),
Tables 4-1–4-3, reports 89,774 Lowest Level Terms (LLTs, including PTs), 26,920
Preferred Terms (PTs), 1,739 High Level Terms (HLTs), 337 High Level Group Terms
(HLGTs), 27 System Organ Classes (SOCs), and 230 Standardised MedDRA Queries
(SMQs). These counts reproduce our export totals exactly:

- PT + HLT + SMQ = 26,920 + 1,739 + 230 = **28,889 populated identifiers**.
- LLT-only + HLGT + SOC = (89,774 − 26,920) + 337 + 27 = **63,218 blank identifiers**.
- Their combined total is **92,107 distinct identifiers**, matching the CSV.

This is strong evidence to investigate a label export restricted by hierarchy
level: the code list may include all levels while the label query covers only
certain levels. The hypothesis has not been verified against every code's term
type or the export query, so it must not be reported as a proven per-code cause.
Do not describe the labelled subset as “all Preferred Terms”: its count includes
more entries than the official PT count.

## Other plausible export mechanisms

For SNOMED, one possibility is that all concept identifiers were exported while
descriptions were joined only for active/current descriptions, a language subset,
or a particular release. A wrong or overly restrictive join could have the same
visible effect. The source has no active-status, language or release-matching
fields with which to distinguish these explanations. The blank count alone does
not prove inactivity or erroneous source codes.

The observed fact is that blanks are already present in the received CSVs, before
our loader runs. The immediate explanation is an export/label-coverage gap. The
exact generating filter or join remains to be confirmed with the source provider.

A useful question for the provider is: **Does the code export include every term
type/status, while the term field is populated only for selected hierarchy levels,
preferred terms, active descriptions, languages or releases? Can the export include
term type, status, language and exact release metadata?**

Reproduce with `PYTHONPATH=. .venv-study/bin/python artifacts/reference_availability/20261029/audit_full_export_descriptions.py`.
The accompanying JSON files retain source hashes, complete counts, the MedDRA
count comparison and the independent cached-API example. No pipeline inputs or
reported results have been modified.
