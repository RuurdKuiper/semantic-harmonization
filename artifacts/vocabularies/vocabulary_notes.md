# Slide: Six vocabularies, different purposes and scales

Suggested spoken takeaway: **The same phenotype must be found across vocabularies built for different jobs—from primary-care encounters to medical-product safety and detailed clinical records.**

`vocabulary_comparison.png` can be inserted directly into a slide. The SVG preserves editable text and vector bars; the PDF is suitable for export. All are 16:9.

## What the sizes mean

The chart reports **distinct codes retained in this project's retrieval corpus**, calculated directly from the six local CSV exports using `src/data/code_systems.py` and `preprocess_corpus` in `src/data/preprocessing.py`. Counts exclude entries without descriptions, deduplicate codes within a vocabulary, and remove rows recognized as code ranges. These are not official published terminology sizes, counts of synonyms, or counts of billable codes. Some parent/grouping codes remain.

| Vocabulary | Local source identifier | Retained codes |
|---|---|---:|
| SNOMED CT US | 2025_09_01 | 384,620 |
| ICD-10-CM | 2026 | 98,187 |
| Read v2 (RCD2) | 20200401 | 84,888 |
| MedDRA (MDR) | 28.0 | 28,889 |
| ICD-9-CM | 2014 | 22,203 |
| ICPC | 1993 | 740 |
| **Total** | | **619,527** |

SNOMED CT accounts for **62.1%** of this local corpus and has about **520 times** as many retained codes as ICPC. These are calculations from the local counts, not comparisons of official worldwide terminology sizes. Size does not measure quality or suitability.

## Distinctions and tidbits for speaker notes

- **SNOMED CT US — detailed clinical meaning.** Concepts have descriptions and explicit relationships. Its hierarchy allows multiple parents: a concept can be classified in more than one clinically relevant way. This makes it richer than a flat list of labels. The chart uses the project's US export, not a count of active International Edition concepts. [SNOMED International: logical model](https://docs.snomed.org/snomed-ct-practical-guides/snomed-ct-starter-guide/5-snomed-ct-logical-model), [concepts, descriptions and relationships](https://www.snomed.org/what-is-snomed-ct).

- **ICD-10-CM — US diagnosis classification.** This is the US clinical modification of ICD-10, with more diagnostic detail than ICD-9-CM. A useful slide distinction: **CM codes diagnoses; ICD-10-PCS is the separate procedure system.** The local count includes retained parent categories and should not be presented as the number of billable diagnoses. [CDC: ICD-10-CM](https://www.cdc.gov/nchs/icd/icd-10-cm/index.html), [CDC: comparison with ICD-9-CM](https://www.cdc.gov/nchs/hus/sources-definitions/icd-cm.htm).

- **Read v2 / RCD2 — legacy UK clinical records.** Read codes captured clinical findings and procedures across primary and secondary care. The surprising date distinction: the project's filename says **20200401**, but NHS England identifies **April 2016** as the last updated Read v2 release. Describe 2020 as the local source identifier, not a new terminology release. NHS guidance directs clinical content toward SNOMED CT. [NHS England: Read Codes](https://digital.nhs.uk/services/terminology-and-classifications/read-codes).

- **MedDRA / MDR — medical-product safety and regulatory reporting.** Its five levels run from Lowest Level Terms through Preferred Terms, High Level Terms and High Level Group Terms to System Organ Classes. Some Preferred Terms connect to multiple organ classes. A particularly relevant tidbit for phenotype retrieval: **Standardised MedDRA Queries (SMQs) are predefined tools for retrieving safety-related cases.** The local two-column export does not identify hierarchy levels, so do not call its 28,889 codes a verified count of all Preferred Terms or all MedDRA terms. [MedDRA: hierarchy](https://www.meddra.org/how-to-use/basics/hierarchy), [multiaxiality](https://www.meddra.org/multiaxiality), [SMQs](https://www.meddra.org/standardised-meddra-queries), [regulatory and safety use](https://www.meddra.org/how-to-use/support-documentation/english/welcome).

- **ICD-9-CM — historical US diagnoses and procedures.** It is still relevant when harmonizing older records: the US transition to ICD-10-CM/PCS occurred on **1 October 2015**. Unlike ICD-10-CM alone, ICD-9-CM includes a procedure classification; this project's file visibly contains procedure entries as well as diagnoses. Its count therefore is not a diagnosis-only denominator for comparing ICD revisions. [CDC: history and transition](https://www.cdc.gov/nchs/hus/sources-definitions/icd-cm.htm), [CMS: diagnosis and procedure code titles](https://www.cms.gov/medicare/coding-billing/icd-10-codes/icd-9-cm-diagnosis-procedure-codes-abbreviated-and-full-code-titles).

- **ICPC — the primary-care encounter.** ICPC is designed around what brings a patient to primary care, including symptoms and complaints, the diagnosis managed, and processes of care. It can describe an encounter before a definitive diagnosis exists. WHO's ICPC-2 description illustrates this purpose and its 17-chapter / 7-component structure. **Version caveat:** this project's file is labeled ICPC@1993; do not relabel it ICPC-2 or ICPC-3 or assign it their official code counts. [WHO: ICPC-2 purpose and structure](https://www.who.int/standards/classifications/other-classifications/international-classification-of-primary-care).

Sources checked online on 16 October 2026. Terminology characteristics above describe their purposes; the chart's numerical sizes come exclusively from the local project data.

## Regeneration

From the repository root:

```sh
.venv-study/bin/python artifacts/vocabularies/make_diagram.py
```
