# Recovered provenance of the terminology CSV files

Recovered on 29 October 2026 from local filesystem metadata, Chrome download
records, macOS download/quarantine records and file hashes. No remote account
was accessed and no source file or browser database was modified. Session and
sharing tokens have been omitted from this record.

## Five CSVs: VAC4EU SharePoint download batch

The following five working CSVs carry the same macOS quarantine/download event
identifier, `D5978A04-B938-44A3-AEC8-8DC7444742D3`:

- `ICD10CM@2026-codes.csv`
- `ICD9CM@2014-codes.csv`
- `ICPC@1993-codes.csv`
- `MDR@28.0-codes.csv`
- `SNOMEDCT_US@2025_09_01-codes.csv`

The matching macOS event timestamp is 17 September 2026, 09:15:50 Europe/Amsterdam.
Chrome's download record 1751 at that exact timestamp identifies the downloaded
file as `OneDrive_1_17-09-2026.zip`. The originating browser tab was the following
VAC4EU SharePoint folder:

**VAC4EUSecretariat → Shared Documents → 21. Strategy → Pillar 2 → AI roadmap →
AI for semantic harmonisation → AI proposal for tagging → Dictionaries for testing**

[Open the original folder](https://vac4eu.sharepoint.com/sites/VAC4EUSecretariat/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FVAC4EUSecretariat%2FShared%20Documents%2F21.%20Strategy%2FPillar%202%2FAI%20roadmap%2FAI%20for%20semantic%20harmonisation%2FAI%20proposal%20for%20tagging%2FDictionaries%20for%20testing)

The ZIP itself is no longer present at its recorded Downloads path, so an
archive-member hash comparison was not possible. The shared event identifier
and exact matching download timestamp strongly tie the five files to that batch.

There is also independent direct-download evidence for ICD-9-CM: the copy still
present as `/Users/rkuiper/Downloads/ICD9CM@2014-codes.csv` is byte-identical
(SHA-256 match) to the working corpus source. Its WhereFroms metadata and Chrome
download record 1748 identify the same VAC4EU folder, downloaded on 16 September
2026 at 11:04:40 Europe/Amsterdam.

## Read v2: an earlier Excel Online export

The working `RCD2@20200401-codes.csv` retains its own WhereFroms metadata. It was
downloaded through Excel Online's `XlFileHandler.aspx` CSV-export endpoint, using
worksheet `in`. The underlying document was hosted in a UMC Utrecht personal
SharePoint area:

- Host: `umcutrecht-my.sharepoint.com`
- Account path: `personal/s_mohammadi-3_umcutrecht_nl`
- Chrome download record 1459: `RCD2@20200401-codes(in).csv`, 20 July 2026,
  08:55:09 Europe/Amsterdam, matching the working file's retained download metadata.

This identifies the shared-document account, not necessarily the person who
originally generated the terminology data. No full personal name was inferred
from the account identifier.

Earlier `ICD10CM@2026-codes(in).csv` and `ICPC@1993-codes(in).csv` files in Downloads
also retain July 20 Excel Online export metadata pointing to that same UMC Utrecht
account. They are earlier copies and were not substituted for the five September
working inputs.

## CodeMapper catalog: separate evidence, not the original download provenance

`/Users/rkuiper/Downloads/vocabularies.csv` was downloaded on 29 October 2026 from
`https://app.vac4eu.org/codemapper-testing/api/vocabularies/?format=csv`.
It is a terminology catalog and is distinct from the full code-description CSVs.
Its existence corroborates the availability of the terminology service but does
not establish which export query produced the September files.

## What is established, and what remains unknown

The local evidence recovers the delivery locations and download batches. It does
not identify the original database extraction query, language/status/term-level
filters, package manifests or person responsible for the export. The VAC4EU
SharePoint folder and the earlier UMC Utrecht shared-document account are the
concrete places to return to for corrected exports and that missing metadata.

The working files first appear in Git commit `c8efca4` on 30 September 2026.
That commit records their addition, not their original generation or download.
