"""Probe direct CodeMapper mappings without changing pipeline results.

Credentials come from the environment or the repository's ignored .env file.
Session cookies stay in memory. Reports omit login/session responses.
"""

from __future__ import annotations

import argparse
from collections import Counter
from http.cookiejar import CookieJar
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.mapping.codemapper import normalize_code_versions,normalize_concept_versions
BASE_URL = "https://app.vac4eu.org/codemapper-testing/api"
TARGETS = [
    "ICD10CM@2026", "ICD9CM@2014", "ICPC@1993", "MDR@28.0",
    "RCD2@20200401", "SNOMEDCT_US@2025_09_01",
]


class ProbeError(Exception):
    """A sanitized error that can be shown without exposing credentials."""


class Client:
    def __init__(self, timeout: float):
        self.opener = build_opener(HTTPCookieProcessor(CookieJar()))
        self.timeout = timeout
        self.calls: list[dict] = []

    def request(self, path: str, method: str = "GET", form: dict | None = None):
        fields = []
        for key, value in (form or {}).items():
            if isinstance(value, list):
                fields.extend((key, item) for item in value)
            else:
                fields.append((key, value))
        body = urlencode(fields).encode() if form is not None else None
        req = Request(BASE_URL + path, data=body, method=method,
                      headers={"Accept": "application/json"})
        started = perf_counter()
        try:
            with self.opener.open(req, timeout=self.timeout) as response:
                raw = response.read()
                status = response.status
        except HTTPError as exc:
            # Do not echo server bodies: authentication errors could contain PII.
            self.calls.append({"path": path, "method": method,
                               "status": exc.code,
                               "seconds": round(perf_counter() - started, 4)})
            raise ProbeError(f"{method} {path}: HTTP {exc.code}") from None
        except (URLError, TimeoutError, OSError):
            raise ProbeError(f"{method} {path}: network error or timeout") from None
        self.calls.append({"path": path, "method": method, "status": status,
                           "seconds": round(perf_counter() - started, 4),
                           "bytes": len(raw)})
        try:
            return json.loads(raw) if raw else None
        except (ValueError, UnicodeDecodeError):
            raise ProbeError(f"{method} {path}: response is not JSON") from None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="ICD10CM@2026")
    parser.add_argument("--codes", nargs="+", default=["K20.0", "J45", "E11"])
    parser.add_argument("--targets", nargs="+", default=TARGETS)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/codemapper/probe.json")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    username = os.getenv("CODEMAPPER_USERNAME")
    password = os.getenv("CODEMAPPER_PASSWORD")
    if not username or not password:
        print("Set CODEMAPPER_USERNAME and CODEMAPPER_PASSWORD in .env first.", file=sys.stderr)
        return 2

    client = Client(args.timeout)
    report = {"base_url": BASE_URL, "source": args.source,
              "codes": args.codes, "targets": args.targets,
              "array_style": "repeated", "calls": client.calls}
    logged_in = False
    failure = None
    try:
        login = client.request("/session", "POST", {"username": username, "password": password})
        if not isinstance(login, dict) or login.get("success") is not True:
            raise ProbeError("Login unsuccessful; check credentials locally.")
        logged_in = True
        # Verify that the cookie survives to a separate request; never save the user object.
        if not client.request("/session"):
            raise ProbeError("Login succeeded but session cookie was not retained.")
        report["vocabularies"] = client.request("/vocabularies")
        code_cuis = client.request("/legacy/code-mapper/cuis-for-codes", "POST",
                                   {"vocId": args.source.partition('@')[0], "codes": args.codes})
        code_cuis=normalize_code_versions(code_cuis,args.source)
        report["code_cuis"] = code_cuis
        cuis = sorted({cui for item in code_cuis.values() for cui in item["cuis"]})
        concepts = client.request("/legacy/code-mapper/umls-concepts", "POST",
                                  {"cuis": cuis, "vocIds": sorted({v.partition('@')[0] for v in args.targets})}) if cuis else []
        concepts=normalize_concept_versions(concepts,args.targets)
        report["concepts"] = concepts
        report["descendants_used"] = False
        mapped = {(s["codingSystem"], s["id"]) for c in concepts for s in c["sourceConcepts"]}
        report["summary"] = {
            "requested_codes": len(args.codes), "codes_with_cuis": len(code_cuis),
            "unmapped_codes": [code for code in args.codes if code not in code_cuis],
            "unique_cuis": len(cuis), "concepts": len(concepts),
            "unique_mapped_codes_by_vocabulary": dict(Counter(voc for voc, _ in mapped)),
        }
    except ProbeError as exc:
        failure = str(exc)
        report["error"] = failure
    finally:
        if logged_in:
            try:
                client.request("/session", "DELETE")
            except ProbeError:
                report["logout_failed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    # Print counts and timings only; full code/term responses stay in the ignored report.
    print(json.dumps({"summary": report.get("summary"), "calls": client.calls,
                      "error": failure, "report": str(args.output)}, indent=2))
    return 1 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
