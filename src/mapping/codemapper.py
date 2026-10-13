"""One-round cross-vocabulary CUI expansion, with caching and paced sessions."""
from __future__ import annotations

from dataclasses import replace
try:
    import fcntl
except ImportError:  # Keep the disabled optional stage importable on Windows.
    fcntl = None
import hashlib
from http.cookiejar import CookieJar
import json
import logging
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener

import pandas as pd

from src.data.preprocessing import standardize_code
from src.llm.classify import ClassifiedCandidate
from src.utils.config import CodeMapperConfig


class CodeMapperError(RuntimeError):
    """Sanitized failure: never include credentials, cookies, or server bodies."""


def normalize_code_versions(response,vocabulary):
    """Select the pinned release from the current all-version API response."""
    _,separator,version=vocabulary.partition('@')
    if not separator:raise CodeMapperError('CodeMapper requires a pinned vocabulary version')
    result={}
    for code,value in response.items():
        if isinstance(value,list):
            matching=[item for item in value if isinstance(item,dict) and str(item.get('version'))==version]
            if len(matching)>1:raise CodeMapperError('Ambiguous CodeMapper code version response')
            if matching:result[code]=matching[0]
        elif isinstance(value,dict):
            result[code]=value  # Older cached/API responses were already release-specific.
        else:raise CodeMapperError('Unexpected CodeMapper code version response')
    return result


def normalize_concept_versions(response,targets):
    """Project series-level codes to requested releases, never to latest by default."""
    versions={target.partition('@')[0]:target.partition('@')[2] for target in targets}
    result=[]
    for concept in response:
        sources=[]
        for source in concept.get('sourceConcepts',[]):
            if 'codingSystem' in source:
                if source['codingSystem'] in targets:sources.append(source)
                continue
            series=source.get('vocId');version=versions.get(series)
            if version is None:continue
            matching=[item for item in source.get('versions',[]) if str(item.get('version'))==version]
            if matching:
                sources.append({**source,'codingSystem':series+'@'+version,
                                'preferredTerm':matching[0].get('term',source.get('preferredTerm'))})
        result.append({**concept,'sourceConcepts':sources})
    return result


class CodeMapperClient:
    def __init__(self, config: CodeMapperConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.opener = build_opener(HTTPCookieProcessor(CookieJar()))
        self.logged_in = False
        self.calls: list[dict] = []
        self.cache_hits = 0

    def _pace(self):
        if fcntl is None:
            raise CodeMapperError("Shared CodeMapper pacing currently requires macOS or Linux")
        # Share the limiter across client instances/processes using this cache.
        with (self.cache_dir / "rate.lock").open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.seek(0)
            previous = float(handle.read() or 0)
            delay = previous + 60 / self.config.requests_per_minute - time.time()
            if delay > 0:
                time.sleep(delay)
            handle.seek(0)
            handle.truncate()
            handle.write(str(time.time()))
            handle.flush()

    def _request(self, path, method="POST", form=None):
        fields = [(key, item) for key, value in (form or {}).items()
                  for item in (value if isinstance(value, list) else [value])]
        request = Request(self.config.base_url.rstrip("/") + path,
                          data=urlencode(fields).encode() if form is not None else None,
                          method=method, headers={"Accept": "application/json"})
        for attempt in range(self.config.max_retries):
            self._pace()
            logging.getLogger(__name__).info('CodeMapper %s %s (attempt %s)',method,path,attempt+1)
            started = time.perf_counter()
            try:
                with self.opener.open(request, timeout=self.config.timeout_seconds) as response:
                    raw = response.read()
                    status = response.status
                self.calls.append({"path": path, "status": status,
                                   "seconds": round(time.perf_counter() - started, 4),
                                   "bytes": len(raw)})
                return json.loads(raw) if raw else None
            except HTTPError as exc:
                self.calls.append({"path": path, "status": exc.code,
                                   "seconds": round(time.perf_counter() - started, 4)})
                if exc.code not in {429, 500, 502, 503, 504}:
                    raise CodeMapperError(f"CodeMapper {path}: HTTP {exc.code}") from None
                try:
                    retry_after = float(exc.headers.get("Retry-After", 0))
                except ValueError:
                    retry_after = 0
                if attempt + 1 < self.config.max_retries:
                    time.sleep(max(retry_after, 2 ** attempt))
            except (URLError, TimeoutError, OSError):
                if attempt + 1 < self.config.max_retries:
                    time.sleep(2 ** attempt)
            except (ValueError, UnicodeDecodeError):
                raise CodeMapperError(f"CodeMapper {path}: invalid JSON") from None
        raise CodeMapperError(f"CodeMapper {path}: failed after retries")

    def _login(self):
        if self.logged_in:
            return
        username, password = os.getenv("CODEMAPPER_USERNAME"), os.getenv("CODEMAPPER_PASSWORD")
        if not username or not password:
            raise CodeMapperError("Set CODEMAPPER_USERNAME and CODEMAPPER_PASSWORD in .env")
        response = self._request("/session", form={"username": username, "password": password})
        if not isinstance(response, dict) or response.get("success") is not True:
            raise CodeMapperError("CodeMapper login unsuccessful")
        self.logged_in = True

    def close(self):
        if self.logged_in:
            try:
                self._request("/session", "DELETE")
            finally:
                self.logged_in = False

    def _cache_path(self, kind, identity):
        key = hashlib.sha256(json.dumps([self.config.base_url, kind, identity],
                                       sort_keys=True).encode()).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _load(self, kind, identity):
        path = self._cache_path(kind, identity)
        if path.exists():
            self.cache_hits += 1
            return json.loads(path.read_text())
        return None

    def _save(self, kind, identity, value):
        path = self._cache_path(kind, identity)
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(value))
        temporary.replace(path)

    def cuis_for_codes(self, vocabulary, codes):
        result, missing = {}, []
        for code in sorted(set(codes)):
            cached = self._load("code_cuis", [vocabulary, code])
            if cached is None:
                missing.append(code)
            elif cached:
                result[code] = cached
        for start in range(0, len(missing), self.config.batch_size):
            batch = missing[start:start + self.config.batch_size]
            self._login()
            response = self._request("/legacy/code-mapper/cuis-for-codes",
                                     form={"vocId": vocabulary.partition('@')[0], "codes": batch})
            if not isinstance(response, dict):
                raise CodeMapperError("Unexpected code-to-CUI response")
            response=normalize_code_versions(response,vocabulary)
            for code in batch:
                value = response.get(code, {})
                self._save("code_cuis", [vocabulary, code], value)
                if value:
                    result[code] = value
        return result

    def concepts(self, cuis, targets):
        result, missing = [], []
        targets = sorted(set(targets))
        for cui in sorted(set(cuis)):
            cached = self._load("concept", [targets, cui])
            if cached is None:
                missing.append(cui)
            else:
                result.extend(cached)
        for start in range(0, len(missing), self.config.batch_size):
            batch = missing[start:start + self.config.batch_size]
            self._login()
            response = self._request("/legacy/code-mapper/umls-concepts",
                                     form={"cuis": batch, "vocIds": sorted({target.partition('@')[0] for target in targets})})
            if not isinstance(response, list):
                raise CodeMapperError("Unexpected concept response")
            response=normalize_concept_versions(response,targets)
            for cui in batch:
                values = [c for c in response if c.get("cui") == cui]
                self._save("concept", [targets, cui], values)
                result.extend(values)
        return result


def expand_narrow_matches(classified: list[ClassifiedCandidate], corpus: pd.DataFrame,
                          vocabularies: list[str], config: CodeMapperConfig,
                          *, client=None):
    """Promote/add direct cross-vocabulary matches from a frozen Narrow seed set.

    Only codes in the loaded/preprocessed corpus are eligible. Same-vocabulary
    links and descendants are excluded. No new additions become seeds this round.
    Mapping confidence is 0 (unassessed), not an invented model probability.
    """
    seeds = [c for c in classified if c.label == "Narrow" and c.vocabulary in vocabularies]
    versions = {v: f"{v}@{config.vocabulary_versions[v]}" for v in vocabularies}
    reverse = {version: voc for voc, version in versions.items()}
    by_cui = {}
    client = client or CodeMapperClient(config)
    started = time.perf_counter()
    seed_counts, unmapped = {}, []
    try:
        for voc in vocabularies:
            group = [c for c in seeds if c.vocabulary == voc]
            if not group:
                continue
            seed_counts[voc] = len(group)
            response = client.cuis_for_codes(versions[voc], [c.code for c in group])
            for seed in group:
                cuis = response.get(seed.code, {}).get("cuis", [])
                if not cuis:
                    unmapped.append({"code": seed.code, "vocabulary": voc})
                for cui in cuis:
                    by_cui.setdefault(cui, []).append(seed)
        concepts = client.concepts(by_cui, list(versions.values())) if by_cui else []
    finally:
        client.close()
    available = {(standardize_code(c, v), v): (c, d)
                 for c, v, d in corpus[["code", "vocabulary", "description"]].itertuples(index=False, name=None)}
    links, unavailable = {}, {}
    for concept in concepts:
        for source in concept.get("sourceConcepts", []):
            voc = reverse.get(source["codingSystem"])
            if voc is None:
                continue
            key = (standardize_code(source["id"], voc), voc)
            provenance = [{"seed_code": s.code, "seed_vocabulary": s.vocabulary,
                           "seed_label": s.label, "cui": concept["cui"],
                           "source_version": versions[s.vocabulary],
                           "target_version": source["codingSystem"]}
                          for s in by_cui.get(concept["cui"], []) if s.vocabulary != voc]
            if not provenance:
                continue
            target = links if key in available else unavailable
            target.setdefault(key, []).extend(provenance)
    merged = list(classified)
    positions = {(standardize_code(c.code, c.vocabulary), c.vocabulary): i
                 for i, c in enumerate(merged)}
    added, promoted, already_narrow = [], [], 0
    for key, provenance in sorted(links.items()):
        index = positions.get(key)
        previous = merged[index] if index is not None else None
        if previous is not None and previous.label == "Narrow":
            already_narrow += 1
            continue
        code, description = available[key]
        item = ClassifiedCandidate(code, description, key[1], "Narrow", 0.0,
                                   "Included by direct cross-vocabulary CodeMapper CUI mapping; "
                                   "not independently classified for EDF scope.",
                                   decision_origin="codemapper", mapping_provenance=provenance,
                                   prior_label=previous.label if previous else None,
                                   prior_confidence=previous.confidence if previous else None)
        if previous is None:
            merged.append(item)
            added.append({"code": code, "vocabulary": key[1]})
        else:
            merged[index] = replace(item, retrieval_score=previous.retrieval_score)
            promoted.append({"code": code, "vocabulary": key[1], "prior_label": previous.label})
    info = {"enabled": True, "rounds": 1, "descendants_used": False,
            "direct_links": [{"code": available[key][0], "vocabulary": key[1],
                              "provenance": provenance} for key, provenance in sorted(links.items())],
            "seeds": len(seeds), "seed_counts": seed_counts, "unique_cuis": len(by_cui),
            "unmapped_seeds": unmapped, "new_codes": added, "promoted_codes": promoted,
            "already_narrow": already_narrow,
            "unavailable_codes": [{"code": c, "vocabulary": v, "provenance": p}
                                  for (c, v), p in sorted(unavailable.items())],
            "calls": client.calls, "cache_hits": client.cache_hits,
            "seconds": round(time.perf_counter() - started, 3)}
    return merged, info
