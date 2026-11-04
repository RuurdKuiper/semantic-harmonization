import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { normalizeEDF, queries, bm25, rankScores, selectReview, mergeReview, validateDecisions, evaluate } from '../src/core.js';

test('name-only and cleared YAML fields work; retrieval leaves out general definitions and exclusion text', () => {
  const edf = normalizeEDF({ preferred_name: 'Migraine', definition: null, synonyms: null, exclusion_criteria: ['Excluded concept'] });
  assert.deepEqual(queries(edf), { lexical: 'Migraine', semantic: 'Preferred name: Migraine' });
  assert.throws(() => normalizeEDF({ preferred_name: '' }), /preferred_name/);
  assert.throws(() => normalizeEDF({ preferred_name: 'A', synonyms: 'B' }), /list/);
});

test('browser query construction matches the Python EDF field order', () => {
  const edf = normalizeEDF({ preferred_name: 'A', definition: 'not for retrieval', narrow_definition: 'B', synonyms: ['C'],
    narrow_decision_rules: ['D'], candidate_anchor_codes: ['E'], morphology: 'F', anatomical_location: 'G' });
  assert.equal(queries(edf).semantic, 'Preferred name: A\nNarrow definition: B\nSynonyms: C\nNarrow decision rules:\n  - D\nCandidate anchor codes:\n  - E\nMorphology/pathology: F\nAnatomical location: G');
});

test('BM25 and hybrid ranking agree with Python golden results, including repeated query tokens', () => {
  const fixture = JSON.parse(readFileSync(new URL('./ranking-fixture.json', import.meta.url)));
  const lexical = fixture.descriptions.map(text => bm25(fixture.query, text, fixture.lexical));
  assert.deepEqual(lexical, fixture.lexicalScores);
  const ranked = rankScores(lexical, fixture.semanticScores, fixture.weight);
  assert.deepEqual([...ranked.order], fixture.order);
  fixture.scores.forEach((score, i) => assert.ok(Math.abs(ranked.scores[i] - score) <= 1e-6));
  assert.deepEqual([...rankScores([0, 0, 0], [1, 1, 1], .5).order], [0, 1, 2]);
});

const original = [
  { code: 'A', vocabulary: 'ICD10CM', label: 'Exclude', confidence: .2, explanation: 'First' },
  { code: 'B', vocabulary: 'ICD10CM', label: 'Narrow', confidence: .9, explanation: 'Second' },
  { code: 'C', vocabulary: 'ICD10CM', label: 'Exclude', confidence: .1, explanation: 'Third' },
];
test('review selects lowest-confidence decisions and preserves original ordering and prior values', () => {
  assert.deepEqual(selectReview(original, .3, 1).map(row => row.code), ['C']);
  const merged = mergeReview(original, [{ ...original[2], label: 'Narrow', confidence: .8, explanation: 'Reviewed' }]);
  assert.deepEqual(merged.map(row => row.code), ['A', 'B', 'C']);
  assert.equal(merged[2].previous_label, 'Exclude');
  assert.equal(merged[2].explanation, 'Reviewed');
  assert.equal(merged[2].label_changed, true);
  assert.equal(original[2].label, 'Exclude');
});
test('incomplete, duplicate and out-of-candidate responses are rejected before advancing a batch', () => {
  assert.throws(() => validateDecisions(original, original.slice(0, 2)), /incomplete/);
  assert.throws(() => validateDecisions(original, [original[0], original[0], original[2]]), /invalid/);
  assert.throws(() => validateDecisions(original, [original[0], original[1], { ...original[2], confidence: 2 }]), /invalid/);
});
test('evaluation includes unclassified positives, excludes unavailable references, and counts unreferenced inclusions as FP', () => {
  const decisions = [{ code: 'A', vocabulary: 'ICD10CM', label: 'Narrow' }, { code: 'D', vocabulary: 'ICD10CM', label: 'Narrow' }];
  const gold = ['A', 'B', 'C'].map(code => ({ code, vocabulary: 'ICD10CM', label: 'Narrow' }));
  const available = new Set(['A', 'B', 'D'].map(code => `ICD10CM\u0000${code}`));
  assert.deepEqual(evaluate(decisions, gold, available), { recall: .5, precision: .5, f1: .5, tp: 1, fp: 1, fn: 1, unavailable: 1 });
});
