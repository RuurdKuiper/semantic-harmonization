export const listFields = new Set(['inclusion_criteria', 'exclusion_criteria', 'narrow_decision_rules',
  'possible_decision_rules', 'exclude_decision_rules', 'synonyms', 'candidate_anchor_codes', 'ambiguities', 'references']);
export const textFields = new Set(['preferred_name', 'definition', 'narrow_definition', 'anatomical_location',
  'population_restrictions', 'temporal_characteristics', 'etiology', 'morphology', 'diagnostic_certainty', 'administrative_context']);

export function normalizeEDF(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw Error('Enter an EDF with a preferred_name.');
  const result = {};
  for (const [key, raw] of Object.entries(value)) {
    if (key === 'name') continue;
    if (listFields.has(key)) {
      if (raw != null && (!Array.isArray(raw) || raw.some(item => typeof item !== 'string'))) throw Error(`${key} must be a list of text values.`);
      result[key] = (raw || []).filter(item => item.trim());
    } else if (textFields.has(key)) {
      if (raw != null && typeof raw !== 'string') throw Error(`${key} must be text.`);
      result[key] = (raw || '').trim();
    } else throw Error(`Unknown EDF field: ${key}`);
  }
  if (!result.preferred_name) throw Error('Fill in preferred_name. All other EDF fields are optional.');
  return result;
}

export function queries(edf) {
  const lexical = [...new Set([edf.preferred_name, ...(edf.synonyms || [])].map(s => s.trim()).filter(Boolean))].join(' ');
  const lines = [`Preferred name: ${edf.preferred_name}`];
  if (edf.narrow_definition) lines.push(`Narrow definition: ${edf.narrow_definition}`);
  if (edf.synonyms?.length) lines.push(`Synonyms: ${edf.synonyms.join(', ')}`);
  for (const [key, label] of [['narrow_decision_rules', 'Narrow decision rules'], ['candidate_anchor_codes', 'Candidate anchor codes']]) {
    if (edf[key]?.length) lines.push(`${label}:`, ...edf[key].map(item => `  - ${item}`));
  }
  if (edf.morphology) lines.push(`Morphology/pathology: ${edf.morphology}`);
  if (edf.anatomical_location) lines.push(`Anatomical location: ${edf.anatomical_location}`);
  return { lexical, semantic: lines.join('\n') };
}

export function tokenize(text) {
  return text.toLowerCase().replace(/[^\p{L}\p{N}_\s]/gu, ' ').split(/\s+/u).filter(token => [...token].length > 1);
}

export const round6 = n => Math.round(n * 1e6) / 1e6;
export function bm25(query, description, stats) {
  const tokens = tokenize(description);
  if (!tokens.length) tokens.push(description);
  const counts = new Map();
  for (const token of tokens) counts.set(token, (counts.get(token) || 0) + 1);
  let score = 0;
  for (const token of tokenize(query)) {
    const df = stats.frequencies[token];
    const tf = counts.get(token) || 0;
    if (!df || !tf) continue;
    let idf = Math.log(stats.count - df + .5) - Math.log(df + .5);
    if (idf < 0) idf = stats.epsilon * stats.averageIdf;
    score += idf * tf * (stats.k1 + 1) / (tf + stats.k1 * (1 - stats.b + stats.b * tokens.length / stats.avgdl));
  }
  return round6(score);
}

export function minMax(values) {
  let lo = Infinity, hi = -Infinity;
  for (const value of values) { lo = Math.min(lo, value); hi = Math.max(hi, value); }
  const span = hi - lo;
  return Float64Array.from(values, value => span < 1e-12 ? 0 : (value - lo) / span);
}

export function rankScores(lexical, semantic, weight) {
  if (lexical.length !== semantic.length || !Number.isFinite(weight) || weight < 0 || weight > 1) throw Error('Invalid ranking inputs.');
  const l = minMax(lexical), s = minMax(semantic);
  const scores = Float64Array.from(l, (value, i) => round6(weight * value + (1 - weight) * s[i]));
  const order = Uint32Array.from(scores, (_, i) => i);
  order.sort((a, b) => scores[b] - scores[a] || a - b);
  return { order, scores, lexical: l, semantic: s };
}

export const keyOf = row => `${row.vocabulary}\u0000${row.code}`;
export function selectReview(decisions, threshold, cap) {
  return decisions.filter(row => row.confidence < threshold).sort((a, b) => a.confidence - b.confidence).slice(0, cap);
}

export function mergeReview(originals, replacements) {
  const byKey = new Map(replacements.map(row => [keyOf(row), row]));
  return originals.map(before => {
    const after = byKey.get(keyOf(before));
    return after ? { ...before, ...after, previous_label: before.label, previous_confidence: before.confidence,
      reviewed_by_gpt: true, label_changed: before.label !== after.label } : { ...before, reviewed_by_gpt: false, label_changed: false };
  });
}

export function validateDecisions(candidates, records, possible = false) {
  if (!Array.isArray(records) || records.length !== candidates.length) throw Error('The classification service returned an incomplete batch.');
  const expected = new Set(candidates.map(keyOf));
  const seen = new Set();
  for (const row of records) {
    const key = keyOf(row);
    if (!expected.has(key) || seen.has(key) || !(possible ? ['Narrow', 'Possible', 'Exclude'] : ['Narrow', 'Exclude']).includes(row.label)
      || !Number.isFinite(row.confidence) || row.confidence < 0 || row.confidence > 1 || typeof row.explanation !== 'string') throw Error('The classification service returned invalid decisions.');
    seen.add(key);
  }
  const byKey = new Map(records.map(row => [keyOf(row), row]));
  return candidates.map(row => ({ ...row, ...byKey.get(keyOf(row)) }));
}

export function standardKey(row) {
  const vocabulary = String(row.vocabulary || row.coding_system || '').trim().toUpperCase();
  let code = String(row.code || '').toUpperCase().replace(/\s+/g, '');
  if (['ICD10', 'ICD10CM'].includes(vocabulary) && !code.includes('.') && code.length > 3) code = `${code.slice(0, 3)}.${code.slice(3)}`;
  return keyOf({ code, vocabulary });
}

export function evaluate(decisions, gold, availableKeys) {
  const references = new Map();
  for (const row of gold) {
    const label = String(row.label || row.tags || '').trim().toLowerCase();
    if (!row.code || !(row.vocabulary || row.coding_system) || !['narrow', 'possible', 'exclude'].includes(label)) throw Error('Reference CSV needs code, vocabulary (or coding_system), and label (or tags: Narrow/Possible/Exclude).');
    if (!references.has(standardKey(row))) references.set(standardKey(row), label);
  }
  const vocabularies = new Set([...references.keys()].map(key => key.split('\u0000')[0]));
  const positive = new Set([...references].filter(([key, label]) => label === 'narrow' && availableKeys.has(key)).map(([key]) => key));
  const included = new Set(decisions.filter(row => row.label === 'Narrow' && vocabularies.has(row.vocabulary)).map(standardKey));
  const tp = [...included].filter(key => positive.has(key)).length;
  const fp = included.size - tp, fn = positive.size - tp;
  return { recall: positive.size ? tp / positive.size : 0, precision: included.size ? tp / included.size : 0,
    f1: 2 * tp + fp + fn ? 2 * tp / (2 * tp + fp + fn) : 0, tp, fp, fn,
    unavailable: [...references].filter(([key, label]) => label === 'narrow' && !availableKeys.has(key)).length };
}
