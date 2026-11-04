import yaml from 'js-yaml';
import Papa from 'papaparse';
import { normalizeEDF, queries, keyOf, selectReview, mergeReview, validateDecisions, evaluate, rankingWeight } from './core.js';
import './style.css';

const $ = id => document.getElementById(id);
const vocabularies = [['ICD10CM', 'ICD-10-CM'], ['ICD9CM', 'ICD-9-CM'], ['ICPC', 'ICPC'],
  ['MDR', 'MedDRA'], ['RCD2', 'Read 2'], ['SNOMEDCT_US', 'SNOMED CT US']];
const state = { worker: null, requests: new Map(), sequence: 0, ranked: null, initial: [], reviewed: [],
  reviewReplacements: [], busy: false, abort: null, page: 0, total: 0, reference: [], config: {}, stopped: false };

function notice(message = '') { $('notice').textContent = message; }
function edf() { return normalizeEDF(yaml.load($('edf-yaml').value)); }
function selected() { return [...document.querySelectorAll('[name=vocabulary]:checked')].map(input => input.value); }
function numeric(id, min, max) {
  const value = Number($(id).value);
  if (!Number.isFinite(value) || value < min || value > max) throw Error(`Enter a value from ${min} to ${max} for ${$(id).closest('label')?.firstChild.textContent.trim() || id}.`);
  return value;
}

function worker() {
  if (!state.worker) {
    state.worker = new Worker(new URL('./retrieval.worker.js', import.meta.url), { type: 'module' });
    state.worker.onmessage = ({ data }) => {
      const pending = state.requests.get(data.id);
      if (!pending) return;
      if (data.progress) { showProgress(data.progress.message, data.progress.fraction); return; }
      state.requests.delete(data.id);
      data.error ? pending.reject(Error(data.error)) : pending.resolve(data.result);
    };
    state.worker.onerror = () => {
      for (const request of state.requests.values()) request.reject(Error('The browser retrieval worker stopped. Retry with fewer terminologies if your device is short of memory.'));
      state.requests.clear(); state.worker.terminate(); state.worker = null; state.ranked = null; updateButtons();
    };
  }
  return state.worker;
}
function request(method, args = {}) {
  return new Promise((resolve, reject) => {
    const id = ++state.sequence;
    state.requests.set(id, { resolve, reject });
    worker().postMessage({ id, method, args });
  });
}

function showProgress(message, fraction) {
  $('progress-box').hidden = false;
  $('progress-text').textContent = message;
  if (Number.isFinite(fraction)) $('progress').value = fraction;
  else $('progress').removeAttribute('value');
}
function updateButtons() {
  const hasAPI = !!state.config.apiBaseUrl;
  $('run-retrieval').disabled = state.busy || !state.pack;
  $('run-classification').disabled = state.busy || !state.ranked || !hasAPI;
  $('run-classification').textContent = state.initial.length ? 'Continue classification' : 'Run classification';
  $('run-review').disabled = state.busy || !state.initial.length || !hasAPI;
  $('run-evaluation').disabled = state.busy || !state.initial.length || !state.reference.length;
  for (const id of ['download-csv', 'download-json']) $(id).disabled = state.busy || !state.initial.length;
  $('cancel').hidden = !state.busy;
  for (const control of document.querySelectorAll('#edf input, #edf select, #edf textarea, #terminologies input, #embedding-provider, #possible-category, #candidate-cap, #sparse-batches, #clear-cache, #confidence-threshold, #review-cap')) {
    control.disabled = state.busy || (control.name === 'vocabulary' && state.pack && !state.pack.vocabularies.some(item => item.id === control.value));
  }
}
async function action(callback) {
  if (state.busy) return;
  state.busy = true; state.stopped = false; notice(); updateButtons();
  try { await callback(); }
  catch (error) { if (error.name !== 'AbortError') notice(error.message || String(error)); }
  finally { state.busy = false; state.abort = null; $('progress-box').hidden = true; updateButtons(); }
}
function resetDecisions() {
  state.initial = []; state.reviewed = []; state.reviewReplacements = [];
  state.metrics = undefined; state.classifier = undefined; state.reviewer = undefined; state.stopReason = undefined;
  $('classification-table').replaceChildren(); $('review-table').replaceChildren(); $('metrics').replaceChildren();
  $('classification-status').textContent = ''; $('review-status').textContent = '';
  updateButtons();
}
function invalidateRanking() {
  state.ranked = null; state.page = 0;
  $('retrieval-results').hidden = true;
  resetDecisions();
}

function table(target, rows, columns) {
  target.replaceChildren();
  if (!rows.length) return;
  const table = document.createElement('table'), head = table.createTHead().insertRow(), body = table.createTBody();
  for (const [key, label] of columns) { const th = document.createElement('th'); th.textContent = label; th.scope = 'col'; head.append(th); }
  for (const row of rows) {
    const tr = body.insertRow();
    for (const [key] of columns) {
      const td = tr.insertCell(), value = row[key]; td.className = key;
      if (key === 'label') { const pill = document.createElement('span'); pill.className = `pill ${value.toLowerCase()}`; pill.textContent = value; td.append(pill); }
      else if (typeof value === 'number') { td.className = 'numeric'; td.textContent = key === 'rank' ? value.toLocaleString() : value.toFixed(3); }
      else td.textContent = value == null ? '' : typeof value === 'boolean' ? (value ? 'Yes' : 'No') : String(value);
    }
  }
  target.append(table);
}
const decisionColumns = [['code', 'Code'], ['vocabulary', 'Terminology'], ['description', 'Description'],
  ['label', 'Decision'], ['confidence', 'Confidence'], ['explanation', 'Explanation']];
function showDecisions() {
  // The initial decisions remain in step 4, even after review.
  table($('classification-table'), state.initial.slice(0, 300), decisionColumns);
  let reviewed = state.reviewed;
  if ($('reviewed-only').checked) reviewed = reviewed.filter(row => row.reviewed_by_gpt);
  table($('review-table'), reviewed.slice(0, 300), [...decisionColumns, ['previous_label', 'Previous decision'],
    ['previous_confidence', 'Previous confidence'], ['label_changed', 'Changed'], ['reviewed_by_gpt', 'GPT reviewed']]);
}

async function showPage() {
  if (!state.ranked) return;
  const page = await request('page', { offset: state.page * 100, limit: 100, filter: $('search-retrieval').value });
  state.total = page.total;
  table($('retrieval-table'), page.items, [['rank', 'Rank'], ['code', 'Code'], ['vocabulary', 'Terminology'],
    ['description', 'Description'], ['lexical_score', 'Lexical'], ['embedding_score', 'Semantic'], ['retrieval_score', 'Combined']]);
  $('page-label').textContent = `${page.total.toLocaleString()} matches · page ${state.page + 1} of ${Math.max(1, Math.ceil(page.total / 100))}`;
  $('previous-page').disabled = state.page === 0;
  $('next-page').disabled = (state.page + 1) * 100 >= page.total;
}

async function postAPI(stage, payload, attempt = 0) {
  state.abort = new AbortController();
  const response = await fetch(`${state.config.apiBaseUrl.replace(/\/$/, '')}/api/${stage}`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload), signal: state.abort.signal,
  });
  let body;
  try { body = await response.json(); } catch { throw Error(`The model service returned an unreadable response (${response.status}).`); }
  const retryAfter = Number(response.headers.get('Retry-After'));
  if (response.status === 429 && retryAfter > 0 && retryAfter <= 60 && attempt < 3) {
    showProgress('The model service is busy. Retrying shortly…');
    await new Promise((resolve, reject) => {
      const timer = setTimeout(resolve, retryAfter * 1000);
      state.abort.signal.addEventListener('abort', () => { clearTimeout(timer); reject(new DOMException('Stopped', 'AbortError')); }, { once: true });
    });
    return postAPI(stage, payload, attempt + 1);
  }
  if (!response.ok) throw Error(typeof body.detail === 'string' ? body.detail : `Model request failed (${response.status}). Please retry.`);
  return body;
}
async function callAPI(stage, candidates) {
  const stripped = candidates.map(({ code, vocabulary, description }) => ({ code, vocabulary, description }));
  const payload = await postAPI(stage, { edf: edf(), candidates: stripped, use_possible_category: $('possible-category').checked });
  return { ...payload, decisions: validateDecisions(candidates, payload.decisions, $('possible-category').checked) };
}

async function classify() {
  if (!state.ranked) throw Error('Run retrieval first.');
  const cap = Math.min(Math.floor(numeric('candidate-cap', 20, 10000)), state.ranked.count);
  const sparse = Math.floor(numeric('sparse-batches', 1, 100));
  state.reviewed = []; state.reviewReplacements = []; state.metrics = undefined; $('metrics').replaceChildren();
  let emptyStreak = 0, batches = 0;
  for (let start = 0; start < state.initial.length; start += 20) {
    emptyStreak = state.initial.slice(start, start + 20).some(row => row.label === 'Narrow') ? 0 : emptyStreak + 1;
    batches++;
  }
  let reason = 'Candidate ceiling reached.';
  while (state.initial.length < cap && !state.stopped) {
    if (batches >= 3 && emptyStreak >= sparse) { reason = `${emptyStreak} consecutive batches contained no Narrow decisions.`; break; }
    const candidates = await request('batch', { offset: state.initial.length, limit: Math.min(20, cap - state.initial.length) });
    if (!candidates.length) { reason = 'All candidates classified.'; break; }
    showProgress(`Classifying candidates ${state.initial.length + 1}–${state.initial.length + candidates.length}…`, state.initial.length / cap);
    const result = await callAPI('classify', candidates);
    state.initial.push(...result.decisions);
    state.classifier = { provider: result.provider, model: result.model };
    emptyStreak = result.decisions.some(row => row.label === 'Narrow') ? 0 : emptyStreak + 1; batches++;
    $('classification-status').textContent = `Classified ${state.initial.length.toLocaleString()} of ${state.ranked.count.toLocaleString()} ranked candidates. Table shows up to 300; downloads include every decision.`;
    showDecisions(); updateButtons();
  }
  if (state.stopped) reason = 'Stopped by user.';
  else if (state.initial.length === state.ranked.count) reason = 'All candidates classified.';
  state.stopReason = reason;
  $('classification-status').textContent = `${state.initial.length.toLocaleString()} decisions. ${reason} Table shows up to 300; downloads include every decision.`;
}

async function review() {
  const selected = selectReview(state.initial, numeric('confidence-threshold', 0, 1), Math.floor(numeric('review-cap', 1, 300)));
  if (!selected.length) throw Error('No initial decisions are below this confidence threshold.');
  // Retain successful review batches on retry, as long as the selection is unchanged.
  const signature = JSON.stringify(selected.map(keyOf));
  if (signature !== state.reviewSignature) state.reviewReplacements = [];
  state.reviewSignature = signature;
  const completed = new Set(state.reviewReplacements.map(keyOf));
  const pending = selected.filter(row => !completed.has(keyOf(row)));
  for (let offset = 0; offset < pending.length && !state.stopped; offset += 10) {
    showProgress(`GPT review: ${state.reviewReplacements.length} of ${selected.length} decisions…`, state.reviewReplacements.length / selected.length);
    const result = await callAPI('review', pending.slice(offset, offset + 10));
    state.reviewer = { provider: result.provider, model: result.model };
    state.reviewReplacements.push(...result.decisions);
    state.reviewed = mergeReview(state.initial, state.reviewReplacements);
    showDecisions();
    state.metrics = undefined;
    $('metrics').replaceChildren();
    $('review-status').textContent = `GPT reviewed ${state.reviewReplacements.length} of ${selected.length} decisions; ${state.reviewed.filter(row => row.label_changed).length} labels changed.`;
  }
}

function download(name, content, mime) {
  const url = URL.createObjectURL(new Blob([content], { type: mime }));
  const link = document.createElement('a'); link.href = url; link.download = name; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
const finalDecisions = () => state.reviewed.length ? state.reviewed : state.initial;

function updateQueryPreview() {
  try { const q = queries(edf()); $('query-preview').textContent = `BM25\n${q.lexical}\n\nMiniLM\n${q.semantic}`; }
  catch { $('query-preview').textContent = 'Fill in preferred_name to preview the retrieval queries.'; }
}
function selectedProvider() { return $('embedding-provider').value; }
function manifestURL() {
  const provider = selectedProvider();
  const value = state.config.embeddingManifests?.[provider] || (provider === 'minilm' ? state.config.manifestUrl : `${state.config.apiBaseUrl}/api/assets/openai-256-int8/manifest.json`);
  if (!value) throw Error('The tool is not configured yet. Please contact its administrator.');
  return new URL(value, location.href).href;
}
async function loadPack() {
  state.pack = undefined; updateButtons();
  let response;
  try { response = await fetch(manifestURL(), { cache: 'no-cache' }); }
  catch { throw Error('Could not connect to the terminology service. Please retry; it may be starting.'); }
  if (!response.ok) throw Error('The terminology service is temporarily unavailable. Please retry or contact the tool administrator.');
  const pack = await response.json();
  if (pack.format !== `semantic-harmonization-${selectedProvider()}-v1`) throw Error('The terminology data is incompatible with the selected model.');
  state.pack = pack;
  const available = new Set(pack.vocabularies.map(row => row.id));
  for (const input of document.querySelectorAll('[name=vocabulary]')) {
    input.disabled = !available.has(input.value);
    if (input.disabled) input.checked = false;
  }
  updateEstimate(); updateButtons();
}
function updateEstimate() {
  if (!state.pack) return;
  const items = state.pack.vocabularies.filter(row => selected().includes(row.id));
  const modelDownload = selectedProvider() === 'minilm' ? ', plus the MiniLM model' : '';
  $('download-estimate').textContent = `${items.reduce((n, row) => n + row.count, 0).toLocaleString()} codes · ${Math.ceil(items.reduce((n, row) => n + row.bytes, 0) / 1024 / 1024).toLocaleString()} MB on first use${modelDownload}. Saved downloads are reused when browser storage is available.`;
}

for (const [id, name] of vocabularies) {
  const label = document.createElement('label'), input = document.createElement('input');
  input.type = 'checkbox'; input.name = 'vocabulary'; input.value = id; input.checked = true;
  input.addEventListener('change', () => { invalidateRanking(); updateEstimate(); });
  label.append(input, name); $('vocabularies').append(label);
}
$('edf-yaml').addEventListener('input', () => { invalidateRanking(); updateQueryPreview(); });
$('embedding-provider').addEventListener('change', () => action(async () => { invalidateRanking(); await loadPack(); }));
$('possible-category').addEventListener('change', resetDecisions);
$('reviewed-only').addEventListener('change', showDecisions);
$('clear-cache').addEventListener('click', () => action(async () => { await request('clear'); notice('Downloaded terminology data cleared. The MiniLM model uses a separate browser cache.'); }));
$('run-retrieval').addEventListener('click', () => action(async () => {
  const definition = edf();
  if (!selected().length) throw Error('Select at least one terminology.');
  invalidateRanking();
  const provider = selectedProvider();
  let embeddingVector;
  if (provider === 'openai') {
    showProgress('Embedding your event definition with OpenAI…');
    const result = await postAPI('embed', { edf: definition, dimensions: state.pack.dimensions });
    if (result.dimensions !== state.pack.dimensions || result.model !== state.pack.model) throw Error('The query embedding does not match the selected terminology pack.');
    embeddingVector = result.vector;
    state.abort = null;
  }
  state.ranked = await request('retrieve', { edf: definition, manifestUrl: manifestURL(), provider,
    vocabularies: selected(), weight: rankingWeight(provider), embeddingVector });
  $('retrieval-summary').textContent = `Ranked ${state.ranked.count.toLocaleString()} codes on your computer.`;
  $('retrieval-results').hidden = false; state.page = 0; await showPage();
}));
$('run-classification').addEventListener('click', () => action(classify));
$('run-review').addEventListener('click', () => action(review));
$('cancel').addEventListener('click', () => {
  state.stopped = true;
  if (state.abort) state.abort.abort();
  else if (state.worker && !state.ranked) {
    state.worker.terminate(); state.worker = null;
    for (const pending of state.requests.values()) pending.reject(new DOMException('Stopped', 'AbortError'));
    state.requests.clear();
  }
});
$('previous-page').addEventListener('click', () => { state.page = Math.max(0, state.page - 1); showPage().catch(error => notice(error.message)); });
$('next-page').addEventListener('click', () => { state.page++; showPage().catch(error => notice(error.message)); });
let searchTimer;
$('search-retrieval').addEventListener('input', () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => { state.page = 0; showPage().catch(error => notice(error.message)); }, 180); });
$('upload-edf').addEventListener('change', async event => {
  const file = event.target.files[0]; if (!file) return;
  $('edf-yaml').value = await file.text(); invalidateRanking(); updateQueryPreview();
});
$('example').addEventListener('change', async () => {
  if (!$('example').value) return;
  try {
    const response = await fetch(new URL(`examples/${$('example').value}`, document.baseURI));
    if (!response.ok) throw Error('Cannot load this example.');
    $('edf-yaml').value = await response.text(); invalidateRanking(); updateQueryPreview();
  } catch (error) { notice(error.message); }
});
$('upload-reference').addEventListener('change', async event => {
  const file = event.target.files[0]; state.reference = []; $('metrics').replaceChildren();
  if (file) {
    const parsed = Papa.parse(await file.text(), { header: true, skipEmptyLines: 'greedy' });
    if (parsed.errors.length) notice('The reference CSV could not be parsed. Check its column names and quoting.');
    else state.reference = parsed.data;
  }
  updateButtons();
});
$('run-evaluation').addEventListener('click', () => action(async () => {
  const available = new Set(await request('available'));
  const result = evaluate(finalDecisions(), state.reference, available);
  state.metrics = result; $('metrics').replaceChildren();
  for (const [metric, label] of [['recall', 'Recall'], ['precision', 'Precision'], ['f1', 'F1']]) {
    const card = document.createElement('div'); card.className = 'metric';
    const caption = document.createElement('span'); caption.textContent = label;
    const value = document.createElement('strong'); value.textContent = `${(100 * result[metric]).toFixed(1)}%`;
    card.append(caption, value); $('metrics').append(card);
  }
  notice(`TP ${result.tp} · FP ${result.fp} · FN ${result.fn}. ${result.unavailable} Narrow reference codes are unavailable in the selected corpus and excluded from recall.`);
}));
$('download-csv').addEventListener('click', () => {
  const rows = finalDecisions().map(row => Object.fromEntries([...decisionColumns, ['rank'], ['previous_label'], ['previous_confidence'], ['reviewed_by_gpt'], ['label_changed']].map(([key]) => [key, row[key] ?? ''])));
  download('codelist-decisions.csv', Papa.unparse(rows, { escapeFormulae: true }), 'text/csv;charset=utf-8');
});
$('download-json').addEventListener('click', () => download('codelist-results.json', JSON.stringify({
  edf: edf(), retrieval: state.ranked, lexicalWeight: rankingWeight(selectedProvider()),
  classifier: state.classifier, reviewer: state.reviewer, stoppingReason: state.stopReason,
  usePossibleCategory: $('possible-category').checked, initialDecisions: state.initial, decisions: finalDecisions(), metrics: state.metrics,
}, null, 2), 'application/json'));

async function init() {
  try {
    const response = await fetch(new URL('config.json', document.baseURI), { cache: 'no-cache' });
    const config = response.ok ? await response.json() : {};
    state.config = config;
    await loadPack();
    const examples = await fetch(new URL('examples/index.json', document.baseURI));
    if (examples.ok) for (const example of await examples.json()) {
      const option = document.createElement('option'); option.value = example.file; option.textContent = example.name;
      $('example').append(option);
    }
  } catch (error) { notice(error.message); }
  updateQueryPreview(); updateButtons();
}
init();
