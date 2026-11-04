import { test, expect } from '@playwright/test';
import { createHash } from 'node:crypto';
import { gzipSync } from 'node:zlib';

for (const [provider, dtype] of [['minilm', 'int8'], ['minilm', 'fp32'], ['openai', 'fp32']]) test(`automatic public retrieval, weighting, retry, review and export (${provider}/${dtype})`, async ({ page, context }) => {
  const count = 65;
  const packBase = 'https://api.example.test/api/assets/';
  const dimensions = provider === 'openai' ? 3072 : 384;
  const rows = Array.from({ length: count }, (_, i) => [`C${String(i).padStart(3, '0')}`, `Migraine candidate ${i}`]);
  const vectors = new Float32Array(count * dimensions);
  for (let i = 0; i < count; i++) vectors[i * dimensions] = 1;
  const statistics = gzipSync(JSON.stringify({ count, avgdl: 3, averageIdf: 1, frequencies: { migraine: count, candidate: count }, k1: 1.5, b: .75, epsilon: .25 }));
  const vectorBytes = Buffer.from(vectors.buffer);
  const files = new Map();
  const asset = (url, bytes) => { files.set(url, bytes); return { url, bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex') }; };
  const chunks = Array.from({ length: 4 }, (_, i) => {
    const start = i * 20, size = Math.min(20, count - start);
    let storedVectors = vectorBytes.subarray(start * dimensions * 4, (start + size) * dimensions * 4);
    if (dtype === 'int8') {
      storedVectors = Buffer.alloc(size * (dimensions + 4));
      for (let row = 0; row < size; row++) { storedVectors.writeInt8(127, row * dimensions); storedVectors.writeFloatLE(1 / 127, size * dimensions + row * 4); }
    }
    return { start, count: size, rows: asset(`rows-${i}.json.gz`, gzipSync(JSON.stringify(rows.slice(start, start + size)))),
      vectors: asset(`vectors-${i}.${dtype === 'int8' ? 'i8' : 'f32'}`, storedVectors) };
  });
  let releaseVectors;
  const allVectorsStarted = new Promise(resolve => { releaseVectors = resolve; });
  const pack = { format: `semantic-harmonization-${provider}-v1`, dimensions, model: provider === 'openai' ? 'text-embedding-3-large' : 'Xenova/all-MiniLM-L6-v2', dtype, obsoleteVectorHashes: dtype === 'int8' ? ['f'.repeat(64)] : [], quantization: dtype === 'int8' ? 'symmetric-int8-unit-vectors-v1' : undefined, maxLength: provider === 'openai' ? 8192 : 256,
    vocabularies: [{ id: 'ICPC', version: 'test-v1', count, bytes: vectorBytes.length + statistics.length,
      lexical: asset('lexical.json.gz', statistics), chunks }] };
  let vectorDownloads = 0, classificationCalls = 0, reviewCalls = 0, embeddingCalls = 0;
  const headers = { 'Access-Control-Allow-Origin': '*', 'Content-Type': 'application/json' };
  // Intercept the ONNX module only in this test. Production uses the real MiniLM runtime.
  await context.route(/\/assets\/minilm-runtime[^/]*\.js$/, route => route.fulfill({ contentType: 'application/javascript', body: 'export async function createMiniLMEncoder(){ return async()=>({data:Float32Array.from({length:384},(_,i)=>i===0?1:0)}); }' }));
  await context.route(`${packBase}**`, async route => {
    if (route.request().method() === 'OPTIONS') return route.fulfill({ status: 204, headers: { ...headers, 'Access-Control-Allow-Headers': '*', 'Access-Control-Allow-Methods': 'GET' } });
    expect(route.request().headers().authorization).toBeUndefined();
    const name = new URL(route.request().url()).pathname.split('/').pop();
    if (name === 'manifest.json') {
      const requestedOpenAI = route.request().url().includes('/openai/');
      const manifest = { ...pack, format: requestedOpenAI ? 'semantic-harmonization-openai-v1' : 'semantic-harmonization-minilm-v1' };
      return route.fulfill({ headers, body: JSON.stringify(manifest) });
    }
    if (name.startsWith('vectors-')) {
      vectorDownloads++;
      if (vectorDownloads === chunks.length) releaseVectors();
      // All four downloads must start before the first is allowed to finish.
      await allVectorsStarted;
    }
    if (files.has(name)) return route.fulfill({ headers, body: files.get(name) });
    return route.abort();
  });
  await context.route('https://api.example.test/**', async route => {
    if (new URL(route.request().url()).pathname.startsWith('/api/assets/')) return route.fallback();
    if (route.request().method() === 'OPTIONS') return route.fulfill({ status: 204, headers: { ...headers, 'Access-Control-Allow-Headers': '*', 'Access-Control-Allow-Methods': 'POST' } });
    expect(route.request().headers().authorization).toBeUndefined();
    const body = route.request().postDataJSON();
    expect(body.edf.preferred_name).toBe('Migraine');
    if (route.request().url().endsWith('/embed')) { embeddingCalls++; return route.fulfill({ headers, body: JSON.stringify({ vector: Array.from({ length: 3072 }, (_, i) => i === 0 ? 1 : 0), model: 'text-embedding-3-large', dimensions: 3072 }) }); }
    const review = route.request().url().endsWith('/review');
    if (review) reviewCalls++; else classificationCalls++;
    if (!review && classificationCalls === 2) return route.fulfill({ status: 502, headers, body: JSON.stringify({ detail: 'Temporary model failure; retry to continue.' }) });
    const decisions = body.candidates.map(row => ({ ...row, label: review ? 'Narrow' : 'Exclude', confidence: review ? .9 : .2, explanation: review ? 'Explicit event diagnosis.' : 'Insufficient detail.' }));
    return route.fulfill({ headers, body: JSON.stringify({ decisions, provider: review ? 'openai' : 'jev', model: 'test-model' }) });
  });
  await context.route('**/config.json', route => route.fulfill({ json: { manifestUrl: `${packBase}manifest.json`, apiBaseUrl: 'https://api.example.test' } }));
  await page.goto('http://127.0.0.1:4173');
  await expect(page.locator('#run-retrieval')).toBeEnabled();
  if (provider === 'openai') { await page.locator('#embedding-provider').selectOption('openai'); await expect(page.locator('#run-retrieval')).toBeEnabled(); }
  await expect(page.locator('input[type=password], input[type=url], input[type=range], #connections')).toHaveCount(0);
  await expect(page.locator('[name=vocabulary][value=ICD10CM]')).toBeDisabled();
  await page.locator('#edf-yaml').fill('preferred_name: Migraine\ndefinition:\nsynonyms:');
  if (dtype === 'int8') await page.evaluate(() => new Promise((resolve, reject) => {
    const open = indexedDB.open('semantic-harmonization-minilm-v1', 1);
    open.onupgradeneeded = () => open.result.createObjectStore('assets');
    open.onerror = () => reject(open.error);
    open.onsuccess = () => {
      const db = open.result, tx = db.transaction('assets', 'readwrite');
      tx.objectStore('assets').put(new ArrayBuffer(4), 'f'.repeat(64));
      tx.objectStore('assets').put(new ArrayBuffer(4), 'e'.repeat(64));
      tx.oncomplete = () => { db.close(); resolve(); };
    };
  }));
  await page.locator('#run-retrieval').click();
  await expect(page.locator('#retrieval-summary')).toHaveText('Ranked 65 codes on your computer.');
  await expect(page.locator('#retrieval-table tbody tr')).toHaveCount(65);
  await page.locator('#run-retrieval').click();
  await expect(page.locator('#run-retrieval')).toBeEnabled();
  expect(vectorDownloads).toBe(chunks.length);
  if (dtype === 'int8') expect(await page.evaluate(() => new Promise(resolve => {
    const open = indexedDB.open('semantic-harmonization-minilm-v1', 1);
    open.onsuccess = () => {
      const db = open.result, store = db.transaction('assets').objectStore('assets');
      const old = store.get('f'.repeat(64)), kept = store.get('e'.repeat(64));
      kept.onsuccess = () => { resolve({ removed: old.result === undefined, kept: kept.result !== undefined }); db.close(); };
    };
  }))).toEqual({ removed: true, kept: true });
  expect(embeddingCalls).toBe(provider === 'openai' ? 2 : 0);
  await expect(page.locator('#retrieval-table tbody tr').first().locator('td').last()).toHaveText(provider === 'openai' ? '0.350' : '0.500');
  await page.locator('#sparse-batches').fill('1');
  await page.locator('#run-classification').click();
  await expect(page.locator('#notice')).toContainText('Temporary model failure');
  await expect(page.locator('#classification-table tbody tr')).toHaveCount(20);
  await page.locator('#run-classification').click();
  await expect(page.locator('#classification-status')).toContainText('60 decisions. 3 consecutive batches');
  expect(classificationCalls).toBe(4);
  await page.locator('#review-cap').fill('1');
  await page.locator('#run-review').click();
  await expect(page.locator('#review-status')).toContainText('1 labels changed');
  expect(reviewCalls).toBe(1);
  await expect(page.locator('#review-table tbody tr')).toHaveCount(1);
  await expect(page.locator('#review-table')).toContainText('Explicit event diagnosis.');
  await expect(page.locator('#review-table')).toContainText('Exclude');
  await expect(page.locator('#classification-table tbody tr').first()).toContainText('Exclude');
  await page.locator('#upload-reference').setInputFiles({ name: 'reference.csv', mimeType: 'text/csv', buffer: Buffer.from('code,vocabulary,label\nC000,ICPC,Narrow\n') });
  await page.locator('#run-evaluation').click();
  await expect(page.locator('#metrics strong')).toHaveText(['100.0%', '100.0%', '100.0%']);
  const downloadPromise = page.waitForEvent('download');
  await page.locator('#download-csv').click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe('codelist-decisions.csv');
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(1440);
  await page.screenshot({ path: '/tmp/semantic-harmonization-browser.png', fullPage: true });
  await page.locator('#edf-yaml').fill('preferred_name: Another event');
  await expect(page.locator('#run-classification')).toBeDisabled();
  await expect(page.locator('#retrieval-results')).toBeHidden();
});

test('an EDF without a preferred name gives actionable feedback', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.route('**/config.json', route => route.fulfill({ json: { manifestUrl: 'https://packs.example.test/manifest.json' } }));
  await page.route('https://packs.example.test/manifest.json', route => route.fulfill({ json: { format: 'semantic-harmonization-minilm-v1', vocabularies: [] } }));
  await page.goto('http://127.0.0.1:4173');
  await expect(page.locator('#run-retrieval')).toBeEnabled();
  await page.locator('#edf-yaml').fill('definition: Migraine');
  await page.locator('#run-retrieval').click();
  await expect(page.locator('#notice')).toContainText('preferred_name');
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
});
