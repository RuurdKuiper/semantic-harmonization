import { test, expect } from '@playwright/test';
import { createHash } from 'node:crypto';
import { gzipSync } from 'node:zlib';

for (const protectedPack of [false, true]) test(`browser retrieval, caching, retry, review and export (${protectedPack ? 'private' : 'public'} packs)`, async ({ page, context }) => {
  const count = 65;
  const packBase = protectedPack ? 'https://api.example.test/api/assets/' : 'https://packs.example.test/';
  const metadata = gzipSync(JSON.stringify(Array.from({ length: count }, (_, i) => [`C${String(i).padStart(3, '0')}`, `Migraine candidate ${i}`])));
  const vectors = new Float32Array(count * 384);
  for (let i = 0; i < count; i++) vectors[i * 384] = 1;
  const statistics = gzipSync(JSON.stringify({ count, avgdl: 3, averageIdf: 1, frequencies: { migraine: count, candidate: count }, k1: 1.5, b: .75, epsilon: .25 }));
  const vectorBytes = Buffer.from(vectors.buffer);
  const asset = (url, bytes) => ({ url, bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex') });
  const pack = { format: 'semantic-harmonization-minilm-v1', dimensions: 384, model: 'Xenova/all-MiniLM-L6-v2', dtype: 'fp32', maxLength: 256,
    vocabularies: [{ id: 'ICPC', version: 'test-v1', count, bytes: vectorBytes.length + metadata.length + statistics.length,
      lexical: asset('lexical.json.gz', statistics), chunks: [{ start: 0, count, rows: asset('rows.json.gz', metadata), vectors: asset('vectors.f32', vectorBytes) }] }] };
  let vectorDownloads = 0, classificationCalls = 0, reviewCalls = 0;
  const headers = { 'Access-Control-Allow-Origin': '*', 'Content-Type': 'application/json' };
  // Intercept the ONNX module only in this test. Production uses the real MiniLM runtime.
  await context.route(/\/assets\/minilm-runtime[^/]*\.js$/, route => route.fulfill({ contentType: 'application/javascript', body: 'export async function createMiniLMEncoder(){ return async()=>({data:Float32Array.from({length:384},(_,i)=>i===0?1:0)}); }' }));
  await context.route(`${packBase}**`, async route => {
    if (route.request().method() === 'OPTIONS') return route.fulfill({ status: 204, headers: { ...headers, 'Access-Control-Allow-Headers': '*', 'Access-Control-Allow-Methods': 'GET' } });
    expect(route.request().headers().authorization).toBe(protectedPack ? 'Bearer test-token' : undefined);
    const name = new URL(route.request().url()).pathname.split('/').pop();
    if (name === 'manifest.json') return route.fulfill({ headers, body: JSON.stringify(pack) });
    if (name === 'vectors.f32') { vectorDownloads++; return route.fulfill({ headers, body: vectorBytes }); }
    if (name === 'rows.json.gz') return route.fulfill({ headers, body: metadata });
    if (name === 'lexical.json.gz') return route.fulfill({ headers, body: statistics });
    return route.abort();
  });
  await context.route('https://api.example.test/**', async route => {
    if (new URL(route.request().url()).pathname.startsWith('/api/assets/')) return route.fallback();
    if (route.request().method() === 'OPTIONS') return route.fulfill({ status: 204, headers: { ...headers, 'Access-Control-Allow-Headers': '*', 'Access-Control-Allow-Methods': 'POST' } });
    expect(route.request().headers().authorization).toBe('Bearer test-token');
    const body = route.request().postDataJSON();
    expect(body.edf.preferred_name).toBe('Migraine');
    const review = route.request().url().endsWith('/review');
    if (review) reviewCalls++; else classificationCalls++;
    if (!review && classificationCalls === 2) return route.fulfill({ status: 502, headers, body: JSON.stringify({ detail: 'Temporary model failure; retry to continue.' }) });
    const decisions = body.candidates.map(row => ({ ...row, label: review ? 'Narrow' : 'Exclude', confidence: review ? .9 : .2, explanation: review ? 'Explicit event diagnosis.' : 'Insufficient detail.' }));
    return route.fulfill({ headers, body: JSON.stringify({ decisions, provider: review ? 'openai' : 'jev', model: 'test-model' }) });
  });
  await page.goto('http://127.0.0.1:4173');
  await page.locator('#connections summary').click();
  await page.locator('#manifest-url').fill(`${packBase}manifest.json`);
  await page.locator('#api-url').fill('https://api.example.test');
  await page.locator('#api-token').fill('test-token');
  await page.locator('#save-connections').click();
  await expect(page.locator('[name=vocabulary][value=ICD10CM]')).toBeDisabled();
  await page.locator('#edf-yaml').fill('preferred_name: Migraine\ndefinition:\nsynonyms:');
  await page.locator('#run-retrieval').click();
  await expect(page.locator('#retrieval-summary')).toHaveText('Ranked 65 codes on your computer.');
  await expect(page.locator('#retrieval-table tbody tr')).toHaveCount(65);
  await page.locator('#run-retrieval').click();
  await expect(page.locator('#run-retrieval')).toBeEnabled();
  expect(vectorDownloads).toBe(1);
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
  await page.goto('http://127.0.0.1:4173');
  await page.locator('#edf-yaml').fill('definition: Migraine');
  await page.locator('#run-retrieval').click();
  await expect(page.locator('#notice')).toContainText('preferred_name');
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
});
