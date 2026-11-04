// Integration check against real vectors and the selected embedding provider.
// Set PREVIEW_URL for the deployed site or a local preview, and optionally
// EMBEDDING_PROVIDER=openai. No access token is needed for the public service.
import { chromium } from '@playwright/test';

const browser = await chromium.launch({ channel: 'chrome', headless: true });
const page = await browser.newPage();
page.on('pageerror', error => process.stderr.write(`Browser: ${error.message}\n`));
page.on('requestfailed', request => process.stderr.write(`Request failed: ${new URL(request.url()).hostname}: ${request.failure()?.errorText}\n`));
try {
  if (process.env.MANIFEST_URL || process.env.API_URL) {
    await page.route('**/config.json', route => route.fulfill({ json: {
      manifestUrl: process.env.MANIFEST_URL || 'http://127.0.0.1:8766/manifest.json',
      apiBaseUrl: process.env.API_URL || '',
    } }));
  }
  await page.goto(process.env.PREVIEW_URL || 'http://127.0.0.1:4173');
  await page.locator('#run-retrieval').waitFor({ state: 'visible' });
  await page.waitForFunction(() => !document.getElementById('run-retrieval').disabled);
  if (process.env.EMBEDDING_PROVIDER === 'openai') {
    await page.locator('#embedding-provider').selectOption('openai');
    await page.waitForFunction(() => !document.getElementById('run-retrieval').disabled);
  }
  if (process.env.VOCABULARIES !== 'all') for (const input of await page.locator('[name=vocabulary]').all()) {
    if (await input.inputValue() !== 'ICPC') await input.uncheck();
  }
  await page.locator('#edf-yaml').fill('preferred_name: Diabetes mellitus');
  await page.locator('#run-retrieval').click();
  const started = Date.now();
  while (Date.now() - started < 180_000) {
    const summary = await page.locator('#retrieval-summary').textContent();
    if (summary.includes('Ranked')) {
      console.log(summary);
      console.log((await page.locator('#retrieval-table tbody tr').first().innerText()).replaceAll('\n', ' | '));
      if (!(await page.locator('#retrieval-table tbody tr').first().innerText()).toLowerCase().includes('diabet')) throw Error('Diabetes query did not rank a diabetes code first.');
      await page.screenshot({ path: '/tmp/semantic-harmonization-real-minilm.png' });
      process.exitCode = 0;
      break;
    }
    const notice = await page.locator('#notice').textContent();
    if (notice) throw Error(notice);
    console.log(await page.locator('#progress-text').textContent());
    await new Promise(resolve => setTimeout(resolve, 3000));
  }
  if (!(await page.locator('#retrieval-summary').textContent()).includes('Ranked')) throw Error('Real MiniLM inference timed out.');
} finally { await browser.close(); }
