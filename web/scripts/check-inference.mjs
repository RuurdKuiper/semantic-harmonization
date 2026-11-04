// Optional integration check against real exported vectors and real browser MiniLM.
// Start the built preview on 4173 and the asset server on 8766 (origin 4173).
import { chromium } from '@playwright/test';

const browser = await chromium.launch({ channel: 'chrome', headless: true });
const page = await browser.newPage();
page.on('pageerror', error => process.stderr.write(`Browser: ${error.message}\n`));
page.on('requestfailed', request => process.stderr.write(`Request failed: ${new URL(request.url()).hostname}: ${request.failure()?.errorText}\n`));
try {
  await page.goto(process.env.PREVIEW_URL || 'http://127.0.0.1:4173');
  await page.locator('#connections summary').click();
  await page.locator('#manifest-url').fill(process.env.MANIFEST_URL || 'http://127.0.0.1:8766/manifest.json');
  if (process.env.API_URL) await page.locator('#api-url').fill(process.env.API_URL);
  if (process.env.PIPELINE_API_TOKEN) await page.locator('#api-token').fill(process.env.PIPELINE_API_TOKEN);
  await page.locator('#save-connections').click();
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
