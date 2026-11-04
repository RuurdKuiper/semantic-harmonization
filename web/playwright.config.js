import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  timeout: 60_000,
  use: { browserName: 'chromium', channel: 'chrome', headless: true, viewport: { width: 1440, height: 1050 }, serviceWorkers: 'block' },
  webServer: { command: 'npm run preview -- --port 4173', url: 'http://127.0.0.1:4173', reuseExistingServer: !process.env.CI },
});
