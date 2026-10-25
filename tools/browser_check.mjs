// Optional UI smoke check: npm install --no-save --package-lock=false playwright@1.60.0
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
const modulePath = process.env.PLAYWRIGHT_MODULE;
const { chromium } = await import(modulePath && path.isAbsolute(modulePath) ? pathToFileURL(modulePath).href : modulePath || 'playwright');
const browser = await chromium.launch({ headless: true, args: ['--no-proxy-server'], ...(process.env.BROWSER_EXECUTABLE ? { executablePath: process.env.BROWSER_EXECUTABLE } : {}) });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1100 }, deviceScaleFactor: 1 });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  for (let attempt = 0; ; attempt++) {
    try { await page.goto(process.env.DEMO_URL || 'http://127.0.0.1:8765', { waitUntil: 'networkidle' }); break; }
    catch (error) { if (attempt >= 4) throw error; await new Promise(resolve => setTimeout(resolve, 1000)); }
  }
  await page.waitForSelector('body[data-ready="true"]', { timeout: 60000 });
  await page.getByRole('button', { name: '验证重放' }).click();
  await page.waitForFunction(() => document.getElementById('replay-result').textContent.includes('完全一致'));
  const record = await page.locator('#replay-json').textContent();
  await page.locator('#augment').click();
  await page.waitForFunction(() => !document.getElementById('augment').disabled);
  assert.equal(await page.locator('#replay-json').textContent(), record, 'fixed seed/sample must reproduce the same operation record');
  const expectedPixels = await page.locator('#result').evaluate(canvas => canvas.toDataURL());
  await page.locator('#replay-file').setInputFiles({ name: 'replay.json', mimeType: 'application/json', buffer: Buffer.from(record) });
  await page.waitForFunction(() => document.getElementById('replay-result').textContent === '已导入');
  assert.equal(await page.locator('#result').evaluate(canvas => canvas.toDataURL()), expectedPixels, 'imported replay preserves pixels');
  await page.locator('#replay-file').setInputFiles({ name: 'invalid.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify({ ...JSON.parse(record), version: 999 })) });
  await page.waitForSelector('#status.error');
  await page.waitForFunction(() => !document.getElementById('augment').disabled);
  await page.locator('#replay').click();
  await page.waitForFunction(() => document.getElementById('replay-result').textContent.includes('完全一致'));
  await page.locator('#sample-gallery').selectOption('chelsea_alpha');
  await page.waitForFunction(() => !document.getElementById('augment').disabled && document.getElementById('source-meta').textContent.includes('160'));
  await page.locator('#resize-strategy').selectOption('letterbox');
  await page.locator('#augment').click();
  await page.waitForFunction(() => !document.getElementById('augment').disabled && JSON.parse(document.getElementById('replay-json').textContent).version === 2);
  await page.locator('#replay').click();
  await page.waitForFunction(() => document.getElementById('replay-result').textContent.includes('完全一致'));
  fs.mkdirSync('artifacts', { recursive: true });
  await page.screenshot({ path: 'artifacts/moonaug-desktop.png', fullPage: true });
  await page.locator('#tab-warp').click();
  await page.locator('#register').click();
  await page.waitForSelector('body[data-registration="complete"]', { timeout: 60000 });
  const improvement = Number((await page.locator('#improvement').textContent()).replace('%', ''));
  assert.ok(improvement > 95, `registration should reduce loss by >95%, got ${improvement}`);
  await page.screenshot({ path: 'artifacts/moonwarp-desktop.png', fullPage: true });
  await page.locator('#warp-gallery').selectOption('astronaut');
  await page.waitForFunction(() => !document.getElementById('register').disabled && document.getElementById('status').textContent.includes('配准输入已更新'));
  await page.locator('#padding').selectOption('border');
  await page.locator('#multiscale').check();
  await page.locator('#register').click();
  await page.waitForSelector('body[data-registration="complete"]', { timeout: 60000 });
  const realReduction = Number((await page.locator('#improvement').textContent()).replace('%', ''));
  assert.ok(Number.isFinite(realReduction) && realReduction >= 0, 'multiscale real-photo registration retains the best result');
  await page.screenshot({ path: 'artifacts/moonwarp-real-photo.png', fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator('#tab-aug').click();
  await page.screenshot({ path: 'artifacts/moonaug-mobile.png', fullPage: true });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, 'mobile viewport should not overflow horizontally');
  assert.deepEqual(errors, [], 'no uncaught browser errors');
  console.log(JSON.stringify({ passed: true, replay: true, replay_import: true, replay_v2: true, real_rgba_gallery: true, malformed_replay_recovery: true, deterministic: true, registration_loss_reduction_percent: improvement, real_photo_multiscale_reduction_percent: realReduction, mobile_overflow: false, browser_errors: errors }, null, 2));
} finally { await browser.close(); }
