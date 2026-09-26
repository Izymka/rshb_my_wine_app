// Run against a production preview. All scanner responses are fixtures; no ML service is needed.
import assert from 'node:assert/strict';
import { mkdir, readFile } from 'node:fs/promises';
import { createECDH, randomBytes } from 'node:crypto';
import { setTimeout as sleep } from 'node:timers/promises';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const browser = await chromium.launch({ headless: true, ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}) });
const base = process.env.SHELF_TEST_URL || 'http://127.0.0.1:3107';
const context = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 1 });
const pushCalls = [];
const ec = createECDH('prime256v1'); ec.generateKeys();
const fakeSubscription = { endpoint: 'https://fcm.googleapis.com/fcm/send/browser-test', keys: { p256dh: ec.getPublicKey().toString('base64url'), auth: randomBytes(16).toString('base64url') } };
// Mock only the browser push service; verify the application's scheduling and cancellation requests.
await context.addInitScript(subscription => {
  Object.defineProperty(Notification, 'permission', { get: () => 'granted' });
  Notification.requestPermission = async () => 'granted';
  const makeSubscription = () => ({ ...subscription, toJSON: () => subscription, unsubscribe: async () => { sessionStorage.removeItem('test-push'); return true; } });
  PushManager.prototype.getSubscription = async () => sessionStorage.getItem('test-push') ? makeSubscription() : null;
  PushManager.prototype.subscribe = async () => { sessionStorage.setItem('test-push', 'yes'); return makeSubscription(); };
}, fakeSubscription);
await context.route('**/api/shelf/push-config', route => route.fulfill({ json: { publicKey: fakeSubscription.keys.p256dh } }));
await context.route('**/api/shelf/reminders', route => { pushCalls.push(route.request().postDataJSON()); return route.fulfill({ json: { scheduled: route.request().postDataJSON().reminders?.length || 0 } }); });
async function waitForPush(predicate) {
  for (let i = 0; i < 100; i++) { if (predicate(pushCalls.at(-1))) return; await sleep(100); }
  assert.fail('Expected push queue update');
}
const page = await context.newPage();
const errors = [];
page.on('pageerror', e => errors.push(e.message));
page.on('console', msg => { if (msg.type() === 'warning' && /hydration/i.test(msg.text())) errors.push(msg.text()); });
const image = await readFile(new URL('../public/icon.png', import.meta.url));
const card = { name: 'Мерло. Южный берег', winery: 'Винодельня у моря', category: 'Красное', grapes: 'Мерло', region: 'Кубань', description: 'Сухое красное вино с ягодным ароматом.' };
const fixture = { request_id: 'shelf-test', answered: true, item_id: 'test-merlot', card, candidates: [], vintage: { year: 2023, source: 'text', ask: false }, confidence: { top1: .99, top5: 1 }, timings_ms: { total: 1200 }, analogues: [{ slug: 'test-analogue', name: 'Мерло. Резерв', winery: 'Другая винодельня', category: 'Красное', grapes: 'Мерло', region: 'Кубань', score: .8 }] };
await context.route('**/api/scan', route => route.fulfill({ json: fixture }));
await context.route('**/api/sommelier/status', route => route.fulfill({ json: { available: false } }));
await context.route('**/api/catalog/image/**', route => route.fulfill({ contentType: 'image/png', body: image }));
const screenshotDir = process.env.SHELF_SCREENSHOTS;
async function screenshot(name) { if (screenshotDir) { await mkdir(screenshotDir, { recursive: true }); await page.screenshot({ path: `${screenshotDir}/${name}.png`, fullPage: true }); } }
async function visible(locator) { await locator.waitFor({ state: 'visible' }); }
async function scan() {
  await page.goto(base);
  await page.locator('input[type=file]').first().setInputFiles({ name: 'wine.png', mimeType: 'image/png', buffer: image });
  await visible(page.getByRole('heading', { name: card.name, exact: true }));
}
try {
  await page.goto(`${base}/wines`);
  await visible(page.getByRole('heading', { name: 'Здесь будут ваши вина' }));
  await scan();
  await page.getByRole('button', { name: 'Сохранить в мои вина', exact: true }).click();
  await visible(page.getByText('✓ Уже на вашей полке', { exact: true }));
  let stored = await page.evaluate(() => JSON.parse(localStorage.getItem('wine-shelf:v1')));
  assert.equal(stored.entries.length, 1);
  assert.equal(stored.entries[0].remindAt - stored.entries[0].addedAt, 3 * 86400000);
  const addedAt = stored.entries[0].addedAt;
  await page.evaluate(async () => { await navigator.serviceWorker.ready; });
  await page.getByRole('button', { name: 'Включить пуш-напоминания', exact: true }).click();
  await waitForPush(body => body?.reminders?.length === 1);
  assert.equal(pushCalls.at(-1).reminders[0].dueAt, addedAt + 3 * 86400000);
  await page.getByRole('button', { name: 'Понравилось', exact: true }).click();
  await waitForPush(body => body?.reminders?.length === 0);
  await page.getByRole('button', { name: 'Сыры', exact: true }).click();
  await page.getByLabel('Короткая заметка', { exact: true }).fill('Понравилось с выдержанным сыром. Запомнить к ужину.');
  await page.getByRole('button', { name: 'Сохранить заметку' }).click();
  await page.getByRole('link', { name: 'Открыть сохранённую карточку →' }).click();
  await visible(page.getByRole('heading', { name: 'Похожее на понравившееся' }));
  await screenshot('shelf-wine');
  await page.reload();
  await visible(page.getByRole('heading', { name: card.name, exact: true }));
  assert.equal(await page.getByRole('button', { name: 'Сыры', exact: true }).getAttribute('aria-pressed'), 'true');
  assert.match(await page.getByLabel('Короткая заметка', { exact: true }).inputValue(), /сыром/);
  await page.getByRole('link', { name: /Мерло. Резерв/ }).click();
  await visible(page.getByRole('heading', { name: 'Мерло. Резерв', exact: true }));
  await page.getByRole('button', { name: 'Сохранить в мои вина', exact: true }).click();
  await page.getByRole('button', { name: 'Есть дома', exact: true }).click();
  await page.goto(`${base}/wines`);
  await visible(page.getByRole('link', { name: /Мерло. Резерв/ }));
  await screenshot('shelf-list');
  await page.getByRole('button', { name: 'Отключить пуши', exact: true }).click();
  await waitForPush(body => body?.disable === true);
  await page.getByRole('button', { name: 'Есть дома', exact: true }).click();
  assert.equal(await page.locator('.shelf-item').count(), 1);
  await scan();
  await visible(page.getByText('✓ Уже на вашей полке', { exact: true }));
  await visible(page.getByText('Ваша оценка: Понравилось', { exact: true }));
  stored = await page.evaluate(() => JSON.parse(localStorage.getItem('wine-shelf:v1')));
  assert.equal(stored.entries.length, 2);
  assert.equal(stored.entries.find(e => e.slug === 'test-merlot').addedAt, addedAt);
  assert.equal(stored.entries.find(e => e.slug === 'test-merlot').remindAt, null);
  await screenshot('shelf-rescan');
  // Real SW installation + full offline reload into a dynamic route.
  await page.evaluate(async () => { await Promise.race([navigator.serviceWorker.ready, new Promise((_, reject) => setTimeout(() => reject(Error("Worker installation timed out")), 15000))]); });
  await page.goto(`${base}/wines/test-merlot`);
  await visible(page.getByRole('heading', { name: card.name, exact: true }));
  await context.setOffline(true);
  await page.reload();
  await visible(page.getByRole('heading', { name: card.name, exact: true }));
  assert.match(await page.getByLabel('Короткая заметка', { exact: true }).inputValue(), /сыром/);
  await context.setOffline(false);
  await page.getByRole('button', { name: 'Удалить с полки', exact: true }).click();
  await page.getByRole('button', { name: 'Удалить', exact: true }).click();
  await page.waitForURL(`${base}/wines`);
  assert.equal(await page.locator('.shelf-item').count(), 1);
  await page.goto(`${base}/wines/test-merlot`);
  await visible(page.getByRole('button', { name: 'Сохранить в мои вина', exact: true }));
  await page.goto(`${base}/wines/missing-wine`);
  await visible(page.getByRole('heading', { name: 'Вина нет на этой полке' }));
  assert.deepEqual(errors, []);
  console.log('Shelf browser: save, review, dishes, note, reload, recommendations, filter, rescan, offline, delete, push subscribe/schedule/cancel/disable passed');
} catch (error) { console.error('Browser errors:', errors); await screenshot('shelf-failure'); throw error; } finally { await browser.close(); }
