// Integration test for Nitro routes with disposable keys; no real notifications are sent.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createECDH, randomBytes } from 'node:crypto';
import { mkdtemp, rm, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';
import webpush from 'web-push';
import { deviceKey } from '../server/utils/shelf-reminders.mjs';
const directory = await mkdtemp(join(tmpdir(), 'wine-api-test-'));
const keys = webpush.generateVAPIDKeys();
const port = process.env.SHELF_API_TEST_PORT || '3109';
const base = `http://127.0.0.1:${port}`;
const build = resolve(process.env.SHELF_BUILD_DIR || '.');
const server = spawn(process.execPath, [join(build, '.output/server/index.mjs')], {
  cwd: build, stdio: 'ignore', env: { ...process.env, HOST: '127.0.0.1', PORT: port,
    TG_NOTIFY_TOKEN: '', TG_NOTIFY_TARGET: '', NUXT_SCANNER_URL: 'http://127.0.0.1:1',
    NUXT_SHELF_PUSH_PUBLIC_KEY: keys.publicKey, NUXT_SHELF_PUSH_PRIVATE_KEY: keys.privateKey,
    NUXT_SHELF_PUSH_SUBJECT: 'mailto:test@example.invalid', NUXT_SHELF_PUSH_DIRECTORY: directory },
});
const token = randomBytes(32).toString('hex');
const ec = createECDH('prime256v1'); ec.generateKeys();
const subscription = { endpoint: 'https://fcm.googleapis.com/fcm/send/test', keys: { auth: randomBytes(16).toString('base64url'), p256dh: ec.getPublicKey().toString('base64url') } };
const reminder = { id: 'test:future', slug: 'test', name: 'Test', dueAt: Date.now() + 86400000 };
const post = (body, headers = {}) => fetch(`${base}/api/shelf/reminders`, { method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}`, ...headers }, body: JSON.stringify(body) });
try {
  let ready = false;
  for (let i = 0; i < 40; i++) {
    try { if ((await fetch(`${base}/api/shelf/push-config`)).ok) { ready = true; break; } } catch {}
    await sleep(200);
  }
  assert.ok(ready, 'test server starts');
  const config = await (await fetch(`${base}/api/shelf/push-config`)).json();
  assert.deepEqual(config, { publicKey: keys.publicKey });
  assert.equal((await post({ subscription, reminders: [reminder] }, { Origin: 'https://evil.test' })).status, 403);
  assert.equal((await post({ subscription, reminders: [reminder] }, { Authorization: 'Bearer wrong' })).status, 400);
  assert.equal((await post({ subscription: { ...subscription, endpoint: 'https://127.0.0.1/internal' }, reminders: [reminder] })).status, 400);
  let response = await post({ subscription, reminders: [reminder] });
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), { scheduled: 1 });
  let record = JSON.parse(await readFile(join(directory, `${deviceKey(token)}.json`), 'utf8'));
  assert.equal(record.reminders.length, 1);
  response = await post({ subscription, reminders: [] });
  assert.deepEqual(await response.json(), { scheduled: 0 });
  record = JSON.parse(await readFile(join(directory, `${deviceKey(token)}.json`), 'utf8'));
  assert.equal(record.reminders.length, 0);
  assert.equal((await post({ disable: true })).status, 200);
  await assert.rejects(readFile(join(directory, `${deviceKey(token)}.json`)), { code: 'ENOENT' });
  console.log('Shelf API: public config, isolation from scanner, authorization, validation, persistence, cancellation and disable passed');
} finally {
  server.kill('SIGTERM');
  await new Promise(resolve => { if (server.exitCode !== null) resolve(); else server.once('exit', resolve); });
  await rm(directory, { recursive: true, force: true });
}
