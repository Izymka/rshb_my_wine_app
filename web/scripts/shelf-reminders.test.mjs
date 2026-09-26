import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createECDH } from 'node:crypto';
import { deviceKey, validateSubscription, validateReminders, fileStore, syncDevice, dispatchDue, queueLock } from '../server/utils/shelf-reminders.mjs';
const ec = createECDH('prime256v1'); ec.generateKeys();
const subscription = { endpoint: 'https://fcm.googleapis.com/fcm/send/test', keys: { auth: Buffer.alloc(16).toString('base64url'), p256dh: ec.getPublicKey().toString('base64url') } };
const reminder = { id: 'wine:1', name: 'Вино', slug: 'wine', dueAt: 1000 };
function memory() {
  const map = new Map();
  return { get: async k => structuredClone(map.get(k)), set: async (k, v) => { map.set(k, structuredClone(v)); }, keys: async () => [...map.keys()], remove: async k => { map.delete(k); } };
}
test('subscription validation blocks arbitrary URLs and malformed requests', () => {
  assert.deepEqual(validateSubscription(subscription), subscription);
  for (const endpoint of ['http://fcm.googleapis.com/x', 'https://127.0.0.1/x', 'https://fcm.googleapis.com.evil.test/x', 'https://user@web.push.apple.com/x']) assert.throws(() => validateSubscription({ ...subscription, endpoint }));
  assert.throws(() => deviceKey('../bad'));
  assert.throws(() => validateReminders([reminder, reminder]));
  assert.throws(() => validateReminders([{ ...reminder, dueAt: Infinity }]));
});
test('queue persists across process/store recreation and sends once across syncs', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'wine-reminders-'));
  try {
    const key = deviceKey('a'.repeat(64));
    await syncDevice(fileStore(dir), key, subscription, [reminder], 1000);
    const sent = [];
    const store = fileStore(dir);
    await dispatchDue(store, async (_s, p) => { sent.push(p); }, 62000);
    assert.equal(sent.length, 1);
    assert.equal(sent[0].url, '/wines/wine#review');
    await syncDevice(store, key, subscription, [reminder], 63000);
    await dispatchDue(store, async (_s, p) => { sent.push(p); }, 200000);
    assert.equal(sent.length, 1);
  } finally { await rm(dir, { recursive: true, force: true }); }
});
test('rating/deletion cancels queued reminders before dispatch', async () => {
  const store = memory();
  await syncDevice(store, 'device', subscription, [reminder], 1000);
  await syncDevice(store, 'device', subscription, [], 2000);
  let calls = 0;
  await dispatchDue(store, async () => { calls++; }, 62000);
  assert.equal(calls, 0);
});
test('transient failures retry; expired subscriptions are removed', async () => {
  const store = memory();
  await syncDevice(store, 'device', subscription, [reminder], 1000);
  let calls = 0;
  await dispatchDue(store, async () => { calls++; throw Error('offline'); }, 62000);
  await dispatchDue(store, async () => { calls++; }, 63000);
  assert.equal(calls, 1);
  await dispatchDue(store, async () => { calls++; throw { statusCode: 410 }; }, 200000);
  assert.equal(calls, 2);
  assert.equal(await store.get('device'), undefined);
});
test('queue serialization preserves cancellation ordered after an in-flight operation', async () => {
  const store = memory();
  await Promise.all([
    queueLock(() => syncDevice(store, 'device', subscription, [reminder], 1000)),
    queueLock(() => syncDevice(store, 'device', subscription, [], 2000)),
  ]);
  assert.deepEqual((await store.get('device')).reminders, []);
});
