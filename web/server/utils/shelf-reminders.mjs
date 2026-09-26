import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readFile, readdir, rename, writeFile, unlink } from 'node:fs/promises';
import { join } from 'node:path';
const DAY = 86400000;
export function deviceKey(token) {
  if (typeof token !== 'string' || !/^[a-f0-9]{64}$/.test(token)) throw Error('Invalid device token');
  return createHash('sha256').update(token).digest('hex');
}
export function validateSubscription(subscription) {
  if (!subscription || typeof subscription.endpoint !== 'string' || subscription.endpoint.length > 4096) throw Error('Invalid subscription');
  const url = new URL(subscription.endpoint);
  // Only browser push services; never let a submitted endpoint access an arbitrary host.
  const host = url.hostname;
  const allowed = host === 'fcm.googleapis.com' || host === 'updates.push.services.mozilla.com'
    || host.endsWith('.push.services.mozilla.com') || host === 'web.push.apple.com'
    || host.endsWith('.push.apple.com') || host === 'wns.windows.com' || host.endsWith('.notify.windows.com');
  if (!allowed || url.protocol !== 'https:' || url.username || url.password || url.port || url.hash) throw Error('Unsupported push service');
  const { auth, p256dh } = subscription.keys || {};
  if (typeof auth !== 'string' || !/^[\w-]{22}={0,2}$/.test(auth)
    || typeof p256dh !== 'string' || !/^[\w-]{87}={0,2}$/.test(p256dh)) throw Error('Invalid subscription keys');
  if (Buffer.from(p256dh, 'base64url')[0] !== 4) throw Error('Invalid public key');
  return { endpoint: url.href, keys: { auth, p256dh } };
}
export function validateReminders(reminders, now = Date.now()) {
  if (!Array.isArray(reminders) || reminders.length > 500) throw Error('Invalid reminders');
  const ids = new Set();
  return reminders.map(r => {
    if (!r || typeof r.id !== 'string' || r.id.length > 600 || !r.id || ids.has(r.id)
      || typeof r.slug !== 'string' || !r.slug || r.slug.length > 500
      || typeof r.name !== 'string' || !r.name || r.name.length > 500
      || !Number.isFinite(r.dueAt) || r.dueAt > now + 366 * DAY || r.dueAt < 0) throw Error('Invalid reminder');
    ids.add(r.id);
    return { id: r.id, slug: r.slug, name: r.name, dueAt: r.dueAt };
  });
}
export function fileStore(directory) {
  return {
    async keys() { await mkdir(directory, { recursive: true, mode: 0o700 }); return (await readdir(directory)).filter(f => /^[a-f0-9]{64}\.json$/.test(f)).map(f => f.slice(0, -5)); },
    async get(key) { try { return JSON.parse(await readFile(join(directory, `${key}.json`), 'utf8')); } catch (e) { if (e.code === 'ENOENT') return null; throw e; } },
    async set(key, value) {
      await mkdir(directory, { recursive: true, mode: 0o700 });
      const path = join(directory, `${key}.json`);
      const temp = `${path}.${randomUUID()}.tmp`;
      await writeFile(temp, JSON.stringify(value), { mode: 0o600 });
      await rename(temp, path);
    },
    async remove(key) { await unlink(join(directory, `${key}.json`)).catch(e => { if (e.code !== 'ENOENT') throw e; }); },
  };
}
// One persistent Node process owns the queue; serialize sync and delivery to avoid lost cancellations.
let pending = Promise.resolve();
export function queueLock(fn) {
  const result = pending.then(fn);
  pending = result.catch(() => {});
  return result;
}
export async function syncDevice(store, key, subscription, reminders, now = Date.now()) {
  const previous = await store.get(key);
  const old = new Map((previous?.reminders || []).map(r => [r.id, r]));
  const next = reminders.map(r => {
    const existing = old.get(r.id);
    return { ...r, sentAt: existing?.sentAt || null, attempts: existing?.attempts || 0,
      nextAttemptAt: existing?.nextAttemptAt || Math.max(r.dueAt, now + 60000) };
  });
  await store.set(key, { subscription, reminders: next, updatedAt: now });
  return next.filter(r => !r.sentAt && r.attempts < 6 && r.dueAt >= now - 30 * DAY).length;
}
export async function dispatchDue(store, send, now = Date.now()) {
  for (const key of await store.keys()) {
    const device = await store.get(key);
    if (!device) continue;
    if (device.updatedAt < now - 90 * DAY) { await store.remove(key); continue; }
    let expired = false;
    for (const r of device.reminders) {
      if (r.sentAt || r.attempts >= 6 || r.dueAt > now || r.nextAttemptAt > now || r.dueAt < now - 30 * DAY) continue;
      r.attempts += 1;
      r.nextAttemptAt = now + Math.min(DAY, 60000 * 2 ** r.attempts);
      await store.set(key, device); // Persist retry state even if the process stops during delivery.
      try {
        await send(device.subscription, { title: 'Как вам вино?', body: `Уже пробовали «${r.name}»? Сохраните оценку и сочетание с блюдом.`,
          url: `/wines/${encodeURIComponent(r.slug)}#review`, tag: `wine-review-${createHash('sha256').update(r.id).digest('hex').slice(0, 24)}` });
        r.sentAt = now;
      } catch (error) {
        if (error.statusCode === 404 || error.statusCode === 410) { await store.remove(key); expired = true; break; }
        // No endpoint, token or tasting details in logs.
        console.warn('Wine reminder delivery failed', Number(error.statusCode) || 'network');
      }
      await store.set(key, device);
    }
    if (expired) continue;
  }
}
