import { deviceKey, validateSubscription, validateReminders, queueLock, syncDevice } from '../../utils/shelf-reminders.mjs';
import { shelfPushEnabled, shelfPushStore } from '../../utils/shelf-push';
const requests = new Map<string, { count: number; until: number }>();
export default defineEventHandler(async (event) => {
  setResponseHeader(event, 'Cache-Control', 'no-store');
  const origin = getRequestHeader(event, 'origin');
  if (origin && origin !== getRequestURL(event).origin) throw createError({ statusCode: 403 });
  if (!getRequestHeader(event, 'content-type')?.startsWith('application/json')) throw createError({ statusCode: 415 });
  const ip = getRequestIP(event) || 'unknown';
  const now = Date.now();
  for (const [key, value] of requests) if (value.until < now) requests.delete(key);
  const rate = requests.get(ip) || { count: 0, until: now + 60000 };
  rate.count++;
  if (rate.count > 60 || requests.size > 10000) throw createError({ statusCode: 429 });
  requests.set(ip, rate);
  const declared = Number(getRequestHeader(event, 'content-length') || 0);
  if (declared > 262144) throw createError({ statusCode: 413 });
  let body, key;
  try {
    key = deviceKey(getRequestHeader(event, 'authorization')?.replace(/^Bearer /, ''));
    const raw = await readRawBody(event);
    if (!raw || Buffer.byteLength(raw) > 262144) throw Error('Invalid body');
    body = JSON.parse(raw);
  } catch { throw createError({ statusCode: 400, statusMessage: 'Invalid reminder request' }); }
  if (body.disable === true) {
    await queueLock(() => shelfPushStore().remove(key));
    return { scheduled: 0 };
  }
  if (!shelfPushEnabled()) throw createError({ statusCode: 503, statusMessage: 'Push is not configured' });
  let subscription, reminders;
  try { subscription = validateSubscription(body.subscription); reminders = validateReminders(body.reminders); }
  catch { throw createError({ statusCode: 400, statusMessage: 'Invalid reminders or subscription' }); }
  const scheduled = await queueLock(() => syncDevice(shelfPushStore(), key, subscription, reminders));
  return { scheduled };
});
