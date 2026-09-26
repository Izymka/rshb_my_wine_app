import webpush from 'web-push';
import { resolve } from 'node:path';
import { fileStore } from './shelf-reminders.mjs';
export function shelfPushConfig() {
  const c = useRuntimeConfig();
  return { publicKey: c.shelfPushPublicKey, privateKey: c.shelfPushPrivateKey, subject: c.shelfPushSubject };
}
export function shelfPushEnabled() {
  const c = shelfPushConfig();
  return Boolean(c.publicKey && c.privateKey && c.subject);
}
export function shelfPushStore() {
  return fileStore(resolve(useRuntimeConfig().shelfPushDirectory));
}
export async function sendShelfPush(subscription: object, payload: { tag: string }) {
  return webpush.sendNotification(subscription, JSON.stringify(payload), {
    vapidDetails: shelfPushConfig(), TTL: 86400, urgency: 'normal', timeout: 10000,
    topic: payload.tag.slice(0, 32),
  });
}
