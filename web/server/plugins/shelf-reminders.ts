import { dispatchDue, queueLock } from '../utils/shelf-reminders.mjs';
import { shelfPushEnabled, shelfPushStore, sendShelfPush } from '../utils/shelf-push';
export default defineNitroPlugin((app) => {
  if (import.meta.prerender || !shelfPushEnabled()) return;
  let running = false;
  async function tick() {
    if (running) return;
    running = true;
    try { await queueLock(() => dispatchDue(shelfPushStore(), sendShelfPush)); }
    catch { console.warn('Wine reminder queue unavailable'); }
    finally { running = false; }
  }
  const timer = setInterval(() => { void tick(); }, 30000);
  timer.unref();
  void tick();
  app.hooks.hook('close', () => { clearInterval(timer); });
});
