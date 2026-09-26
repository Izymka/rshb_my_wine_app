import { SHELF_KEY } from '~/utils/shelf.mjs';
export default defineNuxtPlugin((app) => {
  const shelf = useShelf();
  const push = useShelfPush();
  app.hook('app:mounted', () => {
    shelf.load();
    void push.refresh();
    window.addEventListener('storage', e => {
      if (e.key === SHELF_KEY || e.key === null) { shelf.load(); void push.sync(); }
    });
    window.addEventListener('shelf-changed', () => { void push.sync(); });
    window.addEventListener('online', () => { void push.refresh(); });
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') { shelf.load(); void push.refresh(); }
    });
    // Retry unsent offline edits without requiring a reload.
    window.setInterval(() => { if (document.visibilityState === 'visible') void push.sync(); }, 60000);
  });
});
