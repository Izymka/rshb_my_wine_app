import { remindersFor } from '~/utils/shelf.mjs';
const TOKEN_KEY = 'wine-shelf:push-token';
const ENABLED_KEY = 'wine-shelf:push-enabled';
let pending: Promise<void> = Promise.resolve();
export function useShelfPush() {
  const { entries, error: shelfError } = useShelf();
  const enabled = useState('shelf.push.enabled', () => false);
  const available = useState('shelf.push.available', () => false);
  const busy = useState('shelf.push.busy', () => false);
  const message = useState('shelf.push.message', () => 'Напоминание появится на полке. Для пуша включите уведомления.');
  const publicKey = useState<string | null>('shelf.push.key', () => null);
  function token() {
    let value = localStorage.getItem(TOKEN_KEY);
    if (!value) { value = [...crypto.getRandomValues(new Uint8Array(32))].map(x => x.toString(16).padStart(2, '0')).join(''); localStorage.setItem(TOKEN_KEY, value); }
    return value;
  }
  async function post(body: object) {
    return await $fetch('/api/shelf/reminders', { method: 'POST', headers: { Authorization: `Bearer ${token()}` }, body, timeout: 10000 });
  }
  function sync() {
    pending = pending.then(async () => {
      try {
        const intent = localStorage.getItem(ENABLED_KEY);
        if (intent === 'off') {
          if (localStorage.getItem(TOKEN_KEY)) await post({ disable: true });
          localStorage.removeItem(ENABLED_KEY);
          enabled.value = false;
          message.value = 'Пуши отключены. Напоминания остаются на полке.';
          return;
        }
        enabled.value = intent === 'on';
        if (!enabled.value || !navigator.onLine || shelfError.value) return;
        if (Notification.permission !== 'granted') {
          await post({ disable: true });
          enabled.value = false;
          localStorage.removeItem(ENABLED_KEY);
          message.value = 'Уведомления запрещены в настройках браузера. Напоминание останется на полке.';
          return;
        }
        const registration = await navigator.serviceWorker.getRegistration();
        const subscription = await registration?.pushManager.getSubscription();
        if (!subscription) { message.value = 'Подписка истекла. Включите уведомления повторно.'; enabled.value = false; return; }
        await post({ subscription: subscription.toJSON(), reminders: remindersFor(entries.value) });
        message.value = 'Пуши включены. После оценки напоминание отменяется.';
      } catch { message.value = 'Изменения пуш-напоминаний ещё не отправлены. Повторим при подключении к сети.'; }
    });
    return pending;
  }
  async function refresh() {
    if (!('serviceWorker' in navigator) || !('PushManager' in window) || !('Notification' in window) || !window.isSecureContext) {
      message.value = 'Напоминание появится на полке. Для пушей нужен поддерживаемый браузер; на iPhone добавьте приложение на экран «Домой».';
      return;
    }
    try {
      const config = await $fetch('/api/shelf/push-config', { timeout: 10000 });
      publicKey.value = config.publicKey;
      available.value = Boolean(config.publicKey) && Boolean(await navigator.serviceWorker.getRegistration());
      if (!available.value) message.value = 'Пуши пока недоступны. Напоминание появится на полке.';
      await sync();
    } catch { message.value = 'Нет связи для настройки пушей. Напоминания сохраняются на полке.'; }
  }
  async function enable() {
    if (!publicKey.value || busy.value) return;
    busy.value = true;
    try {
      // Ask directly from the button gesture (required by Safari).
      const permission = await Notification.requestPermission();
      if (permission !== 'granted') { message.value = 'Пуши не разрешены. Напоминание останется на полке; разрешение можно изменить в настройках браузера.'; return; }
      const registration = await navigator.serviceWorker.getRegistration();
      if (!registration?.active) throw Error('Worker not ready');
      const key = Uint8Array.from(atob(publicKey.value.replace(/-/g, '+').replace(/_/g, '/')), c => c.charCodeAt(0));
      await registration.pushManager.getSubscription() || await registration.pushManager.subscribe({ userVisibleOnly: true, applicationServerKey: key });
      localStorage.setItem(ENABLED_KEY, 'on');
      enabled.value = true;
      await sync();
    } catch { message.value = 'Не удалось включить пуши. Попробуйте ещё раз. На iPhone откройте приложение с экрана «Домой».'; }
    finally { busy.value = false; }
  }
  async function disable() {
    busy.value = true;
    try {
      localStorage.setItem(ENABLED_KEY, 'off');
      enabled.value = false;
      const registration = await navigator.serviceWorker.getRegistration();
      await (await registration?.pushManager.getSubscription())?.unsubscribe();
      await sync();
    } catch { message.value = 'Не удалось отключить уведомления. Повторите при подключении к сети.'; }
    finally { busy.value = false; }
  }
  return { enabled, available, busy, message, refresh, sync, enable, disable };
}
