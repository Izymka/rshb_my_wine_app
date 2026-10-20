// «Игровые уведомления» (макет 1m): подписка этого устройства на пуши «Моей винодельни».
// Подписка браузера общая с напоминаниями полки, поэтому выключение здесь не отписывает браузер,
// если напоминания полки ещё включены (и наоборот — см. useShelfPush.disable).
export const GAME_PUSH_KEY = "wine-game:push-enabled";

export function useGamePush() {
  const enabled = useState("game.push.enabled", () => false);
  const available = useState("game.push.available", () => false);
  const busy = useState("game.push.busy", () => false);
  const message = useState("game.push.message", () => "");
  const publicKey = useState<string | null>("game.push.key", () => null);
  const { goal } = useMetrika();

  const supported = () => "serviceWorker" in navigator && "PushManager" in window && "Notification" in window && window.isSecureContext;
  async function subscription() {
    return await (await navigator.serviceWorker.getRegistration())?.pushManager.getSubscription() || null;
  }
  async function post(body: object) {
    return await $fetch("/api/game/push", { method: "POST", body, timeout: 10000 });
  }

  // При открытии профиля: показать честное состояние и освежить подписку на сервере.
  async function refresh() {
    if (!supported()) { message.value = "Нужен браузер с пушами; на iPhone — приложение на экране «Домой»"; return; }
    try {
      const config = await $fetch<{ publicKey: string | null }>("/api/shelf/push-config", { timeout: 10000 });
      publicKey.value = config.publicKey;
      available.value = Boolean(config.publicKey) && Boolean(await navigator.serviceWorker.getRegistration());
      if (!available.value) { message.value = "Пуши пока недоступны"; return; }
      if (localStorage.getItem(GAME_PUSH_KEY) !== "on") { enabled.value = false; return; }
      const sub = await subscription();
      if (Notification.permission !== "granted" || !sub) {
        localStorage.removeItem(GAME_PUSH_KEY);
        enabled.value = false;
        message.value = Notification.permission === "denied" ? "Уведомления запрещены в настройках браузера" : "Подписка истекла — включите заново";
        return;
      }
      enabled.value = true;
      message.value = "";
      await post({ subscription: sub.toJSON() });
    } catch { message.value = enabled.value ? "" : "Нет связи для настройки пушей"; }
  }

  async function enable() {
    if (!publicKey.value || busy.value) return;
    busy.value = true;
    try {
      // Разрешение спрашиваем прямо из нажатия (этого требует Safari).
      const permission = await Notification.requestPermission();
      if (permission !== "granted") { goal("push_denied", { permission, type: "game" }); message.value = "Пуши не разрешены — это можно изменить в настройках браузера"; return; }
      const registration = await navigator.serviceWorker.getRegistration();
      if (!registration?.active) throw Error("Worker not ready");
      const key = Uint8Array.from(atob(publicKey.value.replace(/-/g, "+").replace(/_/g, "/")), (c) => c.charCodeAt(0));
      const sub = await registration.pushManager.getSubscription() || await registration.pushManager.subscribe({ userVisibleOnly: true, applicationServerKey: key });
      await post({ subscription: sub.toJSON() });
      localStorage.setItem(GAME_PUSH_KEY, "on");
      enabled.value = true;
      message.value = "";
      goal("push_enabled", { type: "game" });
    } catch { goal("push_error", { action: "enable", type: "game" }); message.value = "Не удалось включить. На iPhone откройте приложение с экрана «Домой»"; }
    finally { busy.value = false; }
  }

  async function disable() {
    busy.value = true;
    try {
      const sub = await subscription();
      if (sub) await post({ disable: true, endpoint: sub.endpoint });
      localStorage.removeItem(GAME_PUSH_KEY);
      enabled.value = false;
      message.value = "";
      goal("push_disabled", { type: "game" });
      if (sub && localStorage.getItem("wine-shelf:push-enabled") !== "on") await sub.unsubscribe();
    } catch { message.value = "Не удалось отключить. Повторите при подключении к сети"; }
    finally { busy.value = false; }
  }

  return { enabled, available, busy, message, refresh, enable, disable };
}
