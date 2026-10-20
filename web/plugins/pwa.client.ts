export default defineNuxtPlugin(() => {
  if (import.meta.dev || !("serviceWorker" in navigator)) return;
  const push = useShelfPush();
  const gamePush = useGamePush();
  // Профиль может открыться раньше, чем зарегистрируется worker: состояние пушей уточняем после.
  navigator.serviceWorker.ready.then(() => { void push.refresh(); void gamePush.refresh(); });
  navigator.serviceWorker.register("/sw.js", { updateViaCache: "none" }).catch((error) => {
    console.warn("Не удалось включить офлайн-режим", error);
  });
});
