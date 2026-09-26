export default defineNuxtPlugin(() => {
  if (import.meta.dev || !("serviceWorker" in navigator)) return;
  const push = useShelfPush();
  navigator.serviceWorker.ready.then(() => { void push.refresh(); });
  navigator.serviceWorker.register("/sw.js", { updateViaCache: "none" }).catch((error) => {
    console.warn("Не удалось включить офлайн-режим", error);
  });
});
