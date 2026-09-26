export default defineNuxtPlugin(() => {
  if (import.meta.dev || !("serviceWorker" in navigator)) return;
  navigator.serviceWorker.register("/sw.js", { updateViaCache: "none" }).catch((error) => {
    console.warn("Не удалось включить офлайн-режим", error);
  });
});
