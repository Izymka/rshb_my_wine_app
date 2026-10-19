// Номер счётчика Яндекс Метрики для браузера (NUXT_YANDEX_METRIKA_ID); пусто — Метрика выключена.
export default defineEventHandler((event) => {
  setResponseHeader(event, 'Cache-Control', 'no-store');
  const id = Number(useRuntimeConfig(event).yandexMetrikaId);
  return { id: Number.isSafeInteger(id) && id > 0 ? id : null };
});
