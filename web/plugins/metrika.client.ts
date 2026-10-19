// Яндекс Метрика: счётчик грузится после монтирования, переходы внутри SPA отправляются как hit,
// вошедший пользователь — как UserID (внутренний id аккаунта, не телефон).
export default defineNuxtPlugin((nuxtApp) => {
  const metrika = useMetrika();
  const { user } = useAuth();
  const first = { url: location.href, referrer: document.referrer };
  let last = location.href.split('#')[0];
  nuxtApp.hook('page:finish', () => {
    // Заголовок страницы useHead ставит после отрисовки — берём его на следующем тике.
    setTimeout(() => {
      const url = location.href.split('#')[0];
      if (url === last) return;
      metrika.hit(url, { referer: last, title: document.title });
      last = url;
    });
  });
  watch(() => user.value?.id, (id) => { if (id) metrika.user(id); }, { immediate: true });
  nuxtApp.hook('app:mounted', () => {
    $fetch<{ id: number | null }>('/api/metrika/config', { timeout: 10000 })
      .then((config) => startMetrika(config.id, first))
      .catch(() => startMetrika(null, first));
  });
});
