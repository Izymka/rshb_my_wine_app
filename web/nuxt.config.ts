// Мобильный веб-интерфейс сканера: карточка вина в стилистике портала «Своё Вино».
// Всё общение с ML-сервисом идёт через серверные маршруты Nuxt (/api/*), поэтому CORS сервису
// не нужен, а адрес сервиса задаётся одной переменной окружения NUXT_SCANNER_URL.
import { buildPwa } from "./scripts/build-pwa.mjs";
import { SPLASH_SCREENS } from "./scripts/splash-screens.mjs";

export default defineNuxtConfig({
  compatibilityDate: "2025-01-01",
  devtools: { enabled: false },
  ssr: true,
  nitro: { prerender: { routes: ["/", "/offline.html"] } },
  routeRules: { "/offline.html": { ssr: false } },
  hooks: {
    "nitro:build:public-assets": (nitro) => buildPwa(nitro.options.output.publicDir),
  },
  css: [
    "@phosphor-icons/web/regular",
    "@phosphor-icons/web/fill",
    "@phosphor-icons/web/bold",
    "~/assets/css/main.css",
    "~/assets/css/game.css",
  ],
  runtimeConfig: {
    scannerUrl: "http://127.0.0.1:8080",
    shelfPushPublicKey: "",
    shelfPushPrivateKey: "",
    shelfPushSubject: "",
    shelfPushDirectory: ".data/wine-reminders",
    // PostgreSQL аккаунтов, облачной полки и отзывов (NUXT_DATABASE_URL). Пусто — без аккаунтов.
    databaseUrl: "",
    // Пока СМС-провайдера нет, код входа показывается на экране. "0" — не отдавать код в ответе.
    authDemoCodes: "1",
    // Номер счётчика Яндекс Метрики (NUXT_YANDEX_METRIKA_ID). Пусто — Метрика не подключается.
    yandexMetrikaId: "",
    // Множитель игровых таймеров «Моей винодельни» (NUXT_GAME_SPEED): 60 — час роста за минуту.
    gameSpeed: "1",
  },
  app: {
    head: {
      title: "Своё Вино — сканер этикеток",
      meta: [
        { name: "viewport", content: "width=device-width, initial-scale=1, viewport-fit=cover" },
        { name: "theme-color", content: "#8F3D42" },
        { name: "apple-mobile-web-app-capable", content: "yes" },
        { name: "mobile-web-app-capable", content: "yes" },
        { name: "apple-mobile-web-app-status-bar-style", content: "default" },
        { name: "apple-mobile-web-app-title", content: "Своё Вино" },
        { name: "description", content: "Сфотографируйте этикетку — найдём карточку вина на платформе «Своё Вино»" },
      ],
      link: [
        { rel: "manifest", href: "/manifest.webmanifest" },
        { rel: "apple-touch-icon", href: "/apple-touch-icon.png" },
        { rel: "icon", type: "image/svg+xml", href: "/icon.svg" },
        { rel: "icon", type: "image/png", href: "/icon.png" },
        ...SPLASH_SCREENS.map(([width, height, ratio]) => ({
          rel: "apple-touch-startup-image",
          href: `/splash/${width * ratio}x${height * ratio}.png`,
          media: `(device-width: ${width}px) and (device-height: ${height}px) and (-webkit-device-pixel-ratio: ${ratio}) and (orientation: portrait)`,
        })),
        { rel: "preconnect", href: "https://fonts.googleapis.com" },
        { rel: "preconnect", href: "https://fonts.gstatic.com", crossorigin: "" },
        {
          rel: "stylesheet",
          href: "https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600&display=swap",
        },
      ],
      htmlAttrs: { lang: "ru" },
    },
  },
});
