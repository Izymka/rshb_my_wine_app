// Мобильный веб-интерфейс сканера: карточка вина в стилистике портала «Своё Вино».
// Всё общение с ML-сервисом идёт через серверные маршруты Nuxt (/api/*), поэтому CORS сервису
// не нужен, а адрес сервиса задаётся одной переменной окружения NUXT_SCANNER_URL.
export default defineNuxtConfig({
  compatibilityDate: "2025-01-01",
  devtools: { enabled: false },
  ssr: true,
  css: ["~/assets/css/main.css"],
  runtimeConfig: {
    scannerUrl: process.env.NUXT_SCANNER_URL || "http://127.0.0.1:8080",
  },
  app: {
    head: {
      title: "Своё Вино — сканер этикеток",
      meta: [
        { name: "viewport", content: "width=device-width, initial-scale=1, viewport-fit=cover" },
        { name: "theme-color", content: "#8F3D42" },
        { name: "description", content: "Сфотографируйте этикетку — найдём карточку вина на платформе «Своё Вино»" },
      ],
      link: [
        { rel: "icon", type: "image/png", href: "/icon.png" },
        { rel: "preconnect", href: "https://fonts.googleapis.com" },
        { rel: "preconnect", href: "https://fonts.gstatic.com", crossorigin: "" },
        {
          rel: "stylesheet",
          href: "https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600&display=swap",
        },
      ],
      htmlAttrs: { lang: "ru" },
    },
  },
});
