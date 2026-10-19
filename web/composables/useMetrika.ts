// Яндекс Метрика: просмотры страниц, цели (reachGoal) и электронная коммерция по карточкам вин.
// Номер счётчика приходит с сервера (/api/metrika/config, переменная NUXT_YANDEX_METRIKA_ID):
// главная пререндерится, и публичный runtimeConfig застыл бы в ней на этапе сборки образа.
// Пока номер неизвестен, вызовы копятся в очереди; без номера счётчик не грузится вовсе.
type Ym = ((id: number, method: string, ...args: unknown[]) => void) & { a?: unknown[][]; l?: number };
type Product = { slug: string; card: { name?: unknown; winery?: unknown; category?: unknown; color?: unknown } };

declare global {
  interface Window { ym?: Ym; dataLayer?: unknown[] }
}

const TAG_URL = "https://mc.yandex.ru/metrika/tag.js";
// undefined — номер ещё не пришёл, null — Метрика выключена.
let counter: number | null | undefined;
let queue: [string, unknown[]][] = [];

function call(method: string, ...args: unknown[]) {
  if (!import.meta.client || counter === null) return;
  if (counter === undefined) {
    queue.push([method, args]);
    if (queue.length > 200) queue.shift();
    return;
  }
  window.ym?.(counter, method, ...args);
}

// Код установки счётчика из кабинета Метрики; url и referrer — первой открытой страницы,
// потому что номер приходит асинхронно и пользователь мог уже перейти дальше.
export function startMetrika(id: number | null, first: { url: string; referrer: string }) {
  if (!import.meta.client || counter !== undefined) return;
  counter = id;
  const pending = queue;
  queue = [];
  if (!id) return;
  window.dataLayer ||= [];
  if (!window.ym) {
    const ym: Ym = (...args) => { (ym.a ||= []).push(args); };
    ym.l = Date.now();
    window.ym = ym;
    const src = `${TAG_URL}?id=${id}`;
    if (![...document.scripts].some((s) => s.src === src)) {
      const script = document.createElement("script");
      script.async = true;
      script.src = src;
      document.head.appendChild(script);
    }
  }
  window.ym(id, "init", {
    ssr: true, webvisor: true, clickmap: true, ecommerce: "dataLayer",
    referrer: first.referrer, url: first.url, accurateTrackBounce: true, trackLinks: true,
  });
  for (const [method, args] of pending) window.ym(id, method, ...args);
}

function product({ slug, card }: Product) {
  return {
    id: slug,
    name: String(card.name || slug),
    brand: card.winery ? String(card.winery) : undefined,
    category: [card.category, card.color].filter(Boolean).join("/") || undefined,
  };
}

export function useMetrika() {
  return {
    // Цель с необязательными параметрами визита: goal("scan_start", { source: "camera" }).
    goal: (name: string, params?: Record<string, unknown>) => call("reachGoal", name, params),
    hit: (url: string, options: { referer: string; title: string }) => call("hit", url, options),
    user: (id: string) => call("setUserID", id),
    // Электронная коммерция: просмотр карточки, сохранение на полку и удаление с неё.
    ecommerce(action: "detail" | "add" | "remove", item: Product) {
      if (!import.meta.client || counter === null) return;
      (window.dataLayer ||= []).push({ ecommerce: { currencyCode: "RUB", [action]: { products: [product(item)] } } });
    },
  };
}
