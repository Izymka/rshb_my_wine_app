// Карточки, которые пользователь видел, но не сохранял: ответ сканера и его аналоги. По ним
// страница /wines/:slug открывает аналог сразу, не дожидаясь каталога сервиса (useCatalog), и
// без сети. Живут до закрытия вкладки (sessionStorage).
import type { Card } from "~/composables/useScan";

const KEY = "wine-seen-cards";
const LIMIT = 60;
export type SeenCard = Card & { slug: string };

export function useSeenCards() {
  const cards = useState<SeenCard[]>("seen.cards", () => []);
  const loaded = useState("seen.loaded", () => false);
  if (import.meta.client && !loaded.value) {
    loaded.value = true;
    try { cards.value = JSON.parse(sessionStorage.getItem(KEY) || "[]"); } catch { cards.value = []; }
  }

  function remember(list: SeenCard[]) {
    const fresh = list.filter((c) => c?.slug && c.name);
    if (!fresh.length) return;
    // Полная карточка (с описанием) важнее урезанной карточки аналога: не затираем её.
    const bySlug = new Map(cards.value.map((c) => [c.slug, c]));
    for (const c of fresh) bySlug.set(c.slug, { ...c, ...bySlug.get(c.slug), ...(c.description ? c : {}) });
    cards.value = [...bySlug.values()].slice(-LIMIT);
    try { sessionStorage.setItem(KEY, JSON.stringify(cards.value)); } catch { /* приватный режим */ }
  }
  const seen = (slug: string) => cards.value.find((c) => c.slug === slug);
  return { cards, remember, seen };
}
