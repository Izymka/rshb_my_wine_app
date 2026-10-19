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
    const next = [...bySlug.values()].slice(-LIMIT);
    const serialized = JSON.stringify(next);
    // Аналоги вычисляются из этого кеша и снова попадают в remember через WineDetails.
    // Не меняем реактивное состояние без изменения данных, иначе возникает цикл обновлений.
    if (serialized === JSON.stringify(cards.value)) return;
    cards.value = next;
    try { sessionStorage.setItem(KEY, serialized); } catch { /* приватный режим */ }
  }
  const seen = (slug: string) => cards.value.find((c) => c.slug === slug);
  return { cards, remember, seen };
}
