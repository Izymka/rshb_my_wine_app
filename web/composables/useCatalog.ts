// Карточка вина из каталога сервиса (GET /catalog/:slug через прокси /api) — для ссылки на вино,
// открытой без сканирования, и живые сведения портала vino-svoe.ru: народный рейтинг, крепость,
// подача. Всё это дополнение: без сети или с выключенным порталом карточка работает как раньше.
import type { Card, Analogue } from "~/composables/useScan";

export interface PortalFacts {
  rating: number | null;
  alcohol: number | null;
  temperature: string | null;
  dishes: string[];
  url: string;
}
export interface CatalogCard { item_id: string; card: Card; analogues: Analogue[]; portal: PortalFacts | null }

export function useCatalog(slug: () => string) {
  // undefined — ещё не спрашивали, null — сервис не знает вино или не ответил.
  const data = useState<CatalogCard | null | undefined>(`catalog.${slug()}`, () => undefined);
  const loading = useState(`catalog.${slug()}.loading`, () => false);
  async function load() {
    if (data.value !== undefined || loading.value) return;
    loading.value = true;
    try {
      data.value = await $fetch<CatalogCard>(`/api/catalog/${encodeURIComponent(slug())}`, { timeout: 10000 });
    } catch { data.value = null; }
    finally { loading.value = false; }
  }
  if (import.meta.client) void load();
  return { data, loading, load };
}
