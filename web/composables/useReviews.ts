// Отзывы и «Что говорят другие». Публикация берёт оценку, блюда и заметку с полки: сначала
// полка уходит в аккаунт (сервер принимает отзыв только на вино со своей полки), затем отзыв.
export interface CommunityReview {
  mine: boolean; liked: boolean; dishes: string[]; note: string; anonymous: boolean;
  name: string; avatarUrl: string | null; date: string;
}
export interface Community {
  added: number; reviewCount: number; likedPct: number | null;
  pairings: { dish: string; pct: number }[];
  reviews: CommunityReview[];
  mine: { liked: boolean; dishes: string[]; note: string; anonymous: boolean } | null;
  card: Record<string, unknown> | null;
}

const PENDING_KEY = 'wine-pending-review';

export function useCommunity(slug: () => string) {
  const data = useState<Community | null>(`community.${slug()}`, () => null);
  const all = ref(false);
  const loading = ref(false);
  async function refresh() {
    const { enabled } = useAuth();
    if (!enabled.value) return;
    loading.value = true;
    try {
      data.value = await $fetch<Community>(`/api/community/${encodeURIComponent(slug())}`, {
        query: all.value ? { all: '1' } : {}, timeout: 10000,
      });
    } catch { /* сообщество — дополнение к карточке, без него она работает */ }
    finally { loading.value = false; }
  }
  async function showAll() { all.value = true; await refresh(); }
  return { data, all, loading, refresh, showAll };
}

export function useReviews() {
  const shelf = useShelf();
  async function publish(slug: string, anonymous: boolean) {
    const entry = shelf.find(slug);
    if (!entry?.reaction) throw new Error('Сначала оцените вино');
    if (!(await shelf.syncNow())) throw new Error('Нет связи с сервером — отзыв сохранён на полке');
    await $fetch('/api/reviews', {
      method: 'POST',
      body: { slug, liked: entry.reaction === 'liked', dishes: entry.dishes, note: entry.note, anonymous },
    });
  }
  async function unpublish(slug: string) {
    await $fetch(`/api/reviews/${encodeURIComponent(slug)}`, { method: 'DELETE' });
  }
  // Публикация, отложенная до входа: шторка отзыва → вход → экран «Добро пожаловать».
  function setPending(slug: string, anonymous: boolean) {
    try { sessionStorage.setItem(PENDING_KEY, JSON.stringify({ slug, anonymous })); } catch {}
  }
  function takePending(): { slug: string; anonymous: boolean } | null {
    try {
      const raw = sessionStorage.getItem(PENDING_KEY);
      sessionStorage.removeItem(PENDING_KEY);
      const value = raw ? JSON.parse(raw) : null;
      return typeof value?.slug === 'string' ? { slug: value.slug, anonymous: value.anonymous === true } : null;
    } catch { return null; }
  }
  return { publish, unpublish, setPending, takePending };
}
