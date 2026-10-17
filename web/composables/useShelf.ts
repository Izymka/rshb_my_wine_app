import type { Card, Analogue } from './useScan';
import { SHELF_KEY, parseShelf, addWine, updateWine, mergeShelves } from '~/utils/shelf.mjs';

export interface ShelfWine {
  slug: string; card: Card; analogues: Analogue[]; addedAt: number; updatedAt: number;
  status: 'want' | 'home' | 'tried'; reaction: 'liked' | 'disliked' | null;
  dishes: string[]; note: string; remindAt: number | null;
}
interface Deleted { slug: string; at: number }

// Удаления, ещё не отправленные в аккаунт: без них вино вернулось бы с облачной копией.
const DELETED_KEY = 'wine-shelf:deleted';
let syncTimer: ReturnType<typeof setTimeout> | undefined;
let syncing: Promise<boolean> | null = null;

function readDeleted(): Deleted[] {
  try {
    const value = JSON.parse(localStorage.getItem(DELETED_KEY) || '[]');
    return Array.isArray(value) ? value.filter(d => typeof d?.slug === 'string' && Number.isFinite(d.at)) : [];
  } catch { return []; }
}
function writeDeleted(items: Deleted[]) {
  try {
    if (items.length) localStorage.setItem(DELETED_KEY, JSON.stringify(items.slice(-1000)));
    else localStorage.removeItem(DELETED_KEY);
  } catch {}
}

export function useShelf() {
  const entries = useState<ShelfWine[]>('shelf.entries', () => []);
  const ready = useState('shelf.ready', () => false);
  const error = useState('shelf.error', () => '');
  const synced = useState<number | null>('shelf.synced', () => null);
  const { user } = useAuth();

  function load() {
    try { entries.value = parseShelf(localStorage.getItem(SHELF_KEY)); error.value = ''; }
    catch { error.value = 'Не удалось прочитать полку. Данные не перезаписаны. Проверьте доступ к хранилищу браузера.'; }
    ready.value = true;
  }
  function write(next: ShelfWine[]) {
    const serialized = JSON.stringify({ version: 1, entries: next });
    parseShelf(serialized);
    localStorage.setItem(SHELF_KEY, serialized);
    entries.value = next;
    window.dispatchEvent(new Event('shelf-changed'));
  }
  function change(fn: (items: ShelfWine[]) => ShelfWine[]) {
    try {
      write(fn(parseShelf(localStorage.getItem(SHELF_KEY))));
      error.value = '';
      queueSync();
      return true;
    } catch (e) {
      error.value = e instanceof Error && e.message.startsWith('На полке') ? e.message
        : 'Изменения не сохранены. Проверьте свободное место и доступ к хранилищу браузера.';
      return false;
    }
  }

  // Слияние с аккаунтом. Возвращает true, если сервер ответил; локальная копия остаётся рабочей.
  function sync(): Promise<boolean> {
    if (!user.value || !navigator.onLine) return Promise.resolve(false);
    syncing ??= (async () => {
      try {
        const sent = readDeleted();
        const remote = await $fetch<{ entries: ShelfWine[]; deleted: Deleted[] }>('/api/shelf/sync', {
          method: 'POST', body: { entries: parseShelf(localStorage.getItem(SHELF_KEY)), deleted: sent }, timeout: 15000,
        });
        // Пока шёл запрос, полку могли поправить: такие правки новее ответа и переживут слияние.
        const local = parseShelf(localStorage.getItem(SHELF_KEY));
        const pending = readDeleted();
        write(mergeShelves(local, remote.entries, [...remote.deleted, ...pending]));
        writeDeleted(pending.filter(d => !sent.some(s => s.slug === d.slug && s.at === d.at)));
        synced.value = Date.now();
        return true;
      } catch { return false; }
      finally { syncing = null; }
    })();
    return syncing;
  }
  function queueSync() {
    if (!user.value) return;
    clearTimeout(syncTimer);
    syncTimer = setTimeout(() => { void sync(); }, 800);
  }
  async function syncNow() {
    clearTimeout(syncTimer);
    if (syncing) await syncing;
    return await sync();
  }

  const find = (slug: string) => entries.value.find(e => e.slug === slug);
  return { entries, ready, error, synced, load, find, sync, syncNow,
    add: (slug: string, card: Card, analogues: Analogue[] = []) => change(items => addWine(items, slug, card, analogues)),
    update: (slug: string, patch: Partial<Pick<ShelfWine, 'status' | 'reaction' | 'note' | 'dishes' | 'remindAt'>>) => change(items => updateWine(items, slug, patch)),
    remove: (slug: string) => {
      const ok = change(items => items.filter(e => e.slug !== slug));
      if (ok) writeDeleted([...readDeleted().filter(d => d.slug !== slug), { slug, at: Date.now() }]);
      return ok;
    },
  };
}
