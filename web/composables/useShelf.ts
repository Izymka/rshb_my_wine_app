import type { Card, Analogue } from './useScan';
import { SHELF_KEY, parseShelf, addWine, updateWine } from '~/utils/shelf.mjs';

export interface ShelfWine {
  slug: string; card: Card; analogues: Analogue[]; addedAt: number;
  status: 'want' | 'home' | 'tried'; reaction: 'liked' | 'disliked' | null;
  dishes: string[]; note: string; remindAt: number | null;
}
export function useShelf() {
  const entries = useState<ShelfWine[]>('shelf.entries', () => []);
  const ready = useState('shelf.ready', () => false);
  const error = useState('shelf.error', () => '');
  function load() {
    try { entries.value = parseShelf(localStorage.getItem(SHELF_KEY)); error.value = ''; }
    catch { error.value = 'Не удалось прочитать полку. Данные не перезаписаны. Проверьте доступ к хранилищу браузера.'; }
    ready.value = true;
  }
  function change(fn: (items: ShelfWine[]) => ShelfWine[]) {
    try {
      const next = fn(parseShelf(localStorage.getItem(SHELF_KEY)));
      const serialized = JSON.stringify({ version: 1, entries: next });
      parseShelf(serialized);
      localStorage.setItem(SHELF_KEY, serialized);
      entries.value = next;
      error.value = '';
      window.dispatchEvent(new Event('shelf-changed'));
      return true;
    } catch (e) {
      error.value = e instanceof Error && e.message.startsWith('На полке') ? e.message
        : 'Изменения не сохранены. Проверьте свободное место и доступ к хранилищу браузера.';
      return false;
    }
  }
  const find = (slug: string) => entries.value.find(e => e.slug === slug);
  return { entries, ready, error, load, find,
    add: (slug: string, card: Card, analogues: Analogue[] = []) => change(items => addWine(items, slug, card, analogues)),
    update: (slug: string, patch: Partial<Pick<ShelfWine, 'status' | 'reaction' | 'note' | 'dishes' | 'remindAt'>>) => change(items => updateWine(items, slug, patch)),
    remove: (slug: string) => change(items => items.filter(e => e.slug !== slug)),
  };
}
