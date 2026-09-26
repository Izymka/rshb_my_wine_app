export const SHELF_KEY = 'wine-shelf:v1';
export const DAY = 86400000;
export const STATUS = { want: 'Хочу попробовать', home: 'Есть дома', tried: 'Пробовал' };
export const REACTIONS = { liked: 'Понравилось', disliked: 'Не моё' };
export const DISHES = ['Рыба', 'Морепродукты', 'Сыры', 'Мясо', 'Птица', 'Паста', 'Овощи', 'Десерт', 'Без еды', 'Другое'];
const fields = ['name', 'winery', 'category', 'color', 'region', 'grapes', 'description', 'vintage'];
export function cleanCard(card) {
  return Object.fromEntries(fields.filter(k => typeof card?.[k] === 'string' || typeof card?.[k] === 'number')
    .map(k => [k, String(card[k]).slice(0, k === 'description' ? 6000 : 500)]));
}
export function parseShelf(raw) {
  if (!raw) return [];
  const data = JSON.parse(raw);
  if (data.version !== 1 || !Array.isArray(data.entries) || data.entries.length > 500) throw Error('Invalid shelf');
  const ids = new Set();
  return data.entries.map(e => {
    if (!e || typeof e.slug !== 'string' || !e.slug || e.slug.length > 500 || ids.has(e.slug)
      || !Object.hasOwn(STATUS, e.status) || !Number.isFinite(e.addedAt)
      || !e.card || typeof e.card.name !== 'string'
      || !(e.reaction === null || Object.hasOwn(REACTIONS, e.reaction))
      || !Array.isArray(e.dishes) || e.dishes.some(d => !DISHES.includes(d))
      || typeof e.note !== 'string' || e.note.length > 500
      || !(e.remindAt === null || Number.isFinite(e.remindAt))
      || !Array.isArray(e.analogues)) throw Error('Invalid shelf entry');
    ids.add(e.slug);
    return { ...e, card: cleanCard(e.card), analogues: e.analogues.slice(0, 5)
      .filter(a => a && typeof a.slug === 'string' && a.slug && typeof a.name === 'string')
      .map(a => ({ ...cleanCard(a), slug: a.slug })) };
  });
}
export function addWine(entries, slug, card, analogues = [], now = Date.now()) {
  if (entries.some(e => e.slug === slug)) return entries;
  if (entries.length >= 500) throw Error('На полке уже 500 вин. Удалите ненужные, чтобы добавить новое.');
  if (!slug || !card?.name) throw Error('Не удалось сохранить карточку вина.');
  return [{ slug, card: cleanCard(card), analogues: analogues.filter(a => a.slug !== slug).slice(0, 5)
    .map(a => ({ ...cleanCard(a), slug: a.slug })), addedAt: now, status: 'want',
    reaction: null, dishes: [], note: '', remindAt: now + 3 * DAY }, ...entries];
}
export function updateWine(entries, slug, patch) {
  return entries.map(e => {
    if (e.slug !== slug) return e;
    const next = { ...e, ...patch, slug: e.slug, addedAt: e.addedAt, card: e.card, analogues: e.analogues };
    if (patch.reaction) next.status = 'tried';
    if (next.reaction) next.remindAt = null;
    return next;
  });
}
export function remindersFor(entries) {
  return entries.filter(e => !e.reaction && e.remindAt !== null)
    .map(e => ({ id: `${e.slug}:${e.addedAt}:${e.remindAt}`, slug: e.slug, name: e.card.name, dueAt: e.remindAt }));
}
// Fallback for a manually selected recognition candidate: compare only known cards.
export function similarKnown(slug, card, cards) {
  const words = value => new Set(String(value || '').toLowerCase().split(/[,;\s]+/).filter(Boolean));
  const grapes = words(card.grapes);
  return [...new Map(cards.filter(c => c.slug && c.slug !== slug).map(c => [c.slug, c])).values()]
    .filter(c => card.category && c.category === card.category && c.winery !== card.winery)
    .map(c => ({ ...c, score: [...words(c.grapes)].filter(g => grapes.has(g)).length * 3
      + (card.region && c.region === card.region ? 1 : 0) }))
    .sort((a, b) => b.score - a.score).slice(0, 5);
}
export function winePath(slug) { return `/wines/${encodeURIComponent(slug)}`; }
export function shelfDate(value) { return new Date(value).toLocaleDateString('ru-RU'); }
