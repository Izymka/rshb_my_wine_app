// Признаки карточки для интерфейса: сладость, ароматы чипами, температура подачи, цвет силуэта
// бутылки. В каталоге платформы нет отдельных полей под это, поэтому сладость читается из slug
// (…-rozovoe-suhoe-13), а ароматы — из словаря по описанию. Ничего не придумываем: нет совпадений —
// нет чипа.

const SWEETNESS = [
  [/ekstra-bryut|extra-brut/, 'Экстра брют'],
  [/bryut|brut/, 'Брют'],
  [/polusuhoe/, 'Полусухое'],
  [/polusladkoe/, 'Полусладкое'],
  [/suhoe/, 'Сухое'],
  [/sladkoe|desertnoe/, 'Сладкое'],
];

export function sweetness(slug) {
  const s = String(slug || '').toLowerCase();
  return SWEETNESS.find(([re]) => re.test(s))?.[1] || null;
}

const AROMAS = [
  [/цитрус|лимон|грейпфрут|лайм|апельсин|мандарин/, 'Цитрусовые', 'ph-orange-slice', '#c9902f'],
  [/яблок|груш|айв/, 'Яблоко и груша', 'ph-tree', '#4f7a4a'],
  [/персик|абрикос|нектарин/, 'Персик, абрикос', 'ph-sun', '#c9902f'],
  [/вишн|черешн|слив|чернослив/, 'Вишня и слива', 'ph-drop', '#8f3d42'],
  [/ягод|малин|клубник|земляник|смородин|ежевик|черник/, 'Ягоды', 'ph-drop', '#8f3d42'],
  [/тропическ|ананас|манго|маракуй|личи/, 'Тропические фрукты', 'ph-sun', '#c9902f'],
  [/цвет(ы|ов|оч|ущ)|лепест|акаци|жасмин|фиалк|липов/, 'Цветы', 'ph-flower', '#e2a39b'],
  [/трав|мят|чабрец|полын|фенхел/, 'Травы', 'ph-plant', '#4f7a4a'],
  [/камен|минерал|кремн/, 'Минеральность', 'ph-mountains', '#857e79'],
  [/пряност|специ|перец|перч|гвоздик|корица/, 'Пряности', 'ph-pepper', '#8f3d42'],
  [/ванил|дуб|бочк|подкопч/, 'Дуб и ваниль', 'ph-fire', '#c9902f'],
  [/м[её]д/, 'Мёд', 'ph-drop', '#c9902f'],
  [/шоколад|кофе|какао/, 'Шоколад, кофе', 'ph-coffee', '#7a5a3a'],
  [/хлеб|бриош|дрожж|тост|выпечк/, 'Выпечка', 'ph-bread', '#c9902f'],
];

export function aromas(description, limit = 4) {
  const text = String(description || '').toLowerCase();
  return AROMAS.filter(([re]) => re.test(text)).slice(0, limit)
    .map(([, label, icon, color]) => ({ label, icon, color }));
}

// Общепринятые диапазоны подачи по стилю вина.
export function servingTemp(card, slug) {
  const kind = `${card?.category || ''} ${card?.name || ''}`.toLowerCase();
  const sweet = sweetness(slug);
  if (/игрист|шампан/.test(kind) || sweet === 'Брют' || sweet === 'Экстра брют') return '6–8°';
  if (/красн/.test(kind)) return '16–18°';
  if (/оранж/.test(kind)) return '12–14°';
  if (/бел|роз/.test(kind)) return sweet === 'Сладкое' ? '10–12°' : '8–10°';
  return null;
}

export function bottleColor(card) {
  const kind = `${card?.category || ''}`.toLowerCase();
  if (/бел/.test(kind)) return '#c8b46a';
  if (/роз/.test(kind)) return '#e2a39b';
  if (/оранж/.test(kind)) return '#d08a3c';
  if (/красн/.test(kind)) return '#5b1f29';
  return '#8f3d42';
}

export function grapeList(grapes) {
  return typeof grapes === 'string' ? grapes.split(',').map(s => s.trim()).filter(Boolean) : [];
}

// Чипы под названием: цвет, сладость, сорта (без обобщений вроде «Белые сорта винограда»).
export function wineTags(card, slug) {
  const tags = [];
  if (card?.category) tags.push(card.category);
  const sweet = sweetness(slug);
  if (sweet) tags.push(sweet);
  for (const g of grapeList(card?.grapes)) if (!/сорта/i.test(g) && tags.length < 5) tags.push(g);
  return tags;
}

export const DISH_ICONS = {
  'Рыба': 'ph-fish', 'Морепродукты': 'ph-shrimp', 'Сыры': 'ph-cheese', 'Мясо': 'ph-cow',
  'Птица': 'ph-bird', 'Паста': 'ph-bowl-food', 'Овощи': 'ph-carrot', 'Десерт': 'ph-cake',
  'Без еды': 'ph-wine', 'Другое': 'ph-fork-knife',
};

export function initial(name) {
  const n = String(name || '').trim();
  return n ? n[0].toUpperCase() : '?';
}

// «+7 (916) 123-45-67» из 10 цифр (или их начала — для поля ввода).
export function formatPhone(digits) {
  const d = String(digits || '');
  if (!d) return '';
  let out = '(' + d.slice(0, 3);
  if (d.length > 3) out += ') ' + d.slice(3, 6);
  if (d.length > 6) out += '-' + d.slice(6, 8);
  if (d.length > 8) out += '-' + d.slice(8, 10);
  return out;
}

export function maskedPhone(digits) {
  const d = String(digits || '');
  return d.length === 10 ? `+7 (${d.slice(0, 3)}) •••-••-${d.slice(8)}` : '';
}

export function reviewDate(iso, now = new Date()) {
  const date = new Date(iso);
  const day = value => { const d = new Date(value); d.setHours(0, 0, 0, 0); return d.getTime(); };
  const days = Math.round((day(now) - day(date)) / 86400000);
  if (days <= 0) return 'сегодня';
  if (days === 1) return 'вчера';
  return date.toLocaleDateString('ru-RU', { day: 'numeric', month: 'long' });
}

// Склонение: plural(3, ['вино', 'вина', 'вин']).
export function plural(n, forms) {
  const a = Math.abs(n) % 100, b = a % 10;
  return forms[a > 10 && a < 20 ? 2 : b === 1 ? 0 : b >= 2 && b <= 4 ? 1 : 2];
}
