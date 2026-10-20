// Игра «Моя винодельня»: баланс и правила. Чистые функции без ввода-вывода — их выполняет сервер
// (единственный источник правды: таймеры, валюты, награды), а клиент берёт отсюда же баланс и
// производные значения для отрисовки (стадия лозы, остаток таймера, цена ускорения).
//
// Время в балансе — игровые секунды. Реальная длительность = секунды / speed: на демо-стенде
// NUXT_GAME_SPEED=60 превращает час роста в минуту, сутки выдержки — в 24 минуты. Границы
// игрового дня (задания, календарь, «Вино дня») от speed не зависят: это реальные сутки по Москве.

import { plural } from './wine.mjs';

export const GAME_VERSION = 1;
export const HOUR = 3600;

// Сорта. level — с какого уровня саженец продаётся в магазине; scanOnly — только скан этикетки
// или отзыв о вине из этого сорта (автохтоны и редкие сорта). Скан открывает и сорт выше
// уровня игрока раньше срока — это пометка «ИЗ СКАНА» в шторке участка.
export const VARIETIES = {
  chardonnay: { name: 'Шардоне', color: 'white', grow: 2 * HOUR, price: 10, yld: 3, level: 1, col: '#cfc56b', match: /шардоне|chardonnay/i },
  riesling: { name: 'Рислинг', color: 'white', grow: 4 * HOUR, price: 20, yld: 4, level: 2, col: '#b5c35f', match: /рислинг|riesling/i },
  sauvignon: { name: 'Совиньон блан', color: 'white', grow: 3 * HOUR, price: 15, yld: 3, level: 3, col: '#c2c873', match: /совиньон[\s-]*блан|sauvignon blanc/i },
  cabernet: { name: 'Каберне Совиньон', color: 'red', grow: 8 * HOUR, price: 40, yld: 4, level: 3, col: '#4b2747', match: /каберне[\s-]*совиньон|cabernet sauvignon/i },
  merlot: { name: 'Мерло', color: 'red', grow: 6 * HOUR, price: 30, yld: 4, level: 4, col: '#55284f', match: /мерло|merlot/i },
  saperavi: { name: 'Саперави', color: 'red', grow: 12 * HOUR, price: 60, yld: 5, level: 5, col: '#3a1c3c', match: /саперави|saperavi/i },
  pinot: { name: 'Пино нуар', color: 'red', grow: 10 * HOUR, price: 50, yld: 4, level: 6, col: '#5e2a4a', match: /пино[\s-]*нуар|pinot noir/i },
  aligote: { name: 'Алиготе', color: 'white', grow: 6 * HOUR, price: 30, yld: 4, scanOnly: true, col: '#c9c06a', match: /алиготе|aligot/i },
  rkatsiteli: { name: 'Ркацители', color: 'white', grow: 10 * HOUR, price: 50, yld: 4, scanOnly: true, col: '#d2c060', match: /ркацители|rkatsiteli/i },
  viognier: { name: 'Вионье', color: 'white', grow: 12 * HOUR, price: 60, yld: 4, scanOnly: true, col: '#c8c46a', match: /вионье|viognier/i },
  kokur: { name: 'Кокур', color: 'white', grow: 14 * HOUR, price: 65, yld: 4, scanOnly: true, col: '#d8c56a', match: /кокур|kokur/i },
  sibirkovy: { name: 'Сибирьковый', color: 'white', grow: 16 * HOUR, price: 70, yld: 5, scanOnly: true, col: '#bfc35a', match: /сибирьков/i },
  golubok: { name: 'Голубок', color: 'red', grow: 18 * HOUR, price: 75, yld: 5, scanOnly: true, col: '#3f1d45', match: /голубок/i },
  tsimlyansky: { name: 'Цимлянский чёрный', color: 'red', grow: 20 * HOUR, price: 80, yld: 5, scanOnly: true, col: '#351a38', match: /цимлянск/i },
  krasnostop: { name: 'Красностоп', color: 'red', grow: 24 * HOUR, price: 90, yld: 5, scanOnly: true, col: '#5b2342', match: /красностоп/i },
};

// Порог опыта для каждого уровня: LEVELS[n - 1] — сколько нужно, чтобы стать уровнем n.
export const LEVELS = [0, 100, 250, 450, 800, 1120, 1600, 2200, 3000, 4000, 5200, 6600, 8200, 10000, 12000, 14500, 17500, 21000, 25000, 30000];

export const BALANCE = {
  start: { coins: 300, grozdi: 10 },
  press: { input: 3, time: 30 * 60, slotPrice: 300, upgrade: [450, 900, 1800] },
  tank: { time: 4 * HOUR, slotPrice: 300, upgrade: [450, 900, 1800] },
  barrel: { days: { 1: 1, 3: 2, 7: 3 }, price: 20, max: 4, level: 3 },
  storage: { start: 60, step: 20, prices: [400, 800, 1600] },
  overripe: 48 * HOUR,
  problem: { chance: 0.2, scarecrowChance: 0.08, window: 3 * HOUR, penalty: 0.8 },
  orders: { base: 4, extra: 2, extraLevel: 5, cooldown: 2 * HOUR, shelfMult: 1.5, shelfRepeat: 3 * 86400 },
  plots: { start: 6, extra: [{ price: 500, level: 1 }, { price: 1000, level: 6 }] },
  bottling: { level: 7 },
  blending: { level: 10 },
  skipSeconds: 1500, // одна Гроздь за 25 минут оставшегося таймера
  speedUpgrade: 0.1,
  quests: { chest: { grozdi: 15 } },
  week: { days: 5, grozdi: 10, coins: 200 },
};

// Декор: куда ставится на поле (координаты макета 375×812) и сколько стоит.
export const DECOR = {
  lantern: { name: 'Фонарь', desc: 'Светится ночью', price: 150, cur: 'coins' },
  bench: { name: 'Скамейка', desc: 'Для гостей винодельни', price: 120, cur: 'coins' },
  flowers: { name: 'Клумба в бочке', desc: 'Цветы круглый год', price: 10, cur: 'grozdi', isNew: true },
  scarecrow: { name: 'Пугало', desc: 'Скворцы прилетают реже', price: 200, cur: 'coins' },
};

// Награды за действия в сервисе: что даёт и сколько раз в сутки. Повтор одного и того же вина
// (скан, отзыв, полка) не награждается никогда — это проверяет сервер по журналу наград.
export const SERVICE = {
  scan: { limit: 5, grozdi: 5, xp: 20, label: 'Скан этикетки' },
  review: { limit: 3, grozdi: 3, xp: 15, rep: 1, label: 'Отзыв' },
  shelf: { limit: 10, coins: 15, xp: 5, label: 'Вино на полке' },
  sommelier: { limit: 1, grozdi: 2, xp: 10, label: 'Вопрос сомелье' },
};

// Календарь входа: 7 дней по кругу, пропуск дня не сбрасывает прогресс.
export const CALENDAR = [
  { coins: 50 }, { sapling: 'chardonnay' }, { grozdi: 3 }, { coins: 100 }, { sapling: 'riesling' }, { grozdi: 5 }, { decor: 'lantern', coins: 300 },
];

// Шаблоны ежедневных заданий: два игровых и одно из сервиса.
export const QUESTS = {
  harvest: { title: n => `Соберите ${n} ${plural(n, ['гроздь', 'грозди', 'гроздей'])}`, icon: 'ph-basket', range: [6, 12], xp: 25 },
  orders: { title: n => `Выполните ${n} ${plural(n, ['заказ', 'заказа', 'заказов'])}`, icon: 'ph-clipboard-text', range: [1, 3], xp: 30 },
  press: { title: n => `Выжмите сусло ${n} ${plural(n, ['раз', 'раза', 'раз'])}`, icon: 'ph-drop', range: [2, 4], xp: 20 },
  plant: { title: n => `Посадите ${n} ${plural(n, ['лозу', 'лозы', 'лоз'])}`, icon: 'ph-plant', range: [3, 5], xp: 20 },
  scan: { title: () => 'Отсканируйте этикетку', icon: 'ph-scan', service: true, go: 'Сканер', href: '/', xp: 30, coins: 50 },
  shelf: { title: () => 'Сохраните вино на полку', icon: 'ph-books', service: true, go: 'Полка', href: '/wines', xp: 25, coins: 40 },
  review: { title: () => 'Оцените вино с полки', icon: 'ph-chats-circle', service: true, go: 'Полка', href: '/wines', xp: 35, coins: 60 },
  wordle: { title: () => 'Сыграйте в «Вино дня»', icon: 'ph-puzzle-piece', service: true, go: 'Играть', href: '/winery/wordle', xp: 25, coins: 40 },
};

const WHO = [
  { who: 'Ресторан «Терруар»', icon: 'ph-fork-knife' }, { who: 'Свадьба у озера', icon: 'ph-heart' },
  { who: 'Фестиваль урожая', icon: 'ph-confetti' }, { who: 'Гастробар «Лоза»', icon: 'ph-wine' },
  { who: 'Сырная лавка', icon: 'ph-cheese' }, { who: 'Дегустация в погребе', icon: 'ph-users-three' },
  { who: 'Ужин шефа', icon: 'ph-cooking-pot' }, { who: 'Пикник в саду', icon: 'ph-tree' },
];

export class GameError extends Error {}
const fail = (message) => { throw new GameError(message); };


// --- Случайность: только детерминированная, от зерна игрока и счётчика, чтобы повтор запроса
// и проверка на сервере давали тот же результат.
export function hashString(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return h >>> 0;
}
export function rng(seed) {
  let a = typeof seed === 'number' ? seed >>> 0 : hashString(String(seed));
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const between = (r, [lo, hi]) => lo + Math.floor(r() * (hi - lo + 1));

// Игровой день — календарные сутки по Москве.
export function dayKey(now) { return new Date(now + 3 * 3600000).toISOString().slice(0, 10); }
export function weekKey(now) {
  const d = new Date(now + 3 * 3600000);
  d.setUTCHours(0, 0, 0, 0);
  d.setUTCDate(d.getUTCDate() - ((d.getUTCDay() + 6) % 7));
  return d.toISOString().slice(0, 10);
}
export function nextDayAt(now) {
  const d = new Date(now + 3 * 3600000);
  d.setUTCHours(24, 0, 0, 0);
  return d.getTime() - 3 * 3600000;
}

// --- Уровни
export function levelOf(xp) {
  let level = 1;
  while (level < LEVELS.length && xp >= LEVELS[level]) level++;
  return level;
}
export function levelProgress(xp) {
  const level = levelOf(xp);
  const prev = LEVELS[level - 1], next = LEVELS[level] ?? null;
  return { level, prev, next, pct: next ? (xp - prev) / (next - prev) : 1 };
}

// --- Предметы склада: «grape:cabernet», «must:cabernet», «young:cabernet», «aged:cabernet».
export const KINDS = { grape: 'Виноград', must: 'Сусло', young: 'Молодое', aged: 'Выдержанное' };
export function parseItem(item) {
  const [kind, variety] = String(item).split(':');
  if (!KINDS[kind] || !VARIETIES[variety]) fail('Неизвестный продукт');
  return { kind, variety };
}
export function itemName(item) {
  const { kind, variety } = parseItem(item);
  const v = VARIETIES[variety].name;
  if (kind === 'grape') return `Виноград ${v}`;
  if (kind === 'must') return `Сусло ${v}`;
  return `${kind === 'young' ? 'Молодое' : 'Выдержанное'} ${v}`;
}
export function itemValue(item) {
  const { kind, variety } = parseItem(item);
  const grape = Math.round(VARIETIES[variety].price * 0.25) + 2;
  const must = Math.round(grape * BALANCE.press.input * 1.2);
  const young = Math.round(must * 1.5);
  return { grape, must, young, aged: Math.round(young * 1.6) }[kind];
}
export function storageUsed(state) {
  return Object.values(state.items).reduce((a, b) => a + b, 0);
}
function addItem(state, item, n) {
  if (n > 0 && storageUsed(state) + n > state.storage) fail('Склад полон — продайте лишнее или расширьте склад');
  state.items[item] = (state.items[item] || 0) + n;
  if (state.items[item] <= 0) delete state.items[item];
}
function takeItem(state, item, n) {
  if ((state.items[item] || 0) < n) fail(`Не хватает: ${itemName(item)}`);
  state.items[item] -= n;
  if (!state.items[item]) delete state.items[item];
}

// --- Сорта, доступные игроку
export function varietyAccess(state, key) {
  const v = VARIETIES[key];
  const level = levelOf(state.xp);
  if (state.unlocked[key]) return { open: true, source: state.unlocked[key] };
  if (!v.scanOnly && level >= v.level) return { open: true, source: 'level' };
  return { open: false, lock: v.scanOnly ? 'Только скан или отзыв' : `Уровень ${v.level} или скан` };
}
export function varietiesFrom(text) {
  if (!text) return [];
  return Object.keys(VARIETIES).filter(k => VARIETIES[k].match.test(String(text)));
}

// --- Время: speed ускоряет все игровые таймеры
const ms = (seconds, speed) => Math.round(seconds * 1000 / speed);
export function upgradeFactor(level) { return Math.max(0.5, 1 - BALANCE.speedUpgrade * level); }
export function skipCost(readyAt, now) {
  return Math.max(1, Math.ceil(Math.max(0, readyAt - now) / 1000 / BALANCE.skipSeconds));
}

// Стадия участка для отрисовки: empty, sapling, leaves, green, ripe, overripe, problem.
export function plotStage(plot, now, speed = 1) {
  if (!plot.variety) return 'empty';
  if (problemActive(plot, now, speed)) return 'problem';
  if (now >= plot.readyAt + ms(BALANCE.overripe, speed)) return 'overripe';
  if (now >= plot.readyAt) return 'ripe';
  const f = (now - plot.plantedAt) / (plot.readyAt - plot.plantedAt);
  return f < 1 / 3 ? 'sapling' : f < 2 / 3 ? 'leaves' : 'green';
}
export function problemActive(plot, now, speed = 1) {
  return Boolean(plot.variety && plot.problemAt && !plot.problemFixed && now >= plot.problemAt
    && now < plot.problemAt + ms(BALANCE.problem.window, speed));
}
export function harvestYield(plot, now, speed = 1) {
  let y = VARIETIES[plot.variety].yld;
  if (now >= plot.readyAt + ms(BALANCE.overripe, speed)) y *= 0.5;
  if (plot.problemAt && !plot.problemFixed && now >= plot.problemAt + ms(BALANCE.problem.window, speed)) y *= BALANCE.problem.penalty;
  return Math.max(1, Math.round(y));
}

// --- Новая игра
export function newGame(seed, now, speed = 1) {
  const state = {
    v: GAME_VERSION, seed: String(seed), n: 0, createdAt: now,
    coins: BALANCE.start.coins, grozdi: BALANCE.start.grozdi, xp: 0, rep: 0,
    plots: Array.from({ length: BALANCE.plots.start + BALANCE.plots.extra.length }, (_, i) => ({
      id: i + 1, owned: i < BALANCE.plots.start, variety: null, plantedAt: 0, readyAt: 0, problemAt: 0, problemFixed: false,
    })),
    press: { level: 0, slots: 1, jobs: [] },
    tank: { level: 0, slots: 1, jobs: [] },
    barrels: [{ variety: null, days: 0, startedAt: 0, readyAt: 0 }],
    storage: BALANCE.storage.start, storageLevel: 0,
    items: { 'grape:chardonnay': 3 },
    saplings: { chardonnay: 2 },
    unlocked: {},
    decor: ['lantern'],
    orders: { slots: [], shelfUsed: {} },
    daily: null, calendar: { idx: 0, last: null }, week: null,
    tutorial: { step: 0, done: false },
    wordle: { streak: 0, last: null, best: 0 },
    stats: { harvested: 0, orders: 0, pressed: 0, fermented: 0, aged: 0, planted: 0 },
    enabled: true,
  };
  // Первая лоза уже созрела, вторая растёт: в онбординге есть что собрать и чего ждать.
  Object.assign(state.plots[0], { variety: 'chardonnay', plantedAt: now - ms(VARIETIES.chardonnay.grow, speed), readyAt: now });
  Object.assign(state.plots[1], { variety: 'chardonnay', plantedAt: now, readyAt: now + ms(VARIETIES.chardonnay.grow, speed) });
  return refresh(state, now, speed, {});
}

// --- Пассивное обновление: смена дня, заказы, календарь. Вызывается перед любым действием и
// при чтении состояния. ctx.wants — вина со статусом «Хочу» на облачной полке игрока.
export function refresh(state, now, speed = 1, ctx = {}) {
  const today = dayKey(now);
  if (!state.daily || state.daily.day !== today) state.daily = makeDaily(state, today);
  const wk = weekKey(now);
  if (!state.week || state.week.key !== wk) state.week = { key: wk, days: [], claimed: false };
  if (!state.week.days.includes(today)) state.week.days.push(today);
  fillOrders(state, now, speed, ctx.wants || []);
  return state;
}

function makeDaily(state, day) {
  const r = rng(`${state.seed}:quests:${day}`);
  const game = ['harvest', 'orders', 'press', 'plant'];
  const picks = [];
  while (picks.length < 2) {
    const k = game[Math.floor(r() * game.length)];
    if (!picks.includes(k)) picks.push(k);
  }
  const service = ['scan', 'shelf', 'review', 'wordle'][Math.floor(r() * 4)];
  const quests = [...picks, service].map(kind => {
    const q = QUESTS[kind];
    return { kind, target: q.range ? between(r, q.range) : 1, xp: q.xp, coins: q.coins || 0, claimed: false };
  });
  return { day, quests, counters: {}, chest: false };
}

export function orderSlots(state) {
  return BALANCE.orders.base + (levelOf(state.xp) >= BALANCE.orders.extraLevel ? BALANCE.orders.extra : 0);
}

function fillOrders(state, now, speed, wants) {
  const slots = orderSlots(state);
  while (state.orders.slots.length < slots) state.orders.slots.push({ order: null, cooldownUntil: 0 });
  state.orders.slots.forEach((slot, i) => {
    if (slot.order && slot.order.expiresAt <= now) slot.order = null;
    if (!slot.order && slot.cooldownUntil <= now) slot.order = makeOrder(state, now, speed, i, wants);
  });
}

// Варианты продукции, которые игрок уже может сделать на своём уровне.
function producible(state) {
  const level = levelOf(state.xp);
  const kinds = ['grape', 'must'];
  if (level >= 2) kinds.push('young');
  if (level >= BALANCE.barrel.level) kinds.push('aged');
  const vars = Object.keys(VARIETIES).filter(k => varietyAccess(state, k).open);
  return { kinds, vars };
}

function makeOrder(state, now, speed, slot, wants) {
  state.n++;
  const r = rng(`${state.seed}:order:${state.n}`);
  const hasShelfOrder = state.orders.slots.some(s => s.order?.shelf);
  // Заказ с полки: гость хочет то же, что игрок отметил «Хочу попробовать». Сорт берётся из
  // карточки вина, и если он ещё закрыт — заказ подсказывает открыть его сканом.
  if (!hasShelfOrder && wants.length && r() < 0.6) {
    const fresh = wants.filter(w => !(now - (state.orders.shelfUsed[w.slug] || 0) < BALANCE.orders.shelfRepeat * 1000)
      && varietiesFrom(w.grapes).length);
    if (fresh.length) {
      const w = fresh[Math.floor(r() * fresh.length)];
      const variety = varietiesFrom(w.grapes)[0];
      const item = `young:${variety}`;
      state.orders.shelfUsed[w.slug] = now;
      return {
        id: state.n, who: `Гость: ${w.name}`.slice(0, 60), icon: 'ph-user', item, count: 1,
        coins: Math.round(itemValue(item) * 1.6 * BALANCE.orders.shelfMult) + 20, xp: 60, grozdi: 0,
        shelf: { slug: w.slug, name: w.name, winery: w.winery || '' }, mult: BALANCE.orders.shelfMult,
        expiresAt: now + ms(24 * HOUR, speed),
      };
    }
  }
  const { kinds, vars } = producible(state);
  const kind = kinds[Math.floor(r() * kinds.length)];
  const variety = vars[Math.floor(r() * vars.length)];
  const item = `${kind}:${variety}`;
  const count = kind === 'grape' ? between(r, [3, 8]) : kind === 'must' ? between(r, [1, 3]) : kind === 'young' ? between(r, [1, 3]) : between(r, [1, 2]);
  const base = itemValue(item) * count;
  const who = WHO[Math.floor(r() * WHO.length)];
  const grozdi = kind === 'aged' && r() < 0.5 ? between(r, [2, 5]) : 0;
  return {
    id: state.n, ...who, item, count, coins: Math.round(base * 1.6) + 10,
    xp: grozdi ? 0 : Math.round(base / 4) + 10, grozdi,
    expiresAt: now + ms(between(r, [9, 48]) * HOUR, speed),
  };
}

// --- Опыт и уровень. Новый уровень даёт монеты и Грозди; что открылось, клиент показывает сам.
function gain(state, reward, out) {
  for (const k of ['coins', 'grozdi', 'rep']) if (reward[k]) state[k] += reward[k];
  if (reward.xp) {
    const before = levelOf(state.xp);
    state.xp += reward.xp;
    const after = levelOf(state.xp);
    for (let l = before + 1; l <= after; l++) {
      state.coins += 50 * l;
      state.grozdi += 3;
      out.levelUp = l;
    }
  }
  for (const k of ['coins', 'grozdi', 'xp', 'rep']) if (reward[k]) out.gained[k] = (out.gained[k] || 0) + reward[k];
}
function spend(state, cur, price) {
  if (state[cur] < price) fail(cur === 'grozdi' ? 'Не хватает Гроздей' : 'Не хватает монет');
  state[cur] -= price;
}
function count(state, key, n = 1) {
  state.daily.counters[key] = (state.daily.counters[key] || 0) + n;
}

const building = (state, b) => (b === 'press' || b === 'tank' ? state[b] : fail('Неизвестное здание'));
const intArg = (v, lo, hi) => (Number.isInteger(v) && v >= lo && v <= hi ? v : fail('Неверный запрос'));

// --- Действия игрока. Мутирует state и возвращает, что получено, для анимаций на клиенте.
export function applyAction(state, action, now, speed = 1, ctx = {}) {
  refresh(state, now, speed, ctx);
  const out = { gained: {}, items: {} };
  const a = action || {};
  const level = levelOf(state.xp);
  switch (a.type) {
    case 'plant': {
      const plot = state.plots[intArg(a.plot, 0, state.plots.length - 1)];
      if (!plot.owned) fail('Участок ещё не куплен');
      if (plot.variety) fail('Участок занят');
      const v = VARIETIES[a.variety] || fail('Неизвестный сорт');
      if (!varietyAccess(state, a.variety).open) fail('Сорт ещё не открыт');
      if (state.saplings[a.variety]) {
        state.saplings[a.variety]--;
        if (!state.saplings[a.variety]) delete state.saplings[a.variety];
      } else spend(state, 'coins', v.price);
      const r = rng(`${state.seed}:plant:${++state.n}`);
      const grow = ms(v.grow, speed);
      const chance = state.decor.includes('scarecrow') ? BALANCE.problem.scarecrowChance : BALANCE.problem.chance;
      // В онбординге скворцов нет, чтобы первая лоза не огорчала.
      const problemAt = state.tutorial.done && r() < chance ? now + Math.round(grow * (0.3 + r() * 0.5)) : 0;
      Object.assign(plot, { variety: a.variety, plantedAt: now, readyAt: now + grow, problemAt, problemFixed: false });
      state.stats.planted++;
      count(state, 'plant');
      tutorialStep(state, 3);
      break;
    }
    case 'harvest': {
      const plot = state.plots[intArg(a.plot, 0, state.plots.length - 1)];
      if (!plot.variety || now < plot.readyAt) fail('Урожай ещё не созрел');
      if (problemActive(plot, now, speed)) fail('Сначала прогоните скворцов');
      const n = harvestYield(plot, now, speed);
      addItem(state, `grape:${plot.variety}`, n);
      out.items[`grape:${plot.variety}`] = n;
      gain(state, { xp: n * 2 + 3 }, out);
      Object.assign(plot, { variety: null, plantedAt: 0, readyAt: 0, problemAt: 0, problemFixed: false });
      state.stats.harvested += n;
      count(state, 'harvest', n);
      tutorialStep(state, 1);
      break;
    }
    case 'shoo': {
      const plot = state.plots[intArg(a.plot, 0, state.plots.length - 1)];
      if (!problemActive(plot, now, speed)) fail('Скворцов уже нет');
      plot.problemFixed = true;
      gain(state, { xp: 5 }, out);
      break;
    }
    case 'start': {
      const b = building(state, a.building);
      if (b.jobs.length >= b.slots) fail('Все слоты заняты');
      if (!VARIETIES[a.variety]) fail('Неизвестный сорт');
      const isPress = a.building === 'press';
      takeItem(state, `${isPress ? 'grape' : 'must'}:${a.variety}`, isPress ? BALANCE.press.input : 1);
      const time = ms(BALANCE[a.building].time * upgradeFactor(b.level), speed);
      b.jobs.push({ variety: a.variety, startedAt: now, readyAt: now + time });
      if (isPress) tutorialStep(state, 2);
      break;
    }
    case 'collect': {
      const b = building(state, a.building);
      const ready = b.jobs.filter(j => j.readyAt <= now);
      if (!ready.length) fail('Пока нечего забирать');
      const kind = a.building === 'press' ? 'must' : 'young';
      for (const j of ready) {
        addItem(state, `${kind}:${j.variety}`, 1);
        out.items[`${kind}:${j.variety}`] = (out.items[`${kind}:${j.variety}`] || 0) + 1;
        b.jobs.splice(b.jobs.indexOf(j), 1);
      }
      gain(state, { xp: ready.length * (kind === 'must' ? 5 : 10) }, out);
      if (kind === 'must') { state.stats.pressed += ready.length; count(state, 'press', ready.length); }
      else { state.stats.fermented += ready.length; count(state, 'ferment', ready.length); }
      break;
    }
    case 'age': {
      if (level < BALANCE.barrel.level) fail(`Погреб откроется на ${BALANCE.barrel.level} уровне`);
      const barrel = state.barrels[intArg(a.barrel, 0, state.barrels.length - 1)];
      if (barrel.variety) fail('Бочка занята');
      const days = Number(a.days);
      if (!BALANCE.barrel.days[days]) fail('Выберите срок выдержки');
      takeItem(state, `young:${a.variety}`, 1);
      Object.assign(barrel, { variety: a.variety, days, startedAt: now, readyAt: now + ms(days * 24 * HOUR, speed) });
      break;
    }
    case 'uncork': {
      const barrel = state.barrels[intArg(a.barrel, 0, state.barrels.length - 1)];
      if (!barrel.variety || barrel.readyAt > now) fail('Вино ещё выдерживается');
      const n = BALANCE.barrel.days[barrel.days];
      addItem(state, `aged:${barrel.variety}`, n);
      out.items[`aged:${barrel.variety}`] = n;
      gain(state, { xp: 15 * barrel.days }, out);
      Object.assign(barrel, { variety: null, days: 0, startedAt: 0, readyAt: 0 });
      state.stats.aged += n;
      break;
    }
    case 'skip': {
      let job;
      if (a.building === 'barrel') job = state.barrels[intArg(a.index, 0, state.barrels.length - 1)];
      else job = building(state, a.building).jobs[intArg(a.index ?? 0, 0, 9)];
      if (!job?.variety || job.readyAt <= now) fail('Ускорять нечего');
      spend(state, 'grozdi', skipCost(job.readyAt, now));
      job.readyAt = now;
      break;
    }
    case 'upgrade': {
      const b = building(state, a.building);
      const price = BALANCE[a.building].upgrade[b.level] ?? fail('Здание улучшено до максимума');
      spend(state, 'coins', price);
      b.level++;
      break;
    }
    case 'buy': return buy(state, a, now, speed, ctx, out);
    case 'sell': {
      const n = intArg(a.count ?? 1, 1, 999);
      takeItem(state, a.item, n);
      gain(state, { coins: itemValue(a.item) * n }, out);
      break;
    }
    case 'fulfill': {
      const slot = state.orders.slots[intArg(a.slot, 0, state.orders.slots.length - 1)];
      const o = slot?.order || fail('Заказ уже закрыт');
      if (o.shelf && !varietyAccess(state, parseItem(o.item).variety).open) fail('Сначала откройте сорт сканом');
      takeItem(state, o.item, o.count);
      gain(state, { coins: o.coins, xp: o.xp, grozdi: o.grozdi }, out);
      slot.order = null;
      slot.cooldownUntil = now + ms(BALANCE.orders.cooldown, speed);
      slot.doneAt = now;
      state.stats.orders++;
      count(state, 'orders');
      break;
    }
    case 'quest': {
      const q = state.daily.quests[intArg(a.index, 0, state.daily.quests.length - 1)];
      if (q.claimed) fail('Награда уже получена');
      if (questProgress(state, q) < q.target) fail('Задание ещё не выполнено');
      q.claimed = true;
      gain(state, { xp: q.xp, coins: q.coins }, out);
      break;
    }
    case 'chest': {
      if (state.daily.chest) fail('Сундук уже открыт');
      if (!state.daily.quests.every(q => q.claimed)) fail('Сначала выполните все задания');
      state.daily.chest = true;
      gain(state, BALANCE.quests.chest, out);
      break;
    }
    case 'calendar': {
      const today = dayKey(now);
      if (state.calendar.last === today) fail('Сегодняшний подарок уже получен');
      const reward = CALENDAR[state.calendar.idx];
      if (reward.decor && !state.decor.includes(reward.decor)) { state.decor.push(reward.decor); out.decor = reward.decor; }
      else if (reward.sapling) { state.saplings[reward.sapling] = (state.saplings[reward.sapling] || 0) + 1; out.saplings = { [reward.sapling]: 1 }; }
      else gain(state, { coins: reward.coins, grozdi: reward.grozdi }, out);
      state.calendar = { idx: (state.calendar.idx + 1) % CALENDAR.length, last: today };
      break;
    }
    case 'week': {
      if (state.week.claimed) fail('Недельный сундук уже открыт');
      if (state.week.days.length < BALANCE.week.days) fail(`Нужно ${BALANCE.week.days} активных дней за неделю`);
      state.week.claimed = true;
      gain(state, { grozdi: BALANCE.week.grozdi, coins: BALANCE.week.coins }, out);
      break;
    }
    case 'tutorial': {
      if (a.skip) state.tutorial = { step: TUTORIAL_STEPS, done: true };
      else tutorialStep(state, intArg(a.step, 0, TUTORIAL_STEPS));
      break;
    }
    case 'settings': {
      if (typeof a.enabled === 'boolean') state.enabled = a.enabled;
      break;
    }
    default: fail('Неизвестное действие');
  }
  fillOrders(state, now, speed, ctx.wants || []);
  return out;
}

export const TUTORIAL_STEPS = 6;
// Шаг онбординга продвигается, только если игрок сейчас на нём: повторные действия не перескакивают.
function tutorialStep(state, step) {
  if (state.tutorial.done || state.tutorial.step !== step) return;
  state.tutorial.step = step + 1;
  if (state.tutorial.step >= TUTORIAL_STEPS) state.tutorial.done = true;
}

export function shopCatalog(state) {
  const level = levelOf(state.xp);
  const nextPlot = state.plots.find(p => !p.owned);
  const extraIdx = nextPlot ? nextPlot.id - BALANCE.plots.start - 1 : -1;
  return {
    saplings: Object.entries(VARIETIES).map(([k, v]) => {
      const access = varietyAccess(state, k);
      return { id: `sapling:${k}`, variety: k, price: v.price, cur: 'coins', lock: access.open ? null : access.lock, scan: access.source === 'scan' || access.source === 'review' };
    }),
    buildings: [
      { id: 'barrel', price: BALANCE.barrel.price, cur: 'grozdi', lock: level < BALANCE.barrel.level ? `Уровень ${BALANCE.barrel.level}` : state.barrels.length >= BALANCE.barrel.max ? 'Максимум бочек' : null },
      { id: 'slot:press', price: BALANCE.press.slotPrice, cur: 'coins', lock: state.press.slots >= 2 ? 'Уже куплено' : null },
      { id: 'slot:tank', price: BALANCE.tank.slotPrice, cur: 'coins', lock: state.tank.slots >= 2 ? 'Уже куплено' : null },
      { id: 'bottling', lock: `Уровень ${BALANCE.bottling.level}` },
      { id: 'blending', lock: `Уровень ${BALANCE.blending.level}` },
    ],
    plots: BALANCE.plots.extra.map((p, i) => {
      const plot = state.plots[BALANCE.plots.start + i];
      return { id: `plot:${plot.id}`, plot: plot.id, price: p.price, cur: 'coins',
        lock: plot.owned ? 'Уже ваш' : i !== extraIdx ? 'Сначала предыдущий' : level < p.level ? `Уровень ${p.level}` : null };
    }),
    decor: Object.entries(DECOR).map(([k, d]) => ({ id: `decor:${k}`, decor: k, price: d.price, cur: d.cur, lock: state.decor.includes(k) ? 'Уже стоит' : null })),
    storage: { id: 'storage', price: BALANCE.storage.prices[state.storageLevel] ?? null, next: state.storage + BALANCE.storage.step },
  };
}

function buy(state, a, now, speed, ctx, out) {
  const cat = shopCatalog(state);
  const all = [...cat.saplings, ...cat.buildings, ...cat.plots, ...cat.decor];
  if (a.item === 'storage') {
    const price = cat.storage.price ?? fail('Склад расширен до максимума');
    spend(state, 'coins', price);
    state.storage += BALANCE.storage.step;
    state.storageLevel++;
  } else {
    const item = all.find(i => i.id === a.item) || fail('Такого товара нет');
    if (item.lock) fail(item.lock);
    spend(state, item.cur, item.price);
    const [kind, arg] = item.id.split(':');
    if (kind === 'sapling') state.saplings[arg] = (state.saplings[arg] || 0) + 1;
    else if (kind === 'barrel') state.barrels.push({ variety: null, days: 0, startedAt: 0, readyAt: 0 });
    else if (kind === 'slot') state[arg].slots++;
    else if (kind === 'plot') state.plots.find(p => p.id === Number(arg)).owned = true;
    else if (kind === 'decor') state.decor.push(arg);
  }
  fillOrders(state, now, speed, ctx.wants || []);
  return out;
}

export function questProgress(state, q) {
  const c = state.daily?.counters || {};
  return Math.min(q.target, c[q.kind] || 0);
}

// --- Награды за действия в сервисе. data.grapes — сорта из карточки вина (скан, отзыв):
// закрытый сорт открывается и даёт саженец. Лимиты и повторы проверяет сервер до вызова.
export function serviceReward(state, kind, data, now, speed = 1) {
  refresh(state, now, speed, {});
  const cfg = SERVICE[kind] || fail('Неизвестная награда');
  const out = { gained: {}, items: {}, kind, unlocked: [], saplings: {} };
  gain(state, { coins: cfg.coins, grozdi: cfg.grozdi, xp: cfg.xp, rep: cfg.rep }, out);
  if (kind === 'review' && String(data?.note || '').trim().length >= 40) gain(state, { rep: 1 }, out);
  if (kind === 'scan' || kind === 'review') {
    // Купаж может назвать много сортов: открываем до двух закрытых, редкие (scanOnly) — первыми.
    const found = varietiesFrom(data?.grapes);
    const locked = found.filter(v => !varietyAccess(state, v).open)
      .sort((a, b) => Number(Boolean(VARIETIES[b].scanOnly)) - Number(Boolean(VARIETIES[a].scanOnly)));
    for (const v of locked.slice(0, 2)) { state.unlocked[v] = kind; out.unlocked.push(v); }
    // Саженец даётся за новый сорт, а если всё уже открыто — за первый сорт вина при скане.
    const give = out.unlocked.length ? out.unlocked : kind === 'scan' ? found.slice(0, 1) : [];
    for (const v of give) {
      state.saplings[v] = (state.saplings[v] || 0) + 1;
      out.saplings[v] = 1;
    }
  }
  count(state, kind);
  return out;
}

export function wordleReward(attempt) {
  return { grozdi: [25, 22, 20, 15, 10, 5][attempt - 1] ?? 5, xp: 30 };
}
export function applyWordle(state, day, attempt, now, speed = 1) {
  refresh(state, now, speed, {});
  const out = { gained: {}, items: {} };
  const prevDay = dayKey(new Date(`${day}T12:00:00Z`).getTime() - 86400000);
  state.wordle.streak = attempt ? (state.wordle.last === prevDay ? state.wordle.streak + 1 : 1) : 0;
  state.wordle.best = Math.max(state.wordle.best, state.wordle.streak);
  state.wordle.last = day;
  if (attempt) gain(state, wordleReward(attempt), out);
  count(state, 'wordle');
  return out;
}
