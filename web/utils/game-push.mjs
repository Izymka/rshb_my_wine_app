// Игровые пуши «Моей винодельни»: какие события игры стали поводом вернуться, когда о них можно
// сказать и каким текстом (макет 2i). Чистые функции без ввода-вывода: рассылку по подпискам
// делает server/utils/game-push.ts из общего тика пушей (server/plugins/shelf-reminders.ts).
//
// Правила (ТЗ 10.5): не больше двух игровых пушей за сутки по Москве, тишина с 22:00 до 9:00 по
// Москве, несколько готовых объектов склеиваются в один пуш. Тексты не призывают пить.

import { HOUR, VARIETIES, dayKey, itemName, nextDayAt } from './game.mjs';
import { plural } from './wine.mjs';
import { wordleNumber } from './wordle.mjs';

const MIN = 60000;
export const PUSH = {
  daily: 2,
  quietFrom: 22, quietTo: 9, // часы по Москве: [22:00, 9:00)
  gap: 2 * 60 * MIN, // между пушами — не меньше двух реальных часов: второй пуш дня соберёт накопившееся
  stale: 14 * 60 * MIN, // событие старше — уже не новость (вечерний урожай ещё уйдёт утром в 9:00)
  orderWarn: 4 * HOUR, // «заказ заканчивается» — за 4 игровых часа до конца
  keep: 3 * 24 * 60 * MIN, // сколько помнить отправленное и пропущенное
};

// Порядок в склеенном пуше: сначала то, что можно потерять.
const PRIORITY = { order: 0, harvest: 1, barrel: 2, tank: 3, press: 4, wordle: 5 };

const ms = (seconds, speed) => Math.round(seconds * 1000 / speed);
const mskHour = now => new Date(now + 3 * HOUR * 1000).getUTCHours();
export function quietHours(now) {
  const h = mskHour(now);
  return h >= PUSH.quietFrom || h < PUSH.quietTo;
}
export function sentToday(log, now) {
  const day = dayKey(now);
  return (log?.sent || []).filter(t => dayKey(t) === day).length;
}

// События, о которых стоит сказать сейчас. key стабилен для одного и того же объекта: по нему
// событие не повторяется. at — когда событие наступило (реальное время сервера).
export function gameEvents(state, now, speed = 1) {
  if (!state || state.enabled === false) return [];
  const events = [];
  for (const p of state.plots || []) {
    if (p.variety && p.readyAt && p.readyAt <= now) events.push({ key: `plot:${p.id}:${p.plantedAt}`, kind: 'harvest', variety: p.variety, at: p.readyAt });
  }
  for (const b of ['press', 'tank']) {
    for (const j of state[b]?.jobs || []) {
      if (j.readyAt <= now) events.push({ key: `${b}:${j.variety}:${j.startedAt}`, kind: b, variety: j.variety, at: j.readyAt });
    }
  }
  (state.barrels || []).forEach((b, i) => {
    if (b.variety && b.readyAt && b.readyAt <= now) events.push({ key: `barrel:${i}:${b.startedAt}`, kind: 'barrel', variety: b.variety, at: b.readyAt });
  });
  for (const slot of state.orders?.slots || []) {
    const o = slot.order;
    if (!o || o.expiresAt <= now) continue;
    const at = o.expiresAt - ms(PUSH.orderWarn, speed);
    if (at <= now) events.push({ key: `order:${o.id}`, kind: 'order', who: o.who, item: o.item, count: o.count, expiresAt: o.expiresAt, at });
  }
  const today = dayKey(now);
  if (state.wordle?.last !== today) events.push({ key: `wordle:${today}`, kind: 'wordle', number: wordleNumber(now), at: nextDayAt(now) - 24 * HOUR * 1000 });
  return events;
}

function left(msLeft) {
  const minutes = Math.max(1, Math.round(msLeft / MIN));
  if (minutes < 60) return `${minutes} ${plural(minutes, ['минута', 'минуты', 'минут'])}`;
  const hours = Math.round(minutes / 60);
  return `${hours} ${plural(hours, ['час', 'часа', 'часов'])}`;
}
const name = v => VARIETIES[v]?.name || 'винограда';
// Заголовок события и его же упоминание в хвосте склеенного пуша.
function phrase(e) {
  switch (e.kind) {
    case 'harvest': return `Урожай ${name(e.variety)} созрел`;
    case 'barrel': return `Бочка ${name(e.variety)} готова`;
    case 'press': return `Сусло ${name(e.variety)} готово`;
    case 'tank': return `Молодое ${name(e.variety)} готово`;
    case 'order': return `${e.who} ждёт заказ`;
    default: return 'Новое вино дня';
  }
}
const lower = s => s.charAt(0).toLowerCase() + s.slice(1);
const tail = e => (e.kind === 'order' ? `заказ «${e.who}» скоро закроется` : e.kind === 'wordle' ? 'новое «Вино дня»' : lower(phrase(e)));

// Один пуш из нескольких событий: заголовок — самое важное, остальное — в тексте.
export function pushText(events, now) {
  const seen = new Set();
  const list = [...events].sort((a, b) => PRIORITY[a.kind] - PRIORITY[b.kind] || a.at - b.at)
    .filter((e) => { const k = `${e.kind}:${e.variety || e.key}`; if (seen.has(k)) return false; seen.add(k); return true; });
  const [main, ...rest] = list;
  let body;
  if (main.kind === 'order') body = `${itemName(main.item)} × ${main.count}, осталось ${left(main.expiresAt - now)}`;
  else if (main.kind === 'wordle') body = `Вино дня № ${main.number}: сможете угадать с трёх попыток?`;
  if (rest.length) {
    const more = rest.length > 1 ? ` и ещё ${rest.length - 1}` : '';
    const also = `И ${tail(rest[0])}${more}`;
    body = body ? `${body}. ${also}` : `${also} — загляните на винодельню`;
  }
  return { title: phrase(main), body: body || 'Загляните на винодельню', kind: main.kind };
}

// Решение для одного игрока на очередном тике. log — что уже отправлено и пропущено
// ({ handled: { key: at }, sent: [at] }), lastSeen — когда игрок последний раз открывал игру:
// то, что он уже видел на поле, пушем не повторяется. Возвращает пуш (или null) и новый log;
// log сохраняется, только если пуш доставлен хотя бы на одно устройство (иначе повтор на
// следующем тике).
export function planGamePush(state, log, { now, speed = 1, lastSeen = 0 }) {
  const next = { handled: {}, sent: (log?.sent || []).filter(t => t > now - PUSH.keep) };
  for (const [k, t] of Object.entries(log?.handled || {})) if (t > now - PUSH.keep) next.handled[k] = t;
  const pending = [];
  for (const e of gameEvents(state, now, speed)) {
    if (next.handled[e.key]) continue;
    if (e.at <= lastSeen || e.at < now - PUSH.stale) next.handled[e.key] = now; // уже видел или устарело
    else pending.push(e);
  }
  const last = next.sent.length ? Math.max(...next.sent) : 0;
  let reason = null;
  if (!pending.length) reason = 'nothing';
  else if (quietHours(now)) reason = 'quiet';
  else if (sentToday(next, now) >= PUSH.daily) reason = 'limit';
  else if (now - last < PUSH.gap) reason = 'gap';
  if (reason) return { payload: null, reason, log: next };
  const text = pushText(pending, now);
  for (const e of pending) next.handled[e.key] = now;
  next.sent.push(now);
  return {
    payload: { ...text, url: `/winery?push=game_${text.kind}`, tag: 'game' },
    reason: 'sent', events: pending.map(e => e.key), log: next,
  };
}
