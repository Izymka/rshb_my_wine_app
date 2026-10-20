import assert from 'node:assert/strict';
import { test } from 'node:test';
import { HOUR, applyAction, newGame } from '../utils/game.mjs';
import { PUSH, gameEvents, planGamePush, pushText, quietHours, sentToday } from '../utils/game-push.mjs';

const H = 3600000;
const MSK = (h, m = 0, d = 19) => Date.UTC(2026, 9, d, h - 3, m); // время по Москве
const SPEED = 60;

// Винодельня, открытая игроком в 10:00: всё, что созреет позже, — повод для пуша.
function winery(seenAt = MSK(10)) {
  const s = newGame('push', seenAt, SPEED);
  s.wordle.last = '2026-10-19'; // Винордл сегодня уже сыгран — не мешает остальным проверкам
  for (const slot of s.orders.slots) slot.order.expiresAt = seenAt + 100 * H;
  return s;
}
const plan = (s, log, now, lastSeen = MSK(10)) => planGamePush(s, log, { now, speed: SPEED, lastSeen });

test('ready objects become events with stable keys; a disabled game yields nothing', () => {
  const s = winery();
  const at = MSK(10) + 2 * HOUR * 1000 / SPEED; // Шардоне растёт 2 игровых часа = 2 минуты
  assert.deepEqual(gameEvents(s, at - 1, SPEED).filter(e => e.kind === 'harvest').map(e => e.key), ['plot:1:' + s.plots[0].plantedAt]);
  assert.equal(gameEvents(s, at, SPEED).filter(e => e.kind === 'harvest').length, 2);
  s.press.jobs.push({ variety: 'cabernet', startedAt: 1, readyAt: at });
  s.barrels[0] = { variety: 'saperavi', days: 1, startedAt: 2, readyAt: at };
  assert.deepEqual(gameEvents(s, at, SPEED).map(e => e.kind).sort(), ['barrel', 'harvest', 'harvest', 'press']);
  s.enabled = false;
  assert.deepEqual(gameEvents(s, at, SPEED), []);
});

test('what the player already saw on the field is not pushed', () => {
  const s = winery();
  // Первая лоза созрела ещё до визита в 10:00, вторая — через 2 минуты после.
  const r = plan(s, {}, MSK(10, 5));
  assert.equal(r.reason, 'sent');
  assert.deepEqual(r.events, [`plot:2:${s.plots[1].plantedAt}`]);
  assert.equal(r.payload.title, 'Урожай Шардоне созрел');
  assert.equal(r.payload.url, '/winery?push=game_harvest');
  assert.equal(plan(s, {}, MSK(10, 5), MSK(10, 4)).reason, 'nothing');
});

test('quiet hours 22:00–9:00 Moscow hold pushes; they go out merged at 9:00', () => {
  assert.equal(quietHours(MSK(21, 59)), false);
  assert.equal(quietHours(MSK(22)), true);
  assert.equal(quietHours(MSK(3)), true);
  assert.equal(quietHours(MSK(8, 59)), true);
  assert.equal(quietHours(MSK(9)), false);
  const s = winery(MSK(21, 50));
  s.plots[1].readyAt = MSK(22, 30);
  s.barrels[0] = { variety: 'cabernet', days: 1, startedAt: 3, readyAt: MSK(23) };
  let log = {};
  for (const t of [MSK(22, 31), MSK(23, 30), MSK(5), MSK(8, 59)].map(t => t < MSK(12) ? t + 24 * H : t)) {
    const r = plan(s, log, t, MSK(21, 50));
    assert.equal(r.payload, null);
    assert.equal(r.reason, 'quiet');
    log = r.log;
  }
  const r = plan(s, log, MSK(9, 0, 20), MSK(21, 50));
  assert.equal(r.reason, 'sent');
  assert.equal(r.payload.title, 'Урожай Шардоне созрел');
  // Утром 20-го к ночным событиям добавился новый Винордл — всё в одном пуше.
  assert.equal(r.payload.body, 'И бочка Каберне Совиньон готова и ещё 1 — загляните на винодельню');
  assert.equal(r.events.length, 3);
});

test('no more than 2 game pushes per Moscow day, at least 2 hours apart; the counter resets at midnight', () => {
  const s = winery();
  let log = {};
  let sent = 0;
  // Каждые 10 минут созревает что-то новое — поводов много, пушей не больше двух.
  for (let t = MSK(10, 10), i = 0; t < MSK(22); t += 10 * 60000, i++) {
    s.press.jobs = [{ variety: 'chardonnay', startedAt: 100 + i, readyAt: t }];
    const r = plan(s, log, t);
    if (r.payload) { sent++; assert.ok(sentToday(log, t) < PUSH.daily); }
    log = r.log;
  }
  assert.equal(sent, 2);
  assert.equal(sentToday(log, MSK(21)), 2);
  assert.ok(log.sent[1] - log.sent[0] >= PUSH.gap);
  s.press.jobs = [{ variety: 'merlot', startedAt: 500, readyAt: MSK(21, 25) }];
  assert.equal(plan(s, log, MSK(21, 30)).reason, 'limit');
  assert.equal(sentToday(log, MSK(9, 30, 20)), 0);
  s.press.jobs = [{ variety: 'riesling', startedAt: 999, readyAt: MSK(9, 20, 20) }];
  assert.equal(plan(s, log, MSK(9, 30, 20)).reason, 'sent');
});

test('an event is pushed once; after a delivery failure the caller keeps the old log and it retries', () => {
  const s = winery();
  const first = plan(s, {}, MSK(10, 5));
  assert.equal(first.reason, 'sent');
  const again = plan(s, first.log, MSK(13));
  assert.equal(again.reason, 'nothing');
  // Не доставлено: журнал не сохранили — следующий тик пробует снова.
  assert.equal(plan(s, {}, MSK(10, 6)).reason, 'sent');
});

test('stale events are dropped instead of pushed the next day', () => {
  const s = winery(MSK(10));
  s.plots[1].readyAt = MSK(11);
  s.wordle.last = '2026-10-20';
  const log = { sent: [MSK(10, 30), MSK(10, 45)] }; // лимит дня исчерпан
  assert.equal(plan(s, log, MSK(12)).reason, 'limit');
  const r = plan(s, log, MSK(9, 0, 20));
  assert.equal(r.reason, 'nothing');
  assert.ok(r.log.handled[`plot:2:${s.plots[1].plantedAt}`]);
});

test('an expiring order is warned 4 game hours ahead and leads a merged push', () => {
  const s = winery();
  s.plots[1].readyAt = MSK(11);
  const o = s.orders.slots[0].order;
  Object.assign(o, { who: 'Ресторан «Терруар»', item: 'young:riesling', count: 3, expiresAt: MSK(11, 4) });
  assert.equal(gameEvents(s, MSK(10, 59), SPEED).some(e => e.kind === 'order'), false); // 4 игровых часа = 4 минуты
  const r = plan(s, {}, MSK(11));
  assert.equal(r.payload.title, 'Ресторан «Терруар» ждёт заказ');
  assert.equal(r.payload.body, 'Молодое Рислинг × 3, осталось 4 минуты. И урожай Шардоне созрел');
  assert.equal(r.payload.url, '/winery?push=game_order');
});

test('a new Винордл is announced once per day if the player has not played and not opened the game today', () => {
  const s = winery(MSK(20, 0, 18));
  s.wordle.last = '2026-10-18';
  for (const slot of s.orders.slots) slot.order.expiresAt = MSK(20, 0, 25);
  s.plots[1].variety = null;
  const r = plan(s, {}, MSK(9, 0), MSK(20, 0, 18));
  assert.equal(r.payload.title, 'Новое вино дня');
  assert.match(r.payload.body, /^Винордл № \d+: сможете угадать с трёх попыток\?$/);
  assert.equal(plan(s, r.log, MSK(12), MSK(20, 0, 18)).reason, 'nothing');
  assert.equal(plan(s, {}, MSK(12), MSK(8, 30)).reason, 'nothing'); // уже заходил сегодня
  s.wordle.last = '2026-10-19';
  assert.equal(plan(s, {}, MSK(12), MSK(20, 0, 18)).reason, 'nothing');
});

test('push texts follow mockup 2i and never call to drink', () => {
  const now = MSK(12);
  const texts = [
    pushText([{ kind: 'harvest', variety: 'saperavi', at: 1, key: 'a' }, { kind: 'barrel', variety: 'cabernet', at: 2, key: 'b' }], now),
    pushText([{ kind: 'press', variety: 'merlot', at: 1, key: 'a' }, { kind: 'tank', variety: 'pinot', at: 2, key: 'b' }, { kind: 'wordle', number: 5, at: 3, key: 'c' }], now),
    pushText([{ kind: 'harvest', variety: 'riesling', at: 1, key: 'a' }, { kind: 'harvest', variety: 'riesling', at: 1, key: 'b' }], now),
    pushText([{ kind: 'wordle', number: 215, at: 1, key: 'w' }], now),
  ];
  assert.deepEqual(texts[0], { title: 'Урожай Саперави созрел', body: 'И бочка Каберне Совиньон готова — загляните на винодельню', kind: 'harvest' });
  assert.equal(texts[1].title, 'Молодое Пино нуар готово');
  assert.equal(texts[1].body, 'И сусло Мерло готово и ещё 1 — загляните на винодельню');
  assert.equal(texts[2].body, 'Загляните на винодельню');
  assert.deepEqual(texts[3], { title: 'Новое вино дня', body: 'Винордл № 215: сможете угадать с трёх попыток?', kind: 'wordle' });
  for (const t of texts) assert.doesNotMatch(`${t.title} ${t.body}`, /пить|выпе|выпи|бокал|налей|налив|дегуст|купи/i);
});

test('the game itself still runs while pushes are planned (no state mutation)', () => {
  const s = winery();
  const before = JSON.stringify(s);
  plan(s, {}, MSK(10, 5));
  assert.equal(JSON.stringify(s), before);
  applyAction(s, { type: 'tutorial', skip: true }, MSK(10, 5), SPEED);
});
