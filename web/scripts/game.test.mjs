import assert from 'node:assert/strict';
import { test } from 'node:test';
import { BALANCE, GameError, VARIETIES, applyAction, dayKey, harvestYield, itemValue, levelOf, newGame, plotStage, serviceReward, shopCatalog, skipCost, varietyAccess, varietiesFrom, applyWordle } from '../utils/game.mjs';
import { WORDLE_WINES, compare, wordleAnswer, wordleView } from '../utils/wordle.mjs';

const T0 = Date.UTC(2026, 9, 19, 9);
const H = 3600000;
const act = (s, a, now = T0, ctx) => applyAction(s, a, now, 1, ctx);

test('a new winery starts with a ripe vine, a growing one and the onboarding at step 0', () => {
  const s = newGame('seed', T0);
  assert.equal(plotStage(s.plots[0], T0), 'ripe');
  assert.equal(plotStage(s.plots[1], T0), 'sapling');
  assert.equal(s.plots.filter(p => p.owned).length, BALANCE.plots.start);
  assert.equal(s.orders.slots.length, BALANCE.orders.base);
  assert.ok(s.orders.slots.every(sl => sl.order));
  assert.equal(s.daily.quests.length, 3);
  assert.equal(s.tutorial.step, 0);
});

test('the production chain: harvest → press → tank, with timers enforced', () => {
  const s = newGame('chain', T0);
  act(s, { type: 'tutorial', step: 0 });
  const out = act(s, { type: 'harvest', plot: 0 });
  assert.equal(out.items['grape:chardonnay'], VARIETIES.chardonnay.yld);
  assert.equal(s.items['grape:chardonnay'], 6);
  assert.equal(s.tutorial.step, 2);
  assert.throws(() => act(s, { type: 'harvest', plot: 1 }), GameError);
  act(s, { type: 'start', building: 'press', variety: 'chardonnay' });
  assert.equal(s.items['grape:chardonnay'], 3);
  assert.throws(() => act(s, { type: 'start', building: 'press', variety: 'chardonnay' }), /слоты/);
  assert.throws(() => act(s, { type: 'collect', building: 'press' }, T0 + 60000), /нечего/);
  act(s, { type: 'collect', building: 'press' }, T0 + 31 * 60000);
  assert.equal(s.items['must:chardonnay'], 1);
  act(s, { type: 'start', building: 'tank', variety: 'chardonnay' }, T0 + H);
  act(s, { type: 'collect', building: 'tank' }, T0 + 5 * H);
  assert.equal(s.items['young:chardonnay'], 1);
  assert.equal(s.stats.fermented, 1);
});

test('planting uses a sapling first, then coins; locked varieties cannot be planted', () => {
  const s = newGame('plant', T0);
  const coins = s.coins;
  act(s, { type: 'plant', plot: 2, variety: 'chardonnay' });
  assert.equal(s.saplings.chardonnay, 1);
  assert.equal(s.coins, coins);
  act(s, { type: 'plant', plot: 3, variety: 'chardonnay' });
  act(s, { type: 'plant', plot: 4, variety: 'chardonnay' });
  assert.equal(s.coins, coins - VARIETIES.chardonnay.price);
  assert.throws(() => act(s, { type: 'plant', plot: 5, variety: 'krasnostop' }), /не открыт/);
  assert.throws(() => act(s, { type: 'plant', plot: 6, variety: 'chardonnay' }), /не куплен/);
});

test('a scan unlocks an autochthonous variety and gives a sapling; repeat scans only give a sapling', () => {
  const s = newGame('scan', T0);
  assert.equal(varietyAccess(s, 'krasnostop').open, false);
  const r = serviceReward(s, 'scan', { grapes: 'Красностоп золотовский, Каберне Совиньон' }, T0);
  assert.deepEqual(r.unlocked, ['krasnostop', 'cabernet']);
  assert.equal(s.saplings.krasnostop, 1);
  assert.equal(s.grozdi, BALANCE.start.grozdi + 5);
  assert.equal(varietyAccess(s, 'krasnostop').source, 'scan');
  const again = serviceReward(s, 'scan', { grapes: 'Красностоп' }, T0);
  assert.deepEqual(again.unlocked, []);
  assert.equal(s.saplings.krasnostop, 2);
  assert.deepEqual(varietiesFrom('Шардоне 60%, Пино нуар 40%'), ['chardonnay', 'pinot']);
  const blend = serviceReward(s, 'scan', { grapes: 'Алиготе, Мюллер-Тургау, Пино Блан, Рислинг, Шардоне' }, T0);
  assert.deepEqual(blend.unlocked, ['aligote', 'riesling']);
});

test('orders pay coins and XP, then the slot rests for two hours', () => {
  const s = newGame('orders', T0);
  const slot = s.orders.slots[0];
  const o = slot.order;
  s.items = { [o.item]: o.count, 'grape:riesling': 5 };
  const coins = s.coins;
  act(s, { type: 'fulfill', slot: 0 });
  assert.equal(s.coins, coins + o.coins);
  assert.equal(slot.order, null);
  act(s, { type: 'sell', item: 'grape:riesling', count: 1 }, T0 + H);
  assert.equal(s.orders.slots[0].order, null);
  act(s, { type: 'sell', item: 'grape:riesling', count: 1 }, T0 + 2 * H + 1);
  assert.ok(s.orders.slots[0].order);
  assert.equal(s.daily.counters.orders, 1);
});

test('a shelf “want” wine becomes a guest order that asks to scan a locked variety', () => {
  const s = newGame('shelf', T0);
  s.orders.slots = [];
  const wants = [{ slug: 'krasnostop-reserv', name: 'Красностоп Резерв', winery: 'Лефкадия', grapes: 'Красностоп' }];
  let found = null;
  for (let i = 0; i < 20 && !found; i++) {
    s.orders.slots = [];
    act(s, { type: 'settings' }, T0, { wants });
    found = s.orders.slots.find(sl => sl.order?.shelf)?.order;
  }
  assert.ok(found, 'a shelf order appears');
  assert.equal(found.item, 'young:krasnostop');
  assert.equal(found.mult, 1.5);
});

test('starlings cut the harvest unless shooed, overripe halves it', () => {
  const plot = { variety: 'cabernet', plantedAt: 0, readyAt: 8 * H, problemAt: 2 * H, problemFixed: false };
  assert.equal(plotStage(plot, 3 * H), 'problem');
  assert.equal(harvestYield(plot, 8 * H), Math.round(4 * 0.8));
  assert.equal(harvestYield({ ...plot, problemFixed: true }, 8 * H), 4);
  assert.equal(plotStage({ ...plot, problemFixed: true }, 8 * H + 49 * H), 'overripe');
  assert.equal(harvestYield({ ...plot, problemFixed: true }, 8 * H + 49 * H), 2);
});

test('speed makes timers proportionally shorter', () => {
  const s = newGame('speed', T0, 60);
  applyAction(s, { type: 'plant', plot: 2, variety: 'chardonnay' }, T0, 60);
  assert.equal(s.plots[2].readyAt - T0, 2 * 60000);
});

test('levels, quests, chest, calendar and skip cost', () => {
  assert.equal(levelOf(0), 1);
  assert.equal(levelOf(100), 2);
  assert.equal(levelOf(960), 5);
  const s = newGame('daily', T0);
  s.daily.quests.forEach(q => { s.daily.counters[q.kind] = q.target; });
  for (let i = 0; i < 3; i++) act(s, { type: 'quest', index: i });
  assert.throws(() => act(s, { type: 'quest', index: 0 }), /уже/);
  const g = s.grozdi;
  act(s, { type: 'chest' });
  assert.equal(s.grozdi, g + 15 + (levelOf(s.xp) > 1 ? 3 * (levelOf(s.xp) - 1) : 0) - 0);
  act(s, { type: 'calendar' });
  assert.throws(() => act(s, { type: 'calendar' }), /уже/);
  act(s, { type: 'calendar' }, T0 + 24 * H);
  assert.equal(s.saplings.chardonnay, 3);
  assert.equal(s.daily.day, dayKey(T0 + 24 * H));
  assert.equal(skipCost(T0 + 25 * 60000, T0), 1);
  assert.equal(skipCost(T0 + 4 * H, T0), 10);
});

test('the shop sells the next plot only after the previous one and respects levels', () => {
  const s = newGame('shop', T0);
  s.coins = 5000;
  const cat = shopCatalog(s);
  assert.equal(cat.plots[0].lock, null);
  assert.ok(cat.plots[1].lock);
  act(s, { type: 'buy', item: 'plot:7' });
  assert.equal(s.plots[6].owned, true);
  assert.match(shopCatalog(s).plots[1].lock, /Уровень 6/);
  act(s, { type: 'buy', item: 'storage' });
  assert.equal(s.storage, BALANCE.storage.start + BALANCE.storage.step);
  assert.throws(() => act(s, { type: 'buy', item: 'barrel' }), /Уровень/);
  act(s, { type: 'buy', item: 'decor:scarecrow' });
  assert.throws(() => act(s, { type: 'buy', item: 'decor:scarecrow' }), /Уже/);
  assert.ok(itemValue('aged:cabernet') > itemValue('young:cabernet'));
});

test('the storage limit blocks a harvest that does not fit', () => {
  const s = newGame('full', T0);
  s.items = { 'grape:riesling': s.storage };
  assert.throws(() => act(s, { type: 'harvest', plot: 0 }), /Склад полон/);
});

test('wordle: same wine for everyone per day, hints open after each miss, streak grows day to day', () => {
  const answer = wordleAnswer(214);
  const other = WORDLE_WINES.find(w => w.region !== answer.region && w.color !== answer.color);
  assert.equal(compare(answer, answer).every(c => c.s === 'ok'), true);
  assert.equal(compare(other, answer)[1].s, 'no');
  let v = wordleView(214, [other.id]);
  assert.equal(v.hints.filter(h => h.value).length, 1);
  assert.equal(v.answer, null);
  v = wordleView(214, [other.id, answer.id]);
  assert.equal(v.solved, true);
  assert.equal(v.attempt, 2);
  const s = newGame('wordle', T0);
  const r = applyWordle(s, '2026-10-19', 2, T0);
  assert.equal(r.gained.grozdi, 22);
  applyWordle(s, '2026-10-20', 1, T0 + 24 * H);
  assert.equal(s.wordle.streak, 2);
});
