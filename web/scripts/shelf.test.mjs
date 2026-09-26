import assert from 'node:assert/strict';
import { test } from 'node:test';
import { addWine, updateWine, parseShelf, remindersFor, similarKnown, DAY, winePath } from '../utils/shelf.mjs';
const card = { name: 'Тестовое вино', category: 'Красное', winery: 'А', grapes: 'Мерло' };
test('saved cards survive a reload; a repeated scan preserves date, rating, notes and reminder', () => {
  let shelf = addWine([], 'wine', card, [], 1000);
  assert.equal(shelf[0].remindAt, 1000 + 3 * DAY);
  shelf = updateWine(shelf, 'wine', { reaction: 'liked', dishes: ['Сыры'], note: 'К ужину' });
  shelf = parseShelf(JSON.stringify({ version: 1, entries: shelf }));
  assert.equal(shelf[0].status, 'tried');
  assert.equal(shelf[0].remindAt, null);
  assert.equal(remindersFor(shelf).length, 0);
  const again = addWine(shelf, 'wine', card, [], 2000);
  assert.deepEqual(again, shelf);
  assert.equal(again[0].addedAt, 1000);
  assert.equal(again[0].note, 'К ужину');
  assert.deepEqual(again[0].dishes, ['Сыры']);
});
test('moving a rated wine home preserves its review, cancellation and original identity', () => {
  const shelf = updateWine(addWine([], 'wine', card), 'wine', { reaction: 'disliked' });
  const next = updateWine(shelf, 'wine', { status: 'home', slug: 'evil', addedAt: 0, remindAt: 123 });
  assert.equal(next[0].status, 'home');
  assert.equal(next[0].reaction, 'disliked');
  assert.equal(next[0].slug, 'wine');
  assert.equal(next[0].addedAt, shelf[0].addedAt);
  assert.equal(next[0].remindAt, null);
});
test('cancel, snooze and delete generate the correct push queue snapshot', () => {
  let shelf = addWine([], 'wine', card, [], 1000);
  const firstId = remindersFor(shelf)[0].id;
  shelf = updateWine(shelf, 'wine', { remindAt: 1000 + 7 * DAY });
  assert.notEqual(remindersFor(shelf)[0].id, firstId);
  shelf = updateWine(shelf, 'wine', { remindAt: null });
  assert.deepEqual(remindersFor(shelf), []);
  assert.deepEqual(remindersFor([]), []);
});
test('corrupt data is rejected instead of silently erasing user history', () => {
  for (const raw of ['bad', '{}', '{"version":2,"entries":[]}', '{"version":1,"entries":[{}]}']) assert.throws(() => parseShelf(raw));
  const shelf = addWine([], 'wine', card);
  assert.throws(() => parseShelf(JSON.stringify({ version: 1, entries: [...shelf, ...shelf] })));
});
test('manually selected cards use their own characteristics, never stale scanner analogues', () => {
  const result = similarKnown('wine', card, [
    { slug: 'white', name: 'Белое', category: 'Белое', winery: 'Б' },
    { slug: 'same', ...card },
    { slug: 'red', ...card, winery: 'В' },
  ]);
  assert.deepEqual(result.map(r => r.slug), ['red']);
  assert.equal(winePath('a/b c'), '/wines/a%2Fb%20c');
});
