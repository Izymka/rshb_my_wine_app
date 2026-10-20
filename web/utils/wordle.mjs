// «Винордл»: одно вино дня на всех, шесть попыток. Каждая попытка — вино из списка; по нему
// видно, совпали ли цвет, регион, сорт и винодельня (жёлтым — винодельня из того же региона).
// После каждой ошибки открывается одна подсказка. Список — российские вина портала «Своё Вино»
// с понятными сортами; ответ дня выбирается по номеру дня, одинаково для всех игроков.
import { dayKey } from './game.mjs';

export const WORDLE_TRIES = 6;
const EPOCH = Date.UTC(2026, 2, 21); // № 1 — 21 марта 2026

export const WORDLE_WINES = [
  { name: 'Красностоп Резерв', winery: 'Лефкадия', region: 'Кубань', grape: 'Красностоп', color: 'Красное', desc: 'терпкий, вишня и специи' },
  { name: 'Совиньон блан', winery: 'Лефкадия', region: 'Кубань', grape: 'Совиньон блан', color: 'Белое', desc: 'крыжовник и свежая трава' },
  { name: 'Quintessence. Вионье', winery: 'Мысхако', region: 'Кубань', grape: 'Вионье', color: 'Белое', desc: 'абрикос и белые цветы' },
  { name: 'Мысхако Каберне', winery: 'Мысхако', region: 'Кубань', grape: 'Каберне Совиньон', color: 'Красное', desc: 'смородина и кедр' },
  { name: 'Саперави', winery: 'Гай-Кодзор', region: 'Кубань', grape: 'Саперави', color: 'Красное', desc: 'густой, ежевика и чернослив' },
  { name: 'Рислинг', winery: 'Гай-Кодзор', region: 'Кубань', grape: 'Рислинг', color: 'Белое', desc: 'зелёное яблоко и лайм' },
  { name: 'Розе', winery: 'Усадьба Дивноморское', region: 'Кубань', grape: 'Пино нуар', color: 'Розовое', desc: 'клубника и розовый грейпфрут' },
  { name: 'Шардоне', winery: 'Усадьба Дивноморское', region: 'Кубань', grape: 'Шардоне', color: 'Белое', desc: 'груша, сливочная текстура' },
  { name: 'Мерло', winery: 'Шато де Талю', region: 'Кубань', grape: 'Мерло', color: 'Красное', desc: 'мягкий, слива и какао' },
  { name: 'Алиготе', winery: 'Шато де Талю', region: 'Кубань', grape: 'Алиготе', color: 'Белое', desc: 'цитрусы и минеральность' },
  { name: 'Брют Империал', winery: 'Абрау-Дюрсо', region: 'Кубань', grape: 'Шардоне', color: 'Игристое', desc: 'бриошь и печёное яблоко' },
  { name: 'Цимлянский чёрный', winery: 'Цимлянские вина', region: 'Дон', grape: 'Цимлянский чёрный', color: 'Красное', desc: 'полевые ягоды и фиалка' },
  { name: 'Красностоп Золотовский', winery: 'Ведерниковъ', region: 'Дон', grape: 'Красностоп', color: 'Красное', desc: 'черешня и чёрный перец' },
  { name: 'Сибирьковый', winery: 'Ведерниковъ', region: 'Дон', grape: 'Сибирьковый', color: 'Белое', desc: 'айва и луговые травы' },
  { name: 'Пухляковский', winery: 'Ведерниковъ', region: 'Дон', grape: 'Пухляковский', color: 'Белое', desc: 'дыня и мёд' },
  { name: 'Голубок', winery: 'Аристов', region: 'Дон', grape: 'Голубок', color: 'Красное', desc: 'тёмные ягоды, бархатный' },
  { name: 'Каберне Совиньон', winery: 'Альма Вэлли', region: 'Крым', grape: 'Каберне Совиньон', color: 'Красное', desc: 'чёрная смородина и мята' },
  { name: 'Пино нуар', winery: 'Альма Вэлли', region: 'Крым', grape: 'Пино нуар', color: 'Красное', desc: 'вишня и лесная подстилка' },
  { name: 'Кокур', winery: 'Солнечная долина', region: 'Крым', grape: 'Кокур', color: 'Белое', desc: 'спелая груша и персик' },
  { name: 'Шардоне', winery: 'Бельбек', region: 'Крым', grape: 'Шардоне', color: 'Белое', desc: 'дыня и ореховые ноты' },
  { name: 'Кефесия', winery: 'Бельбек', region: 'Крым', grape: 'Кефесия', color: 'Красное', desc: 'гранат и сухие травы' },
  { name: 'Ркацители', winery: 'Инкерман', region: 'Крым', grape: 'Ркацители', color: 'Белое', desc: 'яблоко и айва' },
  { name: 'Саперави', winery: 'Инкерман', region: 'Крым', grape: 'Саперави', color: 'Красное', desc: 'чернослив и табак' },
  { name: 'Шабли-стиль Шардоне', winery: 'Скалистый берег', region: 'Крым', grape: 'Шардоне', color: 'Белое', desc: 'устрица и лимон' },
  { name: 'Рислинг', winery: 'Сикоры', region: 'Кубань', grape: 'Рислинг', color: 'Белое', desc: 'петролейные ноты и персик' },
  { name: 'Мерло', winery: 'Сикоры', region: 'Кубань', grape: 'Мерло', color: 'Красное', desc: 'вишня и ваниль' },
  { name: 'Шардоне', winery: 'Галицкий и Галицкий', region: 'Кубань', grape: 'Шардоне', color: 'Белое', desc: 'жёлтые фрукты и дуб' },
  { name: 'Каберне Совиньон', winery: 'Галицкий и Галицкий', region: 'Кубань', grape: 'Каберне Совиньон', color: 'Красное', desc: 'смородина, кожа и кофе' },
  { name: 'Цимлянский игристый', winery: 'Цимлянские вина', region: 'Дон', grape: 'Цимлянский чёрный', color: 'Игристое', desc: 'красные ягоды, сладковатое' },
  { name: 'Совиньон блан', winery: 'Фанагория', region: 'Кубань', grape: 'Совиньон блан', color: 'Белое', desc: 'бузина и маракуйя' },
  { name: 'Цимлянский чёрный', winery: 'Фанагория', region: 'Кубань', grape: 'Цимлянский чёрный', color: 'Красное', desc: 'слива и пряности' },
  { name: 'Мускат белый', winery: 'Массандра', region: 'Крым', grape: 'Мускат', color: 'Белое', desc: 'сладкий, чайная роза и мёд' },
  { name: 'Рислинг', winery: 'Узунов', region: 'Кубань', grape: 'Рислинг', color: 'Белое', desc: 'лайм и белые цветы' },
  { name: 'Красностоп', winery: 'Раевское', region: 'Кубань', grape: 'Красностоп', color: 'Красное', desc: 'кизил и чёрный перец' },
  { name: 'Розе Пино нуар', winery: 'Шато де Талю', region: 'Кубань', grape: 'Пино нуар', color: 'Розовое', desc: 'малина и арбуз' },
  { name: 'Ркацители', winery: 'Кузьмин', region: 'Крым', grape: 'Ркацители', color: 'Белое', desc: 'полевые цветы и цитрус' },
].map((w, i) => ({ id: i + 1, ...w }));

export function wordleNumber(now) {
  return Math.floor((Date.parse(`${dayKey(now)}T00:00:00Z`) - EPOCH) / 86400000) + 1;
}

// Ответ дня — перестановка по номеру дня, чтобы вина не повторялись подряд.
export function wordleAnswer(number) {
  const n = WORDLE_WINES.length;
  const idx = ((number * 17 + 5) % n + n) % n;
  return WORDLE_WINES[idx];
}

export function findWine(id) { return WORDLE_WINES.find(w => w.id === Number(id)) || null; }
export function wineLabel(w) { return `${w.name} · ${w.winery}`; }

// Сравнение попытки с ответом: ok — совпало, near — винодельня из того же региона, no — мимо.
export function compare(guess, answer) {
  return [
    { t: guess.color, s: guess.color === answer.color ? 'ok' : 'no' },
    { t: guess.region, s: guess.region === answer.region ? 'ok' : 'no' },
    { t: guess.grape, s: guess.grape === answer.grape ? 'ok' : 'no' },
    { t: guess.winery, s: guess.winery === answer.winery ? 'ok' : guess.region === answer.region ? 'near' : 'no' },
  ];
}

// Итог по списку попыток: строки, подсказки, выигрыш или проигрыш. answer раскрывается только
// после конца игры, чтобы его нельзя было подсмотреть в ответе сервера.
export function wordleView(number, guessIds) {
  const answer = wordleAnswer(number);
  const guesses = guessIds.map(findWine).filter(Boolean);
  const solvedAt = guesses.findIndex(g => g.id === answer.id);
  const solved = solvedAt >= 0;
  const lost = !solved && guesses.length >= WORDLE_TRIES;
  const wrong = solved ? 5 : guesses.length;
  const hints = [['Цвет', answer.color], ['Регион', answer.region], ['Сорт', answer.grape], ['Винодельня', answer.winery], ['Описание', answer.desc]]
    .map(([label, value], i) => (i < wrong ? { label, value } : { label, value: null }));
  return {
    number, solved, lost, attempt: solved ? solvedAt + 1 : null,
    rows: guesses.map(g => ({ id: g.id, name: wineLabel(g), cells: compare(g, answer) })),
    hints, answer: solved || lost ? answer : null,
  };
}
