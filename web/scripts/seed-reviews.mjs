#!/usr/bin/env node
// Синтетические отзывы для демо: авторы, их облачные полки и отзывы, чтобы блок «Что говорят
// другие» был заполнен у большинства вин. Список вин — открытый каталог портала «Своё Вино»
// (api.vino-svoe.ru/v1/wines), доля «понравилось» тянется к народному рейтингу вина, так что
// отзывы в основном положительные и не спорят с рейтингом в карточке.
//
//   NUXT_DATABASE_URL=postgres://wine:wine@127.0.0.1:5432/wine node scripts/seed-reviews.mjs
//
// Авторы заводятся с телефоном вида «seed:0001»: войти под ними нельзя (вход нормализует
// телефон в цифры), а повторный запуск сначала удаляет прежних синтетических авторов вместе
// с их полками и отзывами. Настоящие пользователи и их отзывы не трогаются.
//
// Скрипт самодостаточный (из пакетов — только postgres): образ web кладёт его рядом с
// собранным сервером, и в production он запускается в контейнере web с его NUXT_DATABASE_URL:
//   docker exec wine-scanner-web node .output/server/seed-reviews.mjs
//
// Флаги: --coverage 0.85 (доля вин с отзывами), --authors 240, --seed 42 (повторяемый набор),
// --wines файл.json (список вин вместо портала, формат ответа items[]), --clear (только удалить),
// --dry-run (посчитать и ничего не писать).
import { readFile } from "node:fs/promises";
import postgres from "postgres";

const PORTAL = "https://api.vino-svoe.ru/v1/wines";
const SEED_PHONE = "seed:";
const DAY = 86400000;

const args = parseArgs(process.argv.slice(2));
const coverage = Number(args.coverage ?? 0.85);
const authorCount = Number(args.authors ?? 240);
const random = mulberry32(Number(args.seed ?? 42));

function parseArgs(list) {
  const out = {};
  for (let i = 0; i < list.length; i++) {
    const key = list[i].replace(/^--/, "");
    out[key] = list[i + 1] && !list[i + 1].startsWith("--") ? list[++i] : true;
  }
  return out;
}

function mulberry32(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const pick = (list) => list[Math.floor(random() * list.length)];
const chance = (p) => random() < p;
function sample(list, n) {
  const copy = [...list];
  const out = [];
  while (out.length < n && copy.length) out.push(copy.splice(Math.floor(random() * copy.length), 1)[0]);
  return out;
}

async function loadWines() {
  if (typeof args.wines === "string") {
    const data = JSON.parse(await readFile(args.wines, "utf8"));
    return Array.isArray(data) ? data : data.items;
  }
  const wines = [];
  for (let page = 1; ; page++) {
    const response = await fetch(`${PORTAL}?page=${page}&perPage=30`, { signal: AbortSignal.timeout(30000) });
    if (!response.ok) throw new Error(`портал ответил ${response.status} на странице ${page}`);
    const data = await response.json();
    wines.push(...data.items);
    process.stdout.write(`\rкаталог портала: ${wines.length} из ${data.totalItems}`);
    if (page >= data.totalPages) break;
  }
  process.stdout.write("\n");
  return wines;
}

// --- Авторы -------------------------------------------------------------------------------------

const WOMEN = ["Анна", "Мария", "Екатерина", "Ольга", "Наталья", "Татьяна", "Ирина", "Елена", "Светлана", "Юлия",
  "Ксения", "Дарья", "Алина", "Виктория", "Полина", "Марина", "Анастасия", "Вера", "Софья", "Лариса", "Жанна", "Яна"];
const MEN = ["Алексей", "Дмитрий", "Сергей", "Андрей", "Михаил", "Илья", "Иван", "Павел", "Артём", "Никита",
  "Константин", "Владимир", "Роман", "Евгений", "Максим", "Олег", "Григорий", "Антон", "Тимур", "Кирилл", "Георгий"];
const INITIALS = "АБВГДЕЗИКЛМНОПРСТФХЧШЮЯ";

function makeAuthors(n) {
  return Array.from({ length: n }, (_, i) => {
    const first = pick(chance(0.5) ? WOMEN : MEN);
    // Кто-то подписывается одним именем, кто-то с инициалом фамилии.
    const name = chance(0.6) ? `${first} ${pick([...INITIALS])}.` : first;
    return {
      phone: `${SEED_PHONE}${String(i + 1).padStart(4, "0")}`,
      name,
      anonymity: chance(0.15) ? 0.8 : 0.05, // одни почти всегда анонимны, другие почти никогда
      activity: 0.3 + random() * random() * 4, // немногие пишут много, большинство — пару отзывов
      since: Date.now() - (60 + random() * 300) * DAY,
    };
  });
}

// --- Вино: стиль, блюда, тексты -----------------------------------------------------------------

function styleOf(category = "") {
  const c = category.toLowerCase();
  if (/брют|игрист|петнат/.test(c)) return "sparkling";
  if (/сладк|десерт|ликёр|ликер/.test(c) && !/полусух/.test(c)) return "sweet";
  if (c.startsWith("красн")) return "red";
  if (c.startsWith("розов")) return "rose";
  if (c.startsWith("оранж")) return "orange";
  return "white";
}

const PAIRINGS = {
  red: ["Мясо", "Мясо", "Сыры", "Паста", "Птица", "Овощи"],
  white: ["Рыба", "Морепродукты", "Сыры", "Птица", "Овощи", "Паста"],
  rose: ["Овощи", "Рыба", "Сыры", "Морепродукты", "Паста", "Птица"],
  orange: ["Сыры", "Птица", "Овощи", "Мясо", "Рыба"],
  sparkling: ["Морепродукты", "Сыры", "Рыба", "Без еды", "Десерт"],
  sweet: ["Десерт", "Десерт", "Сыры", "Без еды"],
};

function dishesFor(style) {
  if (chance(0.12)) return [];
  if (chance(0.1)) return ["Без еды"];
  const out = sample(PAIRINGS[style].filter((d) => d !== "Без еды"), chance(0.55) ? 1 : 2);
  if (chance(0.05)) out.push("Другое");
  return [...new Set(out)];
}

const LIKED = {
  any: [
    "Очень понравилось, возьму ещё.", "Приятно удивило, особенно за свои деньги.", "Отличное вино, рекомендую.",
    "Открыли с друзьями — бутылка ушла за вечер.", "Взяли по совету на дегустации и не пожалели.",
    "Сбалансированное и чистое, ничего лишнего.", "Хорошее российское вино, приятно, что такое делают у нас.",
    "Брали в подарок, в итоге оставили себе.", "Долгое приятное послевкусие.", "Покупаю уже не первый раз — стабильно хорошо.",
    "Мягкое и питкое, без резкости.", "Классное вино на выходные.", "Понравилось даже тем, кто обычно не пьёт вино.",
    "Отличное соотношение цены и качества.", "Будет в домашнем запасе.",
  ],
  red: [
    "Мягкие танины, вишня и немного специй.", "Плотное, с тонами чернослива и дуба. Раскрылось через полчаса в бокале.",
    "Отлично к стейку.", "Ягоды, немного табака и ваниль — всё как я люблю.", "Насыщенное, но не тяжёлое.",
    "Хорошо раскрылось после декантера.", "К шашлыку — идеально.",
  ],
  white: [
    "Свежее, с цитрусом и зелёным яблоком.", "Хорошая кислотность, отлично к рыбе.", "Цветочный аромат, лёгкое и освежающее.",
    "Минеральное, чистое, летнее.", "Тона груши и белых цветов, очень приятное.", "Идеально охлаждённым на террасе.",
  ],
  rose: [
    "Клубника и малина в аромате, очень летнее.", "Лёгкое и свежее — то, что нужно в жару.",
    "Красивый цвет, приятная ягодная кислинка.", "Отлично на пикник.",
  ],
  orange: [
    "Необычное, с тонами сухофруктов и чая. Интересный опыт.", "Плотное для белого, с приятной горчинкой.",
    "Понравилось к сырам, хорошая текстура.",
  ],
  sparkling: [
    "Тонкий перляж, свежее и бодрое.", "Хрустящее, с нотами яблока и бриоши.", "Отличная альтернатива шампанскому.",
    "Открыли на праздник — все спрашивали, что за игристое.", "Сухое, свежее, пузырьки держатся долго.",
  ],
  sweet: [
    "Мёд и абрикос, сладость не приторная.", "Отлично к десерту.", "Насыщенное и ароматное, по чуть-чуть к сырам — супер.",
  ],
};
const DISLIKED = [
  "Не моё — показалось слишком кислым.", "За эту цену ждёшь большего.", "Простоватое, без послевкусия.",
  "Слишком терпкое для меня.", "Нормально, но второй раз брать не буду.", "Немного спиртуозное, на мой вкус.",
  "Аромат понравился, вкус — нет.", "Возможно, не повезло с бутылкой.",
];

function noteFor(style, liked) {
  if (chance(0.25)) return ""; // часть людей ставит только оценку
  if (!liked) return pick(DISLIKED);
  const parts = [pick(chance(0.6) ? LIKED[style] : LIKED.any)];
  if (chance(0.45)) parts.push(pick(LIKED.any));
  return [...new Set(parts)].join(" ");
}

// Народный рейтинг портала задаёт долю «понравилось»: 5 → ~93%, 4 → ~78%, 3 → ~55%.
function likeChance(rating) {
  if (typeof rating !== "number" || rating <= 0) return 0.85;
  return Math.min(0.95, Math.max(0.2, 0.93 - (5 - rating) * 0.17));
}

// Сколько отзывов у вина: у большинства 1–4, у заметных вин — до двух десятков.
function reviewCount(wine) {
  const base = 1 + Math.floor(-Math.log(1 - random()) * 2.5);
  const boost = typeof wine.publicRating === "number" && wine.publicRating >= 4.8 ? 1.4 : 1;
  return Math.min(24, Math.round(base * boost));
}

function card(wine) {
  return {
    name: wine.title, winery: wine.manufacturer || "", category: wine.category || "",
    region: wine.region || "", slug: wine.slug,
  };
}

// --- Сборка ----------------------------------------------------------------------------------

function build(wines, authors) {
  const shelf = new Map(); // `${author}|${slug}` -> строка shelf_entries
  const reviews = [];
  const weights = authors.map((a) => a.activity);
  const total = weights.reduce((s, w) => s + w, 0);
  const author = () => {
    let x = random() * total;
    for (let i = 0; i < authors.length; i++) if ((x -= weights[i]) <= 0) return i;
    return authors.length - 1;
  };

  for (const wine of wines) {
    if (!wine?.slug || !wine.title) continue;
    const style = styleOf(wine.category);
    const reviewed = chance(coverage);
    const n = reviewed ? reviewCount(wine) : 0;
    // Кроме отзывов вино лежит на полках «хочу попробовать» / «есть дома» — счётчик «добавили».
    const shelfOnly = Math.floor(random() * (n + 2) * 0.8);
    const used = new Set();
    for (let k = 0; k < n + shelfOnly; k++) {
      let a = author();
      for (let tries = 0; used.has(a) && tries < 20; tries++) a = author();
      if (used.has(a)) continue;
      used.add(a);
      const who = authors[a];
      const at = who.since + random() * (Date.now() - who.since - DAY);
      const key = `${a}|${wine.slug}`;
      if (k < n) {
        const liked = chance(likeChance(wine.publicRating));
        const dishes = dishesFor(style);
        const note = noteFor(style, liked);
        reviews.push({ a, slug: wine.slug, liked, dishes, note, anonymous: chance(who.anonymity), at });
        shelf.set(key, {
          a, slug: wine.slug, card: card(wine), status: "tried", reaction: liked ? "liked" : "disliked",
          dishes, note, added_at: Math.round(at - random() * 20 * DAY), updated_at: Math.round(at),
        });
      } else {
        shelf.set(key, {
          a, slug: wine.slug, card: card(wine), status: chance(0.6) ? "want" : "home", reaction: null,
          dishes: [], note: "", added_at: Math.round(at), updated_at: Math.round(at),
        });
      }
    }
  }
  return { reviews, shelf: [...shelf.values()] };
}

async function main() {
  const url = process.env.NUXT_DATABASE_URL;
  if (!url && !args["dry-run"]) throw new Error("нужен NUXT_DATABASE_URL (например, из web/.env)");
  const sql = url ? postgres(url, { max: 2, onnotice: () => {} }) : null;
  try {
    if (sql) {
      const [{ ready }] = await sql`select to_regclass('public.reviews') is not null as ready`;
      if (!ready) throw new Error("схемы ещё нет: откройте приложение со входом один раз, сервер создаст таблицы");
      const removed = await sql`delete from users where phone like ${SEED_PHONE + "%"} returning id`;
      console.log(`удалены прежние синтетические авторы: ${removed.length}`);
      if (args.clear) return;
    }

    const wines = await loadWines();
    const authors = makeAuthors(authorCount);
    const { reviews, shelf } = build(wines, authors);
    const covered = new Set(reviews.map((r) => r.slug)).size;
    const liked = reviews.filter((r) => r.liked).length;
    console.log(`вин: ${wines.length}, с отзывами: ${covered} (${Math.round((covered / wines.length) * 100)}%)`);
    console.log(`отзывов: ${reviews.length}, «понравилось»: ${Math.round((liked / reviews.length) * 100)}%, на полках: ${shelf.length}`);
    if (!sql) return;

    await sql.begin(async (tx) => {
      const ids = await tx`
        insert into users ${tx(authors.map((a) => ({
          phone: a.phone, name: a.name, adult_confirmed_at: new Date(a.since), created_at: new Date(a.since),
        })))} returning id`;
      const userId = (a) => ids[a].id;
      for (let i = 0; i < shelf.length; i += 2000) {
        await tx`insert into shelf_entries ${tx(shelf.slice(i, i + 2000).map((s) => ({
          user_id: userId(s.a), slug: s.slug, card: s.card, analogues: [], status: s.status,
          reaction: s.reaction, dishes: s.dishes, note: s.note, remind_at: null,
          added_at: s.added_at, updated_at: s.updated_at, deleted: false,
        })))}`;
      }
      for (let i = 0; i < reviews.length; i += 2000) {
        await tx`insert into reviews ${tx(reviews.slice(i, i + 2000).map((r) => ({
          user_id: userId(r.a), slug: r.slug, liked: r.liked, dishes: r.dishes, note: r.note,
          anonymous: r.anonymous, created_at: new Date(r.at), updated_at: new Date(r.at),
        })))}`;
      }
    });
    console.log(`записано: ${authors.length} авторов, ${shelf.length} вин на полках, ${reviews.length} отзывов`);
  } finally {
    await sql?.end();
  }
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
