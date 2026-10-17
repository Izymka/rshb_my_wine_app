// «Что говорят другие»: сколько человек держат вино на полке, доля «понравилось», с чем пьют
// и последние отзывы. Свой отзыв — первым и с пометкой mine. ?all=1 — все отзывы (до 200).
import { avatarUrl, currentUser } from "../../utils/auth";
import { useDb } from "../../utils/db";

export default defineEventHandler(async (event) => {
  setResponseHeader(event, "Cache-Control", "no-store");
  const slug = getRouterParam(event, "slug", { decode: true }) || "";
  if (!slug || slug.length > 500) throw createError({ statusCode: 404 });
  const all = getQuery(event).all === "1";
  const sql = await useDb();
  const me = await currentUser(event);
  const meId = me?.id ?? "0";

  const [[stats], pairings, rows, [cardRow]] = await Promise.all([
    sql<{ added: number; reviews: number; liked: number }[]>`
      select (select count(*)::int from shelf_entries where slug = ${slug} and not deleted) as added,
        count(*)::int as reviews, count(*) filter (where liked)::int as liked
      from reviews where slug = ${slug}`,
    sql<{ dish: string; count: number }[]>`
      select dish, count(*)::int as count from reviews, unnest(dishes) as dish
      where slug = ${slug} and dish <> 'Без еды' group by dish order by count desc, dish limit 4`,
    sql`
      select r.user_id, r.liked, r.dishes, r.note, r.anonymous, r.updated_at, u.name, u.avatar_version
      from reviews r join users u on u.id = r.user_id where r.slug = ${slug}
      order by (r.user_id = ${meId}) desc, r.updated_at desc limit ${all ? 200 : 3}`,
    sql`select card from shelf_entries where slug = ${slug} order by updated_at desc limit 1`,
  ]);
  const reviews = rows.map((r) => {
    const mine = String(r.user_id) === meId;
    const anonymous = r.anonymous || !r.name;
    return {
      mine, liked: r.liked, dishes: r.dishes, note: r.note, anonymous,
      name: anonymous ? "Анонимно" : r.name,
      avatarUrl: anonymous ? null : avatarUrl(String(r.user_id), r.avatar_version),
      date: r.updated_at.toISOString(),
    };
  });
  const [mine] = me ? await sql`select liked, dishes, note, anonymous from reviews where user_id = ${me.id} and slug = ${slug}` : [];
  return {
    added: stats.added,
    reviewCount: stats.reviews,
    likedPct: stats.reviews ? Math.round((stats.liked / stats.reviews) * 100) : null,
    pairings: pairings.map((p) => ({ dish: p.dish, pct: Math.round((p.count / stats.reviews) * 100) })),
    reviews,
    mine: mine || null,
    card: cardRow?.card || null,
  };
});
