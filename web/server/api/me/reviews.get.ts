import { requireUser } from "../../utils/auth";
import { useDb } from "../../utils/db";

export default defineEventHandler(async (event) => {
  setResponseHeader(event, "Cache-Control", "no-store");
  const user = await requireUser(event);
  const sql = await useDb();
  const rows = await sql`
    select r.slug, r.liked, r.dishes, r.note, r.anonymous, r.updated_at,
      coalesce(s.card, (select card from shelf_entries x where x.slug = r.slug order by updated_at desc limit 1)) as card
    from reviews r left join shelf_entries s on s.user_id = r.user_id and s.slug = r.slug
    where r.user_id = ${user.id} order by r.updated_at desc limit 500`;
  return {
    reviews: rows.map((r) => ({
      slug: r.slug, liked: r.liked, dishes: r.dishes, note: r.note, anonymous: r.anonymous,
      updatedAt: r.updated_at.toISOString(), name: r.card?.name || r.slug, winery: r.card?.winery || "",
    })),
  };
});
