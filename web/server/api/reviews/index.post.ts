// Публикация отзыва: оценка, блюда и заметка с полки, от имени или анонимно. Вино должно быть
// на облачной полке автора — оттуда карточка попадает в общую страницу вина.
import { DISHES } from "~/utils/shelf.mjs";
import { assertSameOrigin, rateLimit, requireUser } from "../../utils/auth";
import { useDb } from "../../utils/db";

export default defineEventHandler(async (event) => {
  assertSameOrigin(event);
  const user = await requireUser(event);
  rateLimit(`review:${user.id}`, 30, 60000);
  if (!user.name || !user.adult) throw createError({ statusCode: 403, statusMessage: "Заполните профиль" });
  const body = await readBody(event).catch(() => null);
  const slug = typeof body?.slug === "string" ? body.slug : "";
  const dishes = Array.isArray(body?.dishes) ? [...new Set(body.dishes as string[])].filter((d) => DISHES.includes(d)) : [];
  const note = typeof body?.note === "string" ? body.note.trim().slice(0, 500) : "";
  if (!slug || slug.length > 500 || typeof body?.liked !== "boolean") {
    throw createError({ statusCode: 400, statusMessage: "Нужна оценка вина" });
  }
  const sql = await useDb();
  const [onShelf] = await sql`select 1 from shelf_entries where user_id = ${user.id} and slug = ${slug} and not deleted`;
  if (!onShelf) throw createError({ statusCode: 409, statusMessage: "Сначала сохраните вино на полку" });
  await sql`
    insert into reviews (user_id, slug, liked, dishes, note, anonymous)
    values (${user.id}, ${slug}, ${body.liked}, ${dishes}, ${note}, ${body.anonymous === true})
    on conflict (user_id, slug) do update set liked = excluded.liked, dishes = excluded.dishes,
      note = excluded.note, anonymous = excluded.anonymous, updated_at = now()`;
  return { ok: true };
});
