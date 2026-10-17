// Удаление аккаунта: каскадом уходят сессии, облачная полка и опубликованные отзывы.
// Полка в браузере остаётся — она принадлежит устройству, а не аккаунту.
import { assertSameOrigin, requireUser } from "../../utils/auth";
import { useDb } from "../../utils/db";

export default defineEventHandler(async (event) => {
  assertSameOrigin(event);
  const user = await requireUser(event);
  const sql = await useDb();
  await sql`delete from users where id = ${user.id}`;
  deleteCookie(event, "sv_session", { path: "/" });
  return { ok: true };
});
