// Поля профиля: имя, email, подтверждение 18+ и «публиковать отзывы по умолчанию».
// Телефон меняется только новым входом, аватар — отдельной ручкой.
import { USER_COLUMNS, assertSameOrigin, requireUser, toUser } from "../../utils/auth";
import { useDb } from "../../utils/db";

const EMAIL = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export default defineEventHandler(async (event) => {
  assertSameOrigin(event);
  const user = await requireUser(event);
  const body = (await readBody(event).catch(() => null)) || {};
  const patch: Record<string, unknown> = {};
  if ("name" in body) {
    const name = typeof body.name === "string" ? body.name.trim().replace(/\s+/g, " ") : "";
    if (!name || name.length > 40) throw createError({ statusCode: 400, statusMessage: "Имя — от 1 до 40 символов" });
    patch.name = name;
  }
  if ("email" in body) {
    const email = typeof body.email === "string" ? body.email.trim().toLowerCase() : "";
    if (email && (email.length > 120 || !EMAIL.test(email))) throw createError({ statusCode: 400, statusMessage: "Проверьте email" });
    patch.email = email || null;
  }
  if (body.adult === true && !user.adult) patch.adult_confirmed_at = new Date();
  if (typeof body.defaultShare === "boolean") patch.default_share = body.defaultShare;
  const sql = await useDb();
  if (!Object.keys(patch).length) return { user };
  const [row] = await sql`
    update users set ${sql(patch)}, updated_at = now() where id = ${user.id}
    returning ${sql.unsafe(USER_COLUMNS)}`;
  return { user: toUser(row as never) };
});
