import { useDb } from "../../utils/db";

export default defineEventHandler(async (event) => {
  const id = getRouterParam(event, "id") || "";
  if (!/^\d{1,18}$/.test(id)) throw createError({ statusCode: 404 });
  const sql = await useDb();
  const [row] = await sql<{ avatar: Buffer; avatar_type: string }[]>`
    select avatar, avatar_type from users where id = ${id} and avatar is not null`;
  if (!row) throw createError({ statusCode: 404 });
  setResponseHeader(event, "Content-Type", row.avatar_type);
  // В URL есть версия (?v=), поэтому картинку можно кэшировать надолго.
  setResponseHeader(event, "Cache-Control", "public, max-age=31536000, immutable");
  setResponseHeader(event, "X-Content-Type-Options", "nosniff");
  return row.avatar;
});
