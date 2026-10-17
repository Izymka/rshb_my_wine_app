import { USER_COLUMNS, assertSameOrigin, requireUser, toUser } from "../../utils/auth";
import { useDb } from "../../utils/db";

export default defineEventHandler(async (event) => {
  assertSameOrigin(event);
  const user = await requireUser(event);
  const sql = await useDb();
  const [row] = await sql`
    update users set avatar = null, avatar_type = null, avatar_version = 0, updated_at = now()
    where id = ${user.id} returning ${sql.unsafe(USER_COLUMNS)}`;
  return { user: toUser(row as never) };
});
