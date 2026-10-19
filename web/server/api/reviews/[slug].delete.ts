import { assertSameOrigin, requireUser } from "../../utils/auth";
import { useDb } from "../../utils/db";

export default defineEventHandler(async (event) => {
  assertSameOrigin(event);
  const user = await requireUser(event);
  const slug = getRouterParam(event, "slug", { decode: true }) || "";
  const sql = await useDb();
  await sql`delete from reviews where user_id = ${user.id} and slug = ${slug}`;
  return { ok: true };
});
