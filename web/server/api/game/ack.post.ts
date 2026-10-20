// Игрок увидел тосты о наградах из сервиса (отзыв, полка) — больше их не показываем.
import { assertSameOrigin } from "../../utils/auth";
import { useDb } from "../../utils/db";
import { requireGamer } from "../../utils/game";

export default defineEventHandler(async (event) => {
  assertSameOrigin(event);
  const user = await requireGamer(event);
  const sql = await useDb();
  await sql`update game_service_rewards set seen = true where user_id = ${user.id} and not seen`;
  return { ok: true };
});
