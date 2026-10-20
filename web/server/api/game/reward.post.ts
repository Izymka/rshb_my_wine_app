// Награда за действие в сервисе, о котором сервер узнаёт от клиента: распознанный скан и вопрос
// сомелье идут через прокси в ML-сервис. Сорта для скана сервер берёт из памяти прокси, а не
// из запроса; повтор вина и суточный лимит проверяет grantServiceReward.
import { assertSameOrigin, rateLimit } from "../../utils/auth";
import { dayKey } from "~/utils/game.mjs";
import { grantServiceReward, requireGamer } from "../../utils/game";
import { recallScan } from "../../utils/scan-memory";

export default defineEventHandler(async (event) => {
  assertSameOrigin(event);
  const user = await requireGamer(event);
  rateLimit(`game-reward:${user.id}`, 30, 60000);
  const body = await readBody(event).catch(() => null);
  if (body?.kind === "sommelier") {
    return await grantServiceReward(user.id, "sommelier", dayKey(Date.now()), {}, true);
  }
  const slug = typeof body?.slug === "string" ? body.slug : "";
  if (body?.kind !== "scan" || !slug || slug.length > 500) throw createError({ statusCode: 400, statusMessage: "Неверный запрос" });
  // Сорта — из ответа сканера, который прошёл через наш прокси в этой сессии, а не из запроса:
  // вино, которого игрок не сканировал, не награждается (каталог для этого не годится — он
  // отдаёт любое вино без скана).
  const card = recallScan(event, slug);
  if (!card) throw createError({ statusCode: 404, statusMessage: "Сначала отсканируйте вино" });
  const result = await grantServiceReward(user.id, "scan", slug, { grapes: card.grapes || "", name: card.name || "" }, true);
  return { ...result, name: card.name || "" };
});
