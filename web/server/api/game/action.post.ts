// Действие игрока: посадить, собрать, запустить давильню, выполнить заказ, купить… Правила и
// проверки — в utils/game.mjs; requestId защищает от двойного списания при повторе запроса.
import { assertSameOrigin, rateLimit } from "../../utils/auth";
import { clientState, requireGamer, runAction } from "../../utils/game";

export default defineEventHandler(async (event) => {
  setResponseHeader(event, "Cache-Control", "no-store");
  assertSameOrigin(event);
  const user = await requireGamer(event);
  rateLimit(`game:${user.id}`, 240, 60000);
  const body = await readBody(event).catch(() => null);
  const action = body?.action;
  const requestId = typeof body?.requestId === "string" && body.requestId.length <= 64 ? body.requestId : undefined;
  if (!action || typeof action !== "object" || typeof action.type !== "string") {
    throw createError({ statusCode: 400, statusMessage: "Неверный запрос" });
  }
  const { state, result, duplicate, now, speed } = await runAction(user.id, action, requestId);
  return { state: clientState(state), result, duplicate, now, speed };
});
