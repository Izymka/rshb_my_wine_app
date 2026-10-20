// Игровые уведомления на этом устройстве: { subscription } включает, { disable, endpoint } —
// выключает. Подписка та же, что у напоминаний полки (общий VAPID-ключ и service worker).
import { validateSubscription } from "../../utils/shelf-reminders.mjs";
import { shelfPushEnabled } from "../../utils/shelf-push";
import { assertSameOrigin, rateLimit } from "../../utils/auth";
import { requireGamer } from "../../utils/game";
import { addGamePushDevice, removeGamePushDevice } from "../../utils/game-push";

export default defineEventHandler(async (event) => {
  setResponseHeader(event, "Cache-Control", "no-store");
  assertSameOrigin(event);
  const user = await requireGamer(event);
  rateLimit(`game-push:${user.id}`, 30, 60000);
  const body = await readBody(event).catch(() => null);
  if (body?.disable === true) {
    if (typeof body.endpoint === "string" && body.endpoint.length <= 4096) await removeGamePushDevice(user.id, body.endpoint);
    return { enabled: false };
  }
  if (!shelfPushEnabled()) throw createError({ statusCode: 503, statusMessage: "Push is not configured" });
  let device;
  try { device = validateSubscription(body?.subscription); }
  catch { throw createError({ statusCode: 400, statusMessage: "Invalid subscription" }); }
  await addGamePushDevice(user.id, device);
  return { enabled: true };
});
