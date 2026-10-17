import { assertSameOrigin, destroySession } from "../../utils/auth";

export default defineEventHandler(async (event) => {
  assertSameOrigin(event);
  await destroySession(event);
  return { ok: true };
});
