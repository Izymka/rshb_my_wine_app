import { currentUser } from "../../utils/auth";
import { dbEnabled } from "../../utils/db";

export default defineEventHandler(async (event) => {
  setResponseHeader(event, "Cache-Control", "no-store");
  if (!dbEnabled()) return { enabled: false, user: null };
  return { enabled: true, user: await currentUser(event) };
});
