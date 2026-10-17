// Шаг 2 входа: код → сессия. Новый номер сразу становится аккаунтом; имя и подтверждение
// 18+ интерфейс спрашивает следующим шагом, если их ещё нет.
import { MAX_CODE_ATTEMPTS, USER_COLUMNS, assertSameOrigin, createSession, normalizePhone, rateLimit, sameCode, toUser } from "../../utils/auth";
import { useDb } from "../../utils/db";

export default defineEventHandler(async (event) => {
  setResponseHeader(event, "Cache-Control", "no-store");
  assertSameOrigin(event);
  rateLimit(`verify-ip:${getRequestIP(event) || "unknown"}`, 60, 3600000);
  const body = await readBody(event).catch(() => null);
  const phone = normalizePhone(body?.phone);
  const code = typeof body?.code === "string" ? body.code.replace(/\D/g, "") : "";
  if (!phone || code.length !== 4) throw createError({ statusCode: 400, statusMessage: "Введите код из 4 цифр" });

  const sql = await useDb();
  const [row] = await sql<{ code_hash: string; attempts: number; expired: boolean }[]>`
    select code_hash, attempts, expires_at < now() as expired from auth_codes where phone = ${phone}`;
  if (!row || row.expired || row.attempts >= MAX_CODE_ATTEMPTS) {
    throw createError({ statusCode: 410, statusMessage: "Код устарел — запросите новый" });
  }
  if (!sameCode(code, row.code_hash)) {
    await sql`update auth_codes set attempts = attempts + 1 where phone = ${phone}`;
    const left = MAX_CODE_ATTEMPTS - row.attempts - 1;
    throw createError({ statusCode: 401, statusMessage: left > 0 ? "Неверный код" : "Код устарел — запросите новый", data: { left } });
  }
  await sql`delete from auth_codes where phone = ${phone}`;
  const [existing] = await sql`select id from users where phone = ${phone}`;
  const [user] = await sql`
    insert into users (phone) values (${phone})
    on conflict (phone) do update set updated_at = users.updated_at
    returning ${sql.unsafe(USER_COLUMNS)}`;
  await createSession(event, String(user.id));
  return { user: toUser(user as never), isNew: !existing };
});
