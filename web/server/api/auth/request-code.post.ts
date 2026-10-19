// Шаг 1 входа: номер → одноразовый код. Пока провайдера нет, код уходит в ответ (demoCode)
// и интерфейс показывает его вместо СМС.
import { CODE_TTL_SECONDS, RESEND_SECONDS, assertSameOrigin, newCode, normalizePhone, rateLimit, sha256 } from "../../utils/auth";
import { useDb } from "../../utils/db";

export default defineEventHandler(async (event) => {
  setResponseHeader(event, "Cache-Control", "no-store");
  assertSameOrigin(event);
  rateLimit(`code-ip:${getRequestIP(event) || "unknown"}`, 20, 3600000);
  const body = await readBody(event).catch(() => null);
  const phone = normalizePhone(body?.phone);
  if (!phone) throw createError({ statusCode: 400, statusMessage: "Введите российский мобильный номер" });

  const sql = await useDb();
  const [last] = await sql<{ wait: number }[]>`
    select ceil(extract(epoch from sent_at + ${RESEND_SECONDS} * interval '1 second' - now()))::int as wait
    from auth_codes where phone = ${phone} and sent_at > now() - ${RESEND_SECONDS} * interval '1 second'`;
  if (last) {
    throw createError({ statusCode: 429, statusMessage: "Код уже отправлен", data: { retryIn: last.wait } });
  }
  const code = newCode();
  await sql`
    insert into auth_codes (phone, code_hash, attempts, sent_at, expires_at)
    values (${phone}, ${sha256(code)}, 0, now(), now() + ${CODE_TTL_SECONDS} * interval '1 second')
    on conflict (phone) do update set code_hash = excluded.code_hash, attempts = 0,
      sent_at = excluded.sent_at, expires_at = excluded.expires_at`;
  const demo = useRuntimeConfig(event).authDemoCodes !== "0";
  return { phone, retryIn: RESEND_SECONDS, expiresIn: CODE_TTL_SECONDS, demoCode: demo ? code : null };
});
