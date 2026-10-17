// Вход по телефону и сессии. СМС-провайдер в этой версии не подключён: код возвращается в ответе
// и показывается на экране (NUXT_AUTH_DEMO_CODES=1). Подключение провайдера меняет только
// sendCode — остальная цепочка (хэш кода, попытки, сессия в httpOnly-cookie) уже боевая.
import { createHash, randomBytes, randomInt, timingSafeEqual } from "node:crypto";
import type { H3Event } from "h3";
import { useDb } from "./db";

export const SESSION_COOKIE = "sv_session";
const SESSION_DAYS = 90;
export const CODE_TTL_SECONDS = 300;
export const RESEND_SECONDS = 45;
export const MAX_CODE_ATTEMPTS = 5;

export interface User {
  id: string;
  phone: string;
  name: string;
  email: string | null;
  avatarUrl: string | null;
  adult: boolean;
  defaultShare: boolean;
  createdAt: string;
}

interface UserRow {
  id: string;
  phone: string;
  name: string;
  email: string | null;
  avatar_version: number;
  adult_confirmed_at: Date | null;
  default_share: boolean;
  created_at: Date;
}

export const USER_COLUMNS = "id, phone, name, email, avatar_version, adult_confirmed_at, default_share, created_at";

export function sha256(value: string) {
  return createHash("sha256").update(value).digest("hex");
}

// Российский номер из любого ввода: «+7 (916) 123-45-67», «89161234567», «9161234567».
export function normalizePhone(raw: unknown): string | null {
  if (typeof raw !== "string") return null;
  let digits = raw.replace(/\D/g, "");
  if (digits.length === 11 && (digits[0] === "7" || digits[0] === "8")) digits = digits.slice(1);
  return /^9\d{9}$/.test(digits) ? digits : null;
}

export function avatarUrl(id: string, version: number) {
  return version > 0 ? `/api/avatars/${id}?v=${version}` : null;
}

export function toUser(row: UserRow): User {
  return {
    id: String(row.id),
    phone: row.phone,
    name: row.name,
    email: row.email,
    avatarUrl: avatarUrl(String(row.id), row.avatar_version),
    adult: row.adult_confirmed_at !== null,
    defaultShare: row.default_share,
    createdAt: row.created_at.toISOString(),
  };
}

export function newCode() {
  return String(randomInt(0, 10000)).padStart(4, "0");
}

export function sameCode(code: string, hash: string) {
  const a = Buffer.from(sha256(code));
  const b = Buffer.from(hash);
  return a.length === b.length && timingSafeEqual(a, b);
}

export async function createSession(event: H3Event, userId: string) {
  const sql = await useDb();
  const token = randomBytes(32).toString("base64url");
  const expires = new Date(Date.now() + SESSION_DAYS * 86400000);
  await sql`insert into sessions (token_hash, user_id, expires_at) values (${sha256(token)}, ${userId}, ${expires})`;
  setCookie(event, SESSION_COOKIE, token, {
    httpOnly: true,
    sameSite: "lax",
    secure: getRequestURL(event).protocol === "https:",
    path: "/",
    expires,
  });
}

export async function destroySession(event: H3Event) {
  const token = getCookie(event, SESSION_COOKIE);
  if (token) {
    const sql = await useDb();
    await sql`delete from sessions where token_hash = ${sha256(token)}`;
  }
  deleteCookie(event, SESSION_COOKIE, { path: "/" });
}

export async function currentUser(event: H3Event): Promise<User | null> {
  const token = getCookie(event, SESSION_COOKIE);
  if (!token) return null;
  const sql = await useDb();
  const [row] = await sql<UserRow[]>`
    select ${sql.unsafe(USER_COLUMNS.split(", ").map((c) => `u.${c}`).join(", "))}
    from sessions s join users u on u.id = s.user_id
    where s.token_hash = ${sha256(token)} and s.expires_at > now()`;
  return row ? toUser(row) : null;
}

export async function requireUser(event: H3Event): Promise<User> {
  const user = await currentUser(event);
  if (!user) throw createError({ statusCode: 401, statusMessage: "Sign in required" });
  return user;
}

// Изменяющие запросы принимаем только со своего origin: cookie SameSite=Lax уже закрывает
// большинство CSRF, а проверка Origin — остаток (формы верхнего уровня).
export function assertSameOrigin(event: H3Event) {
  const origin = getRequestHeader(event, "origin");
  if (origin && origin !== getRequestURL(event).origin) throw createError({ statusCode: 403 });
}

const buckets = new Map<string, { count: number; until: number }>();
export function rateLimit(key: string, limit: number, windowMs: number) {
  const now = Date.now();
  if (buckets.size > 10000) for (const [k, v] of buckets) if (v.until < now) buckets.delete(k);
  const bucket = buckets.get(key);
  if (!bucket || bucket.until < now) {
    buckets.set(key, { count: 1, until: now + windowMs });
    return;
  }
  if (++bucket.count > limit) throw createError({ statusCode: 429, statusMessage: "Too many requests" });
}
