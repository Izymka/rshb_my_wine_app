// Серверная сторона «Моей винодельни». Состояние игрока хранится в game_profiles.state и
// меняется только здесь: в транзакции с блокировкой строки (SELECT … FOR UPDATE), так что два
// одновременных запроса не потратят одни монеты дважды. Правила — в utils/game.mjs.
import type { H3Event } from "h3";
import type postgres from "postgres";
import { randomBytes } from "node:crypto";
import { GameError, SERVICE, applyAction, dayKey, newGame, refresh, serviceReward } from "~/utils/game.mjs";
import { requireUser } from "./auth";
import { useDb } from "./db";

type Sql = postgres.Sql | postgres.TransactionSql;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type GameState = any;
export interface Want { slug: string; name: string; winery: string; grapes: string }

export function gameSpeed() {
  const speed = Number(useRuntimeConfig().gameSpeed);
  return Number.isFinite(speed) && speed >= 1 ? Math.min(speed, 3600) : 1;
}

// Игра — для взрослых с завершённым профилем: возраст подтверждается при входе.
export async function requireGamer(event: H3Event) {
  const user = await requireUser(event);
  if (!user.adult) throw createError({ statusCode: 403, statusMessage: "Игра доступна только после подтверждения возраста 18+" });
  return user;
}

export function gameError(error: unknown): never {
  if (error instanceof GameError) throw createError({ statusCode: 409, statusMessage: error.message });
  throw error;
}

// «Хочу попробовать» с облачной полки — из них доска собирает заказы гостей.
async function loadWants(sql: Sql, userId: string): Promise<Want[]> {
  return await sql<Want[]>`
    select slug, coalesce(card->>'name', '') as name, coalesce(card->>'winery', '') as winery, coalesce(card->>'grapes', '') as grapes
    from shelf_entries where user_id = ${userId} and status = 'want' and not deleted
    order by updated_at desc limit 30`;
}

// Транзакция над состоянием игрока: создаёт игру при первом обращении, блокирует строку,
// выполняет fn и сохраняет результат. requestId делает повтор того же запроса безопасным:
// второй раз действие не выполняется, клиент просто получает текущее состояние.
export async function withGame<T>(
  userId: string,
  fn: (state: GameState, ctx: { now: number; speed: number; wants: Want[]; tx: postgres.TransactionSql }) => T | Promise<T>,
  opts: { requestId?: string; action?: string } = {},
): Promise<{ state: GameState; result: T | null; duplicate: boolean; now: number; speed: number }> {
  const sql = await useDb();
  const speed = gameSpeed();
  return await sql.begin(async (tx) => {
    const now = Date.now();
    let [row] = await tx<{ state: GameState }[]>`select state from game_profiles where user_id = ${userId} for update`;
    if (!row) {
      const state = newGame(`${userId}:${randomBytes(6).toString("hex")}`, now, speed);
      await tx`insert into game_profiles (user_id, state) values (${userId}, ${tx.json(state)}) on conflict do nothing`;
      [row] = await tx<{ state: GameState }[]>`select state from game_profiles where user_id = ${userId} for update`;
    }
    const state = row.state;
    const wants = await loadWants(tx, userId);
    if (opts.requestId) {
      const [fresh] = await tx`
        insert into game_ledger (user_id, request_id, action) values (${userId}, ${opts.requestId}, ${opts.action || "action"})
        on conflict (user_id, request_id) where request_id is not null do nothing returning id`;
      if (!fresh) {
        refresh(state, now, speed, { wants });
        return { state, result: null, duplicate: true, now, speed };
      }
    }
    const before = { coins: state.coins, grozdi: state.grozdi, xp: state.xp, rep: state.rep };
    const result = await fn(state, { now, speed, wants, tx });
    const delta = Object.fromEntries(Object.entries(before)
      .map(([k, v]) => [k, (state[k] as number) - v]).filter(([, v]) => v !== 0));
    if (opts.requestId && Object.keys(delta).length) {
      await tx`update game_ledger set delta = ${tx.json(delta)} where user_id = ${userId} and request_id = ${opts.requestId}`;
    } else if (!opts.requestId && Object.keys(delta).length) {
      await tx`insert into game_ledger (user_id, action, delta) values (${userId}, ${opts.action || "system"}, ${tx.json(delta)})`;
    }
    // updated_at — по тем же часам, что и таймеры игры: по нему игровые пуши (utils/game-push.mjs)
    // понимают, что игрок уже видел на поле; now() базы раньше Date.now() и пропустил бы это.
    await tx`update game_profiles set state = ${tx.json(state)}, version = version + 1, updated_at = ${new Date(now)} where user_id = ${userId}`;
    return { state, result, duplicate: false, now, speed };
  });
}

export async function readGame(userId: string) {
  return await withGame(userId, (state, { now, speed, wants }) => { refresh(state, now, speed, { wants }); return null; }, { action: "refresh" });
}

export function runAction(userId: string, action: Record<string, unknown>, requestId?: string) {
  return withGame(userId, (state, { now, speed, wants }) => {
    try { return applyAction(state, action, now, speed, { wants }); } catch (e) { gameError(e); }
  }, { requestId, action: String(action?.type || "action") });
}

// Состояние для клиента: без зерна случайности (по нему можно предсказать скворцов).
export function clientState(state: GameState) {
  const { seed: _seed, n: _n, orders, ...rest } = state;
  return { ...rest, orders: { slots: orders.slots } };
}

export async function unseenRewards(userId: string) {
  const sql = await useDb();
  return await sql<{ kind: string; ref: string; reward: object; createdAt: Date }[]>`
    select kind, ref, reward, created_at as "createdAt" from game_service_rewards
    where user_id = ${userId} and not seen and kind <> 'wordle' order by created_at desc limit 10`;
}

export type RewardStatus = "granted" | "repeat" | "limit" | "disabled";

// Награда за действие в сервисе (скан, отзыв, полка, сомелье). Одно и то же вино награждается
// один раз навсегда, а за сутки — не больше лимита SERVICE[kind].limit. Ошибки игры не должны
// ломать основное действие, поэтому вызывающие хэндлеры оборачивают вызов в try/catch.
// seen — награду уже показал сам экран (шторка после скана), повторять её на поле не нужно.
export async function grantServiceReward(userId: string, kind: keyof typeof SERVICE, ref: string, data: Record<string, unknown> = {}, seen = false) {
  const limit = SERVICE[kind].limit;
  let status: RewardStatus = "granted";
  let used = 0;
  const { result } = await withGame(userId, async (state, { now, speed, tx }) => {
    if (state.enabled === false) { status = "disabled"; return null; }
    const day = dayKey(now);
    const [repeat] = await tx`select 1 from game_service_rewards where user_id = ${userId} and kind = ${kind} and ref = ${ref}`;
    if (repeat) { status = "repeat"; return null; }
    const [{ count }] = await tx<{ count: number }[]>`
      select count(*)::int as count from game_service_rewards where user_id = ${userId} and kind = ${kind} and day = ${day}`;
    used = count;
    if (count >= limit) { status = "limit"; return null; }
    const reward = serviceReward(state, kind, data, now, speed);
    await tx`insert into game_service_rewards (user_id, kind, ref, day, reward, seen)
      values (${userId}, ${kind}, ${ref}, ${day}, ${tx.json({ ...reward, name: data.name || null })}, ${seen})`;
    used = count + 1;
    return reward;
  }, { action: `reward:${kind}` });
  return { status, reward: result, used, limit };
}

// Доля угадавших сегодня — для экрана результата игры «Вино дня».
export async function wordleStats(day: string) {
  const sql = await useDb();
  const [s] = await sql<{ players: number; solved: number }[]>`
    select count(*)::int as players, count(*) filter (where solved)::int as solved
    from wordle_attempts where day = ${day} and finished`;
  return { players: s.players, solvedPct: s.players ? Math.round((s.solved / s.players) * 100) : null };
}
