// Игровые пуши: подписки хранятся по игроку (а не по устройству, как напоминания полки), потому
// что поводы берутся из его серверного состояния game_profiles.state. Рассылку запускает общий
// тик пушей (plugins/shelf-reminders.ts), правила и тексты — utils/game-push.mjs.
import { planGamePush } from "~/utils/game-push.mjs";
import { dbEnabled, useDb } from "./db";
import { gameSpeed } from "./game";

interface Device { endpoint: string; keys: { auth: string; p256dh: string } }
const MAX_DEVICES = 5;

export async function addGamePushDevice(userId: string, device: Device) {
  const sql = await useDb();
  await sql.begin(async (tx) => {
    const [row] = await tx<{ devices: Device[] }[]>`select devices from game_push where user_id = ${userId} for update`;
    const devices = [device, ...(row?.devices || []).filter((d) => d.endpoint !== device.endpoint)].slice(0, MAX_DEVICES);
    await tx`insert into game_push (user_id, devices) values (${userId}, ${tx.json(devices as never)})
      on conflict (user_id) do update set devices = excluded.devices, updated_at = now()`;
  });
}

export async function removeGamePushDevice(userId: string, endpoint: string) {
  const sql = await useDb();
  await sql`update game_push set devices = coalesce((select jsonb_agg(d) from jsonb_array_elements(devices) d
    where d->>'endpoint' <> ${endpoint}), '[]'), updated_at = now() where user_id = ${userId}`;
}

type Send = (subscription: object, payload: { tag: string }) => Promise<unknown>;

// Один проход по игрокам с подписками. Журнал обновляется, только если пуш доставлен хотя бы на
// одно устройство: при сбое сети повод не теряется и уйдёт на следующем тике.
export async function dispatchGamePushes(send: Send, now = Date.now()) {
  if (!dbEnabled()) return;
  const sql = await useDb();
  const speed = gameSpeed();
  const rows = await sql<{ userId: string; devices: Device[]; log: object; state: object; seen: Date }[]>`
    select g.user_id as "userId", g.devices, g.log, p.state, p.updated_at as seen
    from game_push g join game_profiles p using (user_id)
    where jsonb_array_length(g.devices) > 0`;
  for (const row of rows) {
    const plan = planGamePush(row.state, row.log, { now, speed, lastSeen: row.seen.getTime() });
    if (plan.payload) {
      let delivered = false;
      for (const device of row.devices) {
        try { await send(device, plan.payload); delivered = true; }
        catch (error) {
          const code = (error as { statusCode?: number }).statusCode;
          if (code === 404 || code === 410) await removeGamePushDevice(row.userId, device.endpoint);
          else console.warn("Game push delivery failed", Number(code) || "network");
        }
      }
      if (!delivered) continue;
    }
    if (JSON.stringify(plan.log) !== JSON.stringify(row.log)) {
      await sql`update game_push set log = ${sql.json(plan.log as never)} where user_id = ${row.userId}`;
    }
  }
}
