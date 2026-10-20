// Облачная полка: устройство присылает свои вина и удаления, сервер оставляет по каждому вину
// более позднюю правку и возвращает итог. Полка в браузере остаётся рабочей копией без сети.
import { parseShelf } from "~/utils/shelf.mjs";
import { assertSameOrigin, rateLimit, requireUser } from "../../utils/auth";
import { useDb } from "../../utils/db";
import { grantServiceReward } from "../../utils/game";

interface Row {
  slug: string; card: object; analogues: object[]; status: string; reaction: string | null;
  dishes: string[]; note: string; remind_at: string | null; added_at: string; updated_at: string;
}

export default defineEventHandler(async (event) => {
  setResponseHeader(event, "Cache-Control", "no-store");
  assertSameOrigin(event);
  const user = await requireUser(event);
  rateLimit(`sync:${user.id}`, 120, 60000);
  if (Number(getRequestHeader(event, "content-length") || 0) > 4 * 1024 * 1024) throw createError({ statusCode: 413 });
  const body = await readBody(event).catch(() => null);
  let entries;
  try {
    entries = parseShelf(JSON.stringify({ version: 1, entries: body?.entries ?? [] }));
  } catch {
    throw createError({ statusCode: 400, statusMessage: "Invalid shelf" });
  }
  const deleted = Array.isArray(body?.deleted) ? body.deleted.slice(0, 1000)
    .filter((d: { slug?: unknown; at?: unknown }) => typeof d?.slug === "string" && d.slug.length <= 500 && Number.isFinite(d.at)) : [];

  const sql = await useDb();
  await sql.begin(async (tx) => {
    if (entries.length) {
      const rows = entries.map((e) => ({
        user_id: user.id, slug: e.slug, card: tx.json(e.card), analogues: tx.json(e.analogues),
        status: e.status, reaction: e.reaction, dishes: e.dishes, note: e.note,
        remind_at: e.remindAt, added_at: e.addedAt, updated_at: e.updatedAt, deleted: false,
      }));
      await tx`
        insert into shelf_entries ${tx(rows)}
        on conflict (user_id, slug) do update set card = excluded.card, analogues = excluded.analogues,
          status = excluded.status, reaction = excluded.reaction, dishes = excluded.dishes, note = excluded.note,
          remind_at = excluded.remind_at, updated_at = excluded.updated_at, deleted = false,
          added_at = case when shelf_entries.deleted then excluded.added_at else shelf_entries.added_at end
        where shelf_entries.updated_at < excluded.updated_at`;
    }
    for (const d of deleted) {
      await tx`update shelf_entries set deleted = true, updated_at = ${d.at}
        where user_id = ${user.id} and slug = ${d.slug} and updated_at <= ${d.at}`;
    }
  });
  // Награда в винодельне за вина, сохранённые за последние сутки: старая полка при первой
  // синхронизации не превращается в мешок монет. Повторы и суточный лимит проверяет игра.
  const recent = user.adult ? entries.filter((e) => e.addedAt > Date.now() - 86400000).slice(0, 20) : [];
  if (recent.length) {
    const done = new Set((await sql<{ ref: string }[]>`select ref from game_service_rewards
      where user_id = ${user.id} and kind = 'shelf' and ref = any(${recent.map((e) => e.slug)})`).map((r) => r.ref));
    for (const e of recent.filter((e) => !done.has(e.slug))) {
      const r = await grantServiceReward(user.id, "shelf", e.slug, { name: e.card.name })
        .catch((error) => { console.error("game reward failed:", (error as Error).message); return null; });
      if (r?.status === "limit") break;
    }
  }
  const rows = await sql<(Row & { deleted: boolean })[]>`
    select slug, card, analogues, status, reaction, dishes, note, remind_at, added_at, updated_at, deleted
    from shelf_entries where user_id = ${user.id} order by added_at desc`;
  // Удаления с других устройств возвращаются отдельно: без них устройство с устаревшей копией
  // продолжало бы показывать вино, которого в аккаунте уже нет.
  return {
    deleted: rows.filter((r) => r.deleted).map((r) => ({ slug: r.slug, at: Number(r.updated_at) })),
    entries: rows.filter((r) => !r.deleted).slice(0, 500).map((r) => ({
      slug: r.slug, card: r.card, analogues: r.analogues, status: r.status, reaction: r.reaction,
      dishes: r.dishes, note: r.note, remindAt: r.remind_at === null ? null : Number(r.remind_at),
      addedAt: Number(r.added_at), updatedAt: Number(r.updated_at),
    })),
  };
});
