// Последние распознанные вина каждой сессии: прокси /api/scan кладёт сюда карточки из ответа
// сервиса, а награда за скан в «Моей винодельне» берёт сорта отсюда, а не из запроса клиента.
// Память процесса и 30 минут жизни: награда запрашивается сразу после скана.
import type { H3Event } from "h3";
import { SESSION_COOKIE, sha256 } from "./auth";

interface Seen { name: string; grapes: string; at: number }
const TTL = 30 * 60000;
const memory = new Map<string, Map<string, Seen>>();

function key(event: H3Event) {
  const token = getCookie(event, SESSION_COOKIE);
  return token ? sha256(token) : null;
}

export function rememberScan(event: H3Event, cards: { slug: string; name?: string; grapes?: string }[]) {
  const k = key(event);
  if (!k || !cards.length) return;
  const now = Date.now();
  if (memory.size > 5000) for (const [mk, m] of memory) if ([...m.values()].every((s) => now - s.at > TTL)) memory.delete(mk);
  const mine = memory.get(k) ?? new Map<string, Seen>();
  for (const c of cards.slice(0, 8)) mine.set(c.slug, { name: c.name || "", grapes: c.grapes || "", at: now });
  memory.set(k, mine);
}

export function recallScan(event: H3Event, slug: string): Seen | null {
  const seen = memory.get(key(event) || "")?.get(slug);
  return seen && Date.now() - seen.at < TTL ? seen : null;
}
