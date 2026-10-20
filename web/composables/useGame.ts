// Клиент «Моей винодельни». Состояние приходит с сервера и только отображается: каждое действие —
// POST /api/game/action с requestId, ответ заменяет состояние целиком. Часы выравниваются по
// серверу, чтобы таймеры на экране совпадали с тем, что сервер считает готовым.
import { coinArt, grozdArt, repArt, xpArt } from "~/utils/game-art.mjs";
import { itemName, levelOf } from "~/utils/game.mjs";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type GameState = any;
export type GameStatus = "idle" | "loading" | "ready" | "guest" | "adult" | "unavailable" | "error";
export interface GameReward { kind: string; ref: string; reward: Record<string, any>; createdAt: string }
export interface Toast { id: number; text: string; x?: number; y?: number; tone?: "dark" | "error" }

const ENABLED_KEY = "wine-game:enabled";
export const ICON = { coins: coinArt(), grozdi: grozdArt(), xp: xpArt(), rep: repArt() };

export function fmtLeft(msLeft: number) {
  const sec = Math.max(0, Math.ceil(msLeft / 1000));
  const d = Math.floor(sec / 86400), h = Math.floor((sec % 86400) / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  if (d) return `${d} д ${h} ч`;
  return (h ? `${h}:${String(m).padStart(2, "0")}` : String(m)) + ":" + String(s).padStart(2, "0");
}
// Длительность в игровых секундах с учётом ускорения: «2 ч», «30 мин», «3 дня».
export function fmtDur(sec: number, speed = 1) {
  const s = sec / speed;
  if (s >= 86400) { const d = Math.round(s / 86400); return `${d} ${d === 1 ? "день" : d < 5 ? "дня" : "дней"}`; }
  if (s >= 3600) return `${Math.round(s / 360) / 10} ч`.replace(".", ",");
  if (s >= 60) return `${Math.round(s / 60)} мин`;
  return `${Math.round(s)} с`;
}
export function fmtNum(n: number) { return Math.round(n).toLocaleString("ru-RU"); }

export function useGame() {
  const state = useState<GameState | null>("game.state", () => null);
  const status = useState<GameStatus>("game.status", () => "idle");
  const speed = useState("game.speed", () => 1);
  const offset = useState("game.offset", () => 0);
  const now = useState("game.now", () => 0);
  const rewards = useState<GameReward[]>("game.rewards", () => []);
  const toasts = useState<Toast[]>("game.toasts", () => []);
  const busy = useState("game.busy", () => false);
  const levelUp = useState<number | null>("game.levelUp", () => null);
  const enabledLocal = useState("game.enabledLocal", () => true);
  const { online } = useConnectivity();
  const { goal } = useMetrika();

  function accept(r: { state: GameState; now: number; speed: number }) {
    offset.value = r.now - Date.now();
    speed.value = r.speed;
    now.value = r.now;
    state.value = r.state;
    rememberEnabled(r.state.enabled !== false);
  }
  function rememberEnabled(on: boolean) {
    enabledLocal.value = on;
    try { localStorage.setItem(ENABLED_KEY, on ? "1" : "0"); } catch {}
  }
  function readEnabled() {
    try { enabledLocal.value = localStorage.getItem(ENABLED_KEY) !== "0"; } catch {}
  }

  async function load() {
    if (!state.value) status.value = "loading";
    try {
      const r = await $fetch<{ state: GameState; now: number; speed: number; rewards: GameReward[] }>("/api/game/state", { timeout: 15000 });
      accept(r);
      rewards.value = r.rewards;
      status.value = "ready";
    } catch (e) {
      const code = (e as { statusCode?: number })?.statusCode;
      if (code === 401) status.value = "guest";
      else if (code === 403) status.value = "adult";
      else if (code === 503) status.value = "unavailable";
      else if (!state.value) status.value = "error";
    }
  }

  function toast(text: string, opts: Partial<Toast> = {}) {
    const id = Math.random();
    toasts.value = [...toasts.value, { id, text, ...opts }];
    setTimeout(() => { toasts.value = toasts.value.filter((t) => t.id !== id); }, opts.tone === "error" ? 2600 : 1500);
  }

  // Действие игрока. Возвращает то, что получено (для анимаций), или null при отказе.
  async function act(action: Record<string, unknown>, at?: { x: number; y: number }) {
    if (!online.value) { toast("Нет сети — только просмотр. Таймеры идут на сервере", { tone: "error" }); return null; }
    if (busy.value) return null;
    busy.value = true;
    const requestId = typeof crypto !== "undefined" && "randomUUID" in crypto ? crypto.randomUUID() : String(Math.random()).slice(2);
    const level = state.value ? levelOf(state.value.xp) : 1;
    try {
      const r = await $fetch<{ state: GameState; now: number; speed: number; result: any }>("/api/game/action", {
        method: "POST", body: { action, requestId }, timeout: 15000,
      });
      accept(r);
      const res = r.result;
      if (res) {
        const text = describe(res);
        if (text) toast(text, at);
        if (res.levelUp && res.levelUp > level) levelUp.value = res.levelUp;
      }
      goal("game_action", { type: String(action.type) });
      return res || {};
    } catch (e) {
      toast(apiError(e, "Не получилось. Попробуйте ещё раз."), { tone: "error" });
      return null;
    } finally {
      busy.value = false;
    }
  }

  async function ackRewards() {
    rewards.value = [];
    await $fetch("/api/game/ack", { method: "POST" }).catch(() => {});
  }

  return { state, status, speed, now, offset, rewards, toasts, busy, levelUp, enabledLocal, load, act, toast, ackRewards, accept, readEnabled, rememberEnabled };
}

// Часы игры: один интервал на приложение, время — по серверу.
export function useGameClock() {
  const { now, offset } = useGame();
  let timer: ReturnType<typeof setInterval> | null = null;
  onMounted(() => {
    const tick = () => { now.value = Date.now() + offset.value; };
    tick();
    timer = setInterval(tick, 1000);
  });
  onBeforeUnmount(() => { if (timer) clearInterval(timer); });
  return now;
}

// Короткая подпись к полученному: «+3 Шардоне · +9 XP».
export function describe(res: { gained?: Record<string, number>; items?: Record<string, number> }) {
  const parts: string[] = [];
  for (const [item, n] of Object.entries(res.items || {})) parts.push(`+${n} ${itemName(item).replace(/^Виноград /, "")}`);
  const g = res.gained || {};
  if (g.coins) parts.push(`+${g.coins} монет`);
  if (g.grozdi) parts.push(`+${g.grozdi} Гроздей`);
  if (g.xp) parts.push(`+${g.xp} XP`);
  if (g.rep) parts.push(`+${g.rep} репутации`);
  return parts.join(" · ");
}
