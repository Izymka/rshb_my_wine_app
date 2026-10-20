<script setup lang="ts">
// «Моя винодельня» — главный экран (1a): поле с участками и зданиями, HUD валют, быстрые
// действия, подсказки Виноградинки, онбординг (1j), ночь, тизер без входа (1l) и загрузка (1m).
// Координаты объектов — из макета 375×812; сцена масштабируется под свободное место экрана.
import { BALANCE, TUTORIAL_STEPS, VARIETIES, dayKey, levelOf, levelProgress, plotStage, questProgress } from "~/utils/game.mjs";
import { bottlingArt, cellarArt, fieldBg, lanternArt, mascotArt, plotArt, pressArt, tankArt, treeArt, benchArt, flowerBarrelArt, scarecrowArt } from "~/utils/game-art.mjs";

const game = useGame();
const { state, status, speed, act, toasts, levelUp, rewards } = game;
const now = useGameClock();
const auth = useAuth();
const { goal } = useMetrika();
const router = useRouter();
useHead({ title: "Моя винодельня — Своё Вино" });

onMounted(() => {
  void game.load();
  // Вход из игрового пуша: /winery?push=game_harvest — тип пуша для воронки «пуш → открытие».
  const push = useRoute().query.push;
  goal("game_open", typeof push === "string" ? { from: "push" } : {});
  if (typeof push === "string" && /^game_\w+$/.test(push)) { goal("push_opened", { type: push }); void router.replace({ query: {} }); }
});
// Вернулись на вкладку — подтягиваем таймеры и награды, накопленные в сервисе.
function onVisible() { if (document.visibilityState === "visible") void game.load(); }
onMounted(() => document.addEventListener("visibilitychange", onVisible));
onBeforeUnmount(() => document.removeEventListener("visibilitychange", onVisible));

// --- Масштаб сцены: между HUD и кнопками действий.
const Y0 = 140, SCENE_H = 500;
const vw = ref(375), vh = ref(812);
const hudEl = ref<HTMLElement>();
const actionsEl = ref<HTMLElement>();
const ordersBtn = ref<{ $el: HTMLElement } | null>(null);
const hudBottom = ref(130), actionsTop = ref(640);
function measure() {
  vw.value = window.innerWidth;
  vh.value = window.innerHeight;
  hudBottom.value = hudEl.value ? hudEl.value.getBoundingClientRect().bottom + 30 : 150;
  actionsTop.value = actionsEl.value ? actionsEl.value.getBoundingClientRect().top - 8 : vh.value - 180;
}
onMounted(() => { measure(); window.addEventListener("resize", measure); });
onBeforeUnmount(() => window.removeEventListener("resize", measure));
watch(status, () => nextTick(measure));
const scale = computed(() => {
  const avail = actionsTop.value - hudBottom.value;
  return Math.max(0.62, Math.min(1.35, vw.value / 375, avail / SCENE_H));
});
const sceneTop = computed(() => Math.max(hudBottom.value - 20, hudBottom.value + (actionsTop.value - hudBottom.value - SCENE_H * scale.value) / 2));
const sceneStyle = computed(() => ({ top: `${sceneTop.value}px`, transform: `translateX(-50%) scale(${scale.value})` }));
function toScreen(x: number, y: number) {
  return { x: vw.value / 2 + (x - 187.5) * scale.value, y: sceneTop.value + (y - Y0) * scale.value };
}

// --- Объекты поля
const PLOT_POS = [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [3, 0], [3, 1]].map(([c, r]) => ({ x: 68 + (c - r) * 54, y: 300 + (c + r) * 27 }));
const DECOR_POS: Record<string, number[]> = { lantern: [150, 520, 20, 40], bench: [300, 590, 56, 46], flowers: [60, 500, 40, 38], scarecrow: [176, 268, 44, 52] };
const DECOR_ART: Record<string, () => string> = { lantern: lanternArt, bench: benchArt, flowers: flowerBarrelArt, scarecrow: scarecrowArt };
const STATIC = { press: pressArt(true), pressIdle: pressArt(false), tank: tankArt(true), tankIdle: tankArt(false), cellar: cellarArt(), bottling: bottlingArt(), tree: treeArt(), lantern: lanternArt() };

interface Pill { t: string; cls?: string; icon?: string }
interface Obj { k: string; x: number; y: number; w: number; h: number; svg: string; tap?: (e: MouseEvent) => void; pill?: Pill | null; pillY?: number; ready?: boolean; ring?: number | null; plus?: boolean; bubbles?: boolean; locked?: boolean; lantern?: boolean }

const level = computed(() => (state.value ? levelOf(state.value.xp) : 1));
const t = computed(() => now.value || Date.now());

function jobState(b: { jobs: { readyAt: number; startedAt: number }[] }) {
  const ready = b.jobs.filter((j) => j.readyAt <= t.value);
  const running = b.jobs.filter((j) => j.readyAt > t.value).sort((a, c) => a.readyAt - c.readyAt);
  const next = running[0];
  return { ready: ready.length, next, pct: next ? (t.value - next.startedAt) / (next.readyAt - next.startedAt) : 0 };
}

const objects = computed<Obj[]>(() => {
  const s = state.value;
  if (!s) return [];
  const list: Obj[] = [];
  const press = jobState(s.press), tank = jobState(s.tank);
  list.push({ k: "press", x: 10, y: 178, w: 112, h: 125, svg: press.next || press.ready ? STATIC.press : STATIC.pressIdle, tap: (e) => building("press", e),
    ready: press.ready > 0, plus: !press.ready && !press.next,
    pill: press.ready ? { t: "Сусло готово", cls: "ok", icon: "ph-bold ph-check" } : press.next ? { t: fmtLeft(press.next.readyAt - t.value), icon: "ph-bold ph-timer" } : { t: "Давильня" } });
  list.push({ k: "tank", x: 128, y: 150, w: 100, h: 112, svg: tank.next || tank.ready ? STATIC.tank : STATIC.tankIdle, tap: (e) => building("tank", e),
    ready: tank.ready > 0, ring: tank.next ? tank.pct : null, bubbles: Boolean(tank.next), plus: !tank.ready && !tank.next,
    pill: tank.ready ? { t: "Вино готово", cls: "ok", icon: "ph-bold ph-check" } : tank.next ? { t: fmtLeft(tank.next.readyAt - t.value), icon: "ph-bold ph-timer" } : { t: "Чан" } });
  const cellarLocked = level.value < BALANCE.barrel.level;
  const aging = s.barrels.filter((b: any) => b.variety && b.readyAt > t.value).sort((a: any, b: any) => a.readyAt - b.readyAt);
  const barrelReady = s.barrels.some((b: any) => b.variety && b.readyAt <= t.value);
  list.push({ k: "cellar", x: 238, y: 188, w: 128, h: 128, svg: STATIC.cellar, locked: cellarLocked, ready: barrelReady, tap: () => { if (cellarLocked) game.toast(`Погреб откроется на ${BALANCE.barrel.level} уровне`); else sheet.value = "cellar"; },
    pill: cellarLocked ? { t: `Ур. ${BALANCE.barrel.level}`, icon: "ph-fill ph-lock-simple" } : barrelReady ? { t: "Бочка готова", cls: "ok", icon: "ph-bold ph-check" } : aging.length ? { t: `Бочки · ${fmtLeft(aging[0].readyAt - t.value)}`, icon: "ph ph-hourglass-medium" } : { t: "Погреб · пусто" } });
  list.push({ k: "tree1", x: 310, y: 330, w: 54, h: 63, svg: STATIC.tree });
  list.push({ k: "tree2", x: 2, y: 438, w: 54, h: 63, svg: STATIC.tree });
  list.push({ k: "lamp0", x: 54, y: 250, w: 20, h: 40, svg: STATIC.lantern, lantern: true });
  list.push({ k: "bottling", x: 276, y: 490, w: 92, h: 92, svg: STATIC.bottling, locked: true, tap: () => game.toast(`Линия розлива откроется на ${BALANCE.bottling.level} уровне`),
    pill: { t: `Ур. ${BALANCE.bottling.level}`, icon: "ph-fill ph-lock-simple" } });
  for (const d of s.decor as string[]) {
    const p = DECOR_POS[d];
    if (p) list.push({ k: `decor-${d}`, x: p[0], y: p[1], w: p[2], h: p[3], svg: DECOR_ART[d](), lantern: d === "lantern" });
  }
  const nextPlot = s.plots.find((p: any) => !p.owned);
  s.plots.forEach((p: any, i: number) => {
    const pos = PLOT_POS[i];
    if (!p.owned) {
      if (p !== nextPlot) return;
      list.push({ k: `plot${i}`, x: pos.x, y: pos.y, w: 120, h: 106, svg: plotArt("empty"), locked: true, pillY: 94,
        tap: () => router.push("/winery/shop?tab=plots"), pill: { t: "Купить участок", cls: "dashed", icon: "ph-bold ph-storefront" } });
      return;
    }
    const st = plotStage(p, t.value, speed.value);
    let pill: Pill;
    if (st === "ripe" || st === "overripe") pill = { t: st === "ripe" ? "Собрать" : "Перезрел", cls: "accent bounce", icon: "ph-bold ph-hand-grabbing" };
    else if (st === "problem") pill = { t: "Скворцы!", cls: "gold bounce", icon: "ph-bold ph-bird" };
    else if (st === "empty") pill = { t: "Посадить", cls: "dashed", icon: "ph-bold ph-plus" };
    else pill = { t: fmtLeft(p.readyAt - t.value), icon: "ph-bold ph-timer" };
    list.push({ k: `plot${i}`, x: pos.x, y: pos.y, w: 120, h: 106, svg: plotArt(st, p.variety ? VARIETIES[p.variety].col : undefined), pill, pillY: 94, tap: (e) => plotTap(i, st, e) });
  });
  return list.sort((a, b) => a.y + a.h - (b.y + b.h));
});

// --- Действия на поле
const sheet = ref<"" | "press" | "tank" | "cellar" | "settings">("");
const plotSheet = ref<number | null>(null);
function at(e: MouseEvent) {
  const el = (e.currentTarget as HTMLElement).closest(".obj") as HTMLElement | null;
  if (!el) return undefined;
  return { x: parseFloat(el.style.left) + parseFloat(el.style.width) / 2, y: parseFloat(el.style.top) + Y0 + 20 };
}
async function plotTap(i: number, st: string, e: MouseEvent) {
  if (st === "ripe" || st === "overripe") await act({ type: "harvest", plot: i }, at(e));
  else if (st === "problem") { if (await act({ type: "shoo", plot: i }, at(e))) game.toast("Урожай спасён", at(e)); }
  else plotSheet.value = i;
}
async function building(b: "press" | "tank", e: MouseEvent) {
  const ready = state.value[b].jobs.some((j: any) => j.readyAt <= t.value);
  if (ready) await act({ type: "collect", building: b }, at(e));
  else sheet.value = b;
}

// --- Подсказка Виноградинки
const hint = computed(() => {
  const s = state.value;
  if (!s) return "";
  const stages = s.plots.map((p: any) => (p.owned ? plotStage(p, t.value, speed.value) : "none"));
  const ripe = stages.indexOf("ripe");
  if (stages.includes("problem")) return `Ой, скворцы на ${VARIETIES[s.plots[stages.indexOf("problem")].variety].name} — прогоните их одним касанием`;
  if (ripe >= 0) return `${VARIETIES[s.plots[ripe].variety].name} созрел! Нажмите на гроздь, чтобы собрать`;
  if (fulfillable.value) return "Заказ можно отправить — загляните на доску заказов";
  const grapes = Object.entries(s.items).find(([k, n]) => k.startsWith("grape:") && (n as number) >= BALANCE.press.input);
  if (grapes && s.press.jobs.length < s.press.slots) return `Давильня свободна. Три грозди ${VARIETIES[grapes[0].split(":")[1]].name} — и будет сусло`;
  if (stages.includes("empty")) return "Свободный участок ждёт новую лозу";
  return "Всё растёт! Загляните позже — или отсканируйте вино, чтобы открыть новый сорт";
});

// --- Счётчики на кнопках
const fulfillable = computed(() => (state.value ? state.value.orders.slots.filter((sl: any) => sl.order && (state.value.items[sl.order.item] || 0) >= sl.order.count).length : 0));
const questsReady = computed(() => {
  const s = state.value;
  if (!s?.daily) return 0;
  const today = dayKey(t.value);
  return s.daily.quests.filter((q: any) => !q.claimed && questProgress(s, q) >= q.target).length
    + (s.calendar.last !== today ? 1 : 0) + (!s.daily.chest && s.daily.quests.every((q: any) => q.claimed) ? 1 : 0)
    + (!s.week.claimed && s.week.days.length >= BALANCE.week.days ? 1 : 0);
});
const wordleNew = computed(() => state.value?.wordle?.last !== dayKey(t.value));
const progress = computed(() => levelProgress(state.value?.xp || 0));
const name = computed(() => auth.user.value?.name?.split(" ")[0] || "Моя");
const night = computed(() => { const h = new Date(t.value).getHours(); return h >= 21 || h < 6; });
const floats = computed(() => toasts.value.filter((x) => x.x !== undefined));

// --- Онбординг
const TOUR = [
  { tag: "ШАГ 1 · ЗНАКОМСТВО", title: "Привет! Я Виноградинка", text: "Это ваша винодельня. Покажу, как получить первое вино — от лозы до заказа.", next: "Начнём" },
  { tag: "ШАГ 2 · СБОР", title: "Шардоне созрел!", text: "Нажмите на спелую гроздь — виноград отправится на склад, а дальше в давильню.", target: "plot0" },
  { tag: "ШАГ 3 · ДАВИЛЬНЯ", title: "Выжмем сусло", text: "Нажмите на давильню и положите три грозди Шардоне.", target: "press" },
  { tag: "ШАГ 4 · ПОСАДКА", title: "Посадите лозу", text: "Нажмите на свободный участок и выберите саженец. Лоза растёт, пока вас нет.", target: "plotEmpty" },
  { tag: "ШАГ 5 · ЗАКАЗЫ", title: "Заказы — главный доход", text: "Рестораны и гости покупают виноград, сусло и вино. Чем дольше путь, тем дороже.", target: "orders", next: "Дальше" },
  { tag: "ШАГ 6 · СКАНЕР", title: "Сканы открывают сорта", text: "Отсканируйте этикетку в приложении — получите саженец редкого сорта и Грозди. Отзывы приносят репутацию.", next: "Играть" },
];
const tourOn = computed(() => status.value === "ready" && state.value && !state.value.tutorial.done && !sheet.value && plotSheet.value === null);
const step = computed(() => Math.min(state.value?.tutorial.step || 0, TUTORIAL_STEPS - 1));
const tour = computed(() => TOUR[step.value]);
const spot = computed(() => {
  const target = tour.value?.target;
  const s = state.value;
  if (!target || !s) return null;
  if (target === "orders") {
    const r = ordersBtn.value?.$el?.getBoundingClientRect();
    return r ? { x: r.left + r.width / 2, y: r.top + 26, r: 44 } : null;
  }
  let p: { x: number; y: number } | null = null;
  if (target === "plot0") p = { x: PLOT_POS[0].x + 60, y: PLOT_POS[0].y + 52 };
  if (target === "press") p = { x: 66, y: 240 };
  if (target === "plotEmpty") {
    const i = s.plots.findIndex((pl: any) => pl.owned && !pl.variety);
    if (i >= 0) p = { x: PLOT_POS[i].x + 60, y: PLOT_POS[i].y + 52 };
  }
  if (!p) return null;
  const sp = toScreen(p.x, p.y);
  return { ...sp, r: 64 * scale.value };
});
const dimStyle = computed(() => (spot.value
  ? { background: `radial-gradient(circle at ${spot.value.x}px ${spot.value.y}px, transparent 0 ${spot.value.r}px, rgba(36,26,30,.72) ${spot.value.r + 4}px)` }
  : { background: "rgba(36,26,30,.72)" }));
// Цель внизу экрана (кнопка заказов) — карточка подсказки встаёт над ней, а не поверх.
const sayStyle = computed(() => (spot.value && spot.value.y > vh.value - 300 ? { bottom: `${vh.value - spot.value.y + spot.value.r + 12}px` } : {}));
async function tourNext() {
  if (step.value >= TUTORIAL_STEPS - 1) goal("game_tutorial_done");
  await act({ type: "tutorial", step: step.value });
}
async function tourSkip() { goal("game_tutorial_skip", { step: step.value }); await act({ type: "tutorial", skip: true }); }

// --- Новый уровень: что открылось
const unlocks = computed(() => {
  const l = levelUp.value;
  if (!l) return [];
  const out = Object.values(VARIETIES).filter((v: any) => !v.scanOnly && v.level === l).map((v: any) => `Сорт ${v.name}`);
  if (l === BALANCE.barrel.level) out.push("Погреб и выдержка в бочках");
  if (l === BALANCE.orders.extraLevel) out.push(`Ещё ${BALANCE.orders.extra} заказа на доске`);
  BALANCE.plots.extra.forEach((p: any, i: number) => { if (p.level === l && l > 1) out.push(`Участок № ${BALANCE.plots.start + i + 1} в магазине`); });
  if (l === BALANCE.bottling.level) out.push("Линия розлива — в следующем релизе");
  return out;
});

// --- Награды за действия в сервисе (отзыв, полка), накопленные с прошлого захода
const KIND_LABEL: Record<string, string> = { scan: "Скан этикетки", review: "Отзыв", shelf: "Вино на полке", sommelier: "Вопрос сомелье" };
const showRewards = computed(() => status.value === "ready" && rewards.value.length > 0 && !tourOn.value);

async function toggleEnabled() {
  const on = state.value.enabled === false;
  await act({ type: "settings", enabled: on });
  goal("game_toggle", { on });
  if (!on) { sheet.value = ""; router.push("/profile"); }
}
</script>

<template>
  <main class="winery" :class="{ 'is-night': night }">
    <div class="bg" v-html="fieldBg()" />

    <!-- Загрузка (1m) -->
    <div v-if="status === 'idle' || status === 'loading'" style="position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; gap: 10px; color: #4d6a3a; font-weight: 600">
      <span class="spinner ok" />Готовим винодельню…
    </div>

    <!-- Тизер без входа (1l) и без подтверждённого возраста -->
    <template v-else-if="status !== 'ready'">
      <div style="position: absolute; inset: 0; filter: blur(3px) saturate(0.9)">
        <div class="scene" :style="{ top: '120px', transform: `translateX(-50%) scale(${Math.min(1.2, vw / 375)})` }">
          <div class="obj" style="left: 10px; top: 38px; width: 112px; height: 125px" v-html="STATIC.press" />
          <div class="obj" style="left: 128px; top: 10px; width: 100px; height: 112px" v-html="STATIC.tank" />
          <div class="obj" style="left: 238px; top: 48px; width: 128px; height: 128px" v-html="STATIC.cellar" />
          <div v-for="(p, i) in PLOT_POS.slice(0, 6)" :key="i" class="obj" :style="{ left: `${p.x}px`, top: `${p.y - Y0}px`, width: '120px', height: '106px' }" v-html="plotArt(['ripe', 'green', 'leaves', 'sapling', 'ripe', 'empty'][i], VARIETIES.saperavi.col)" />
        </div>
      </div>
      <div style="position: absolute; inset: 0; background: linear-gradient(180deg, rgba(254,253,250,0) 20%, rgba(254,253,250,.94) 52%)" />
      <div class="stack" style="position: absolute; left: 50%; transform: translateX(-50%); width: min(400px, calc(100% - 40px)); bottom: calc(var(--tabbar) + 24px + env(safe-area-inset-bottom)); align-items: center; gap: 12px; text-align: center">
        <div style="width: 110px; height: 120px; animation: svBob 3.2s ease-in-out infinite" v-html="mascotArt({ wave: true, mood: 'happy', hat: true })" />
        <h1 class="serif" style="font-size: 28px; line-height: 1.15">Своя винодельня</h1>
        <p style="font-size: 15px; line-height: 1.45; max-width: 310px">Сажайте лозы, делайте вино и выполняйте заказы. Сканы и отзывы в приложении открывают редкие сорта</p>
        <template v-if="status === 'guest'">
          <NuxtLink class="btn btn-primary" :to="{ path: '/login', query: { redirect: '/winery' } }" @click="goal('game_login_click')">Войти и начать</NuxtLink>
        </template>
        <template v-else-if="status === 'adult'">
          <NuxtLink class="btn btn-primary" to="/profile">Подтвердить возраст в профиле</NuxtLink>
        </template>
        <template v-else>
          <p class="small muted">{{ status === "unavailable" ? "На этом сервере аккаунты не настроены — игра недоступна." : "Не удалось загрузить винодельню." }}</p>
          <button v-if="status === 'error'" class="btn btn-outline" @click="game.load()">Повторить</button>
        </template>
        <NuxtLink to="/winery/wordle" style="height: 44px; display: flex; align-items: center; font-size: 15px; font-weight: 600">Сыграть в Винордл без входа</NuxtLink>
        <div class="small muted" style="font-size: 12px">Только для пользователей старше 18 лет</div>
      </div>
    </template>

    <!-- Игра выключена в настройках -->
    <div v-else-if="state && state.enabled === false" class="stack" style="position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); width: min(360px, calc(100% - 40px)); align-items: center; text-align: center; gap: 12px; background: var(--bg); border-radius: 28px; padding: 24px">
      <div style="width: 84px; height: 92px" v-html="mascotArt({ mood: 'meh', still: true })" />
      <h1 class="serif-title">Игра выключена</h1>
      <p class="small muted">Пока игра выключена, она не показывается в приложении и не начисляет награды. Винодельня сохранена.</p>
      <button class="btn btn-primary" @click="toggleEnabled">Включить игру</button>
    </div>

    <!-- Игра (1a) -->
    <template v-else-if="state">
      <div class="scene" :style="sceneStyle">
        <div v-for="o in objects" :key="o.k" class="obj" :class="{ tap: !!o.tap, locked: o.locked }"
          :style="{ left: `${o.x}px`, top: `${o.y - Y0}px`, width: `${o.w}px`, height: `${o.h}px` }" @click="o.tap?.($event)">
          <span class="art" v-html="o.svg" />
        </div>
        <!-- Второй проход: бейджи и подписи поверх всех объектов, как в макете -->
        <div v-for="o in objects" :key="`ov-${o.k}`" class="obj" style="pointer-events: none"
          :style="{ left: `${o.x}px`, top: `${o.y - Y0}px`, width: `${o.w}px`, height: `${o.h}px` }">
          <span v-if="o.ready" class="badge-ok" :style="{ left: `${o.w - 22}px` }"><i class="ph-bold ph-check" /></span>
          <span v-else-if="o.plus" class="badge-plus"><i class="ph-bold ph-plus" /></span>
          <span v-if="o.ring != null" class="ring"><GameRing :value="o.ring" :size="24" color="#8a5a92" /></span>
          <template v-if="o.bubbles"><span v-for="i in 3" :key="i" class="bub" :style="{ left: `${44 + i * 6}%`, animationDelay: `${i * 0.6}s` }" /></template>
          <button v-if="o.pill" class="g-pill" :class="o.pill.cls" :style="{ top: `${o.pillY || o.h - 10}px`, cursor: o.tap ? 'pointer' : 'default', pointerEvents: o.tap ? 'auto' : 'none' }" tabindex="-1" @click="o.tap?.($event)">
            <i v-if="o.pill.icon" :class="o.pill.icon" />{{ o.pill.t }}
          </button>
        </div>
        <div class="obj" style="left: 16px; width: 66px; height: 73px; animation: svBob 3.2s ease-in-out infinite" :style="{ top: `${536 - Y0}px` }" v-html="mascotArt({ hat: level >= 3, apron: level >= 7, mood: 'happy' })" />
        <div v-if="!tourOn && hint" class="say" :style="{ left: '86px', top: `${538 - Y0}px` }">{{ hint }}</div>
        <template v-if="night">
          <div v-for="o in objects.filter((x) => x.lantern)" :key="`g${o.k}`" class="glow" :style="{ left: `${o.x + o.w / 2 - 50}px`, top: `${o.y - Y0 - 32}px` }" />
        </template>
        <div v-for="f in floats" :key="f.id" class="float" :style="{ left: `${f.x}px`, top: `${(f.y || 0) - Y0}px` }">{{ f.text }}</div>
      </div>
      <div v-if="night" class="night" />

      <header ref="hudEl" class="hud" :style="{ visibility: tourOn ? 'hidden' : 'visible' }">
        <div class="row" style="gap: 10px">
          <div class="lvl"><GameRing :value="progress.pct" :size="44" /><span>{{ progress.level }}</span></div>
          <div class="grow">
            <div class="serif" style="font-size: 17px; line-height: 1.2">Винодельня «{{ name }}»</div>
            <div class="small muted" style="font-size: 12px">Уровень {{ progress.level }} · {{ fmtNum(state.xp) }}{{ progress.next ? ` / ${fmtNum(progress.next)}` : "" }} XP</div>
          </div>
          <button class="icon-btn" style="width: 38px; height: 38px; font-size: 18px" aria-label="Настройки игры" @click="sheet = 'settings'"><i class="ph ph-gear-six" /></button>
        </div>
        <div class="curs">
          <div class="g-cur" title="Монеты"><span class="art-inline" v-html="ICON.coins" />{{ fmtNum(state.coins) }}</div>
          <div class="g-cur" title="Грозди"><span class="art-inline" v-html="ICON.grozdi" />{{ state.grozdi }}</div>
          <div class="g-cur" title="Репутация"><span class="art-inline" v-html="ICON.rep" />{{ state.rep }}</div>
        </div>
      </header>
      <NuxtLink v-if="!tourOn" to="/winery/wordle" class="chip-daily" :style="{ top: `${hudBottom - 22}px` }"><span class="art-inline" v-html="ICON.grozdi" />Вино дня<span v-if="wordleNew" class="dot" /></NuxtLink>

      <nav ref="actionsEl" class="actions" aria-label="Винодельня">
        <NuxtLink to="/winery/storage"><span class="ico"><i class="ph-fill ph-warehouse" /></span><span class="lbl">Склад</span></NuxtLink>
        <NuxtLink ref="ordersBtn" to="/winery/orders"><span class="ico"><i class="ph-fill ph-clipboard-text" /></span><span class="lbl">Заказы</span><span v-if="fulfillable" class="cnt">{{ fulfillable }}</span></NuxtLink>
        <NuxtLink to="/winery/quests"><span class="ico"><i class="ph-fill ph-list-checks" /></span><span class="lbl">Задания</span><span v-if="questsReady" class="cnt">{{ questsReady }}</span></NuxtLink>
        <NuxtLink to="/winery/shop"><span class="ico"><i class="ph-fill ph-storefront" /></span><span class="lbl">Магазин</span></NuxtLink>
      </nav>

      <!-- Онбординг (1j) -->
      <template v-if="tourOn && tour">
        <div class="tour-dim" :style="dimStyle" />
        <div v-if="spot" class="tour-ring" :style="{ left: `${spot.x - spot.r}px`, top: `${spot.y - spot.r}px`, width: `${spot.r * 2}px`, height: `${spot.r * 2}px` }" />
        <div class="tour-top">
          <div class="tour-steps"><span v-for="i in TUTORIAL_STEPS" :key="i" :class="{ on: i <= step + 1 }" /></div>
          <button @click="tourSkip">Пропустить</button>
        </div>
        <div class="tour-say" :style="sayStyle">
          <div style="width: 96px; height: 106px; flex: none; animation: svBob 3.2s ease-in-out infinite" v-html="mascotArt({ hat: true, wave: true, mood: 'happy' })" />
          <div class="card">
            <div style="font-size: 12px; font-weight: 700; color: var(--accent)">{{ tour.tag }}</div>
            <div class="serif" style="font-size: 20px; line-height: 1.2">{{ tour.title }}</div>
            <div style="font-size: 14px; line-height: 1.4">{{ tour.text }}</div>
            <button v-if="tour.next" class="btn btn-primary" style="min-height: 44px; font-size: 15px; margin-top: 4px" @click="tourNext">{{ tour.next }}</button>
          </div>
        </div>
      </template>
    </template>

    <GamePlotSheet :index="plotSheet" @close="plotSheet = null" />
    <GameBuildingSheet :building="sheet === 'press' || sheet === 'tank' ? sheet : null" @close="sheet = ''" />
    <GameCellarSheet :open="sheet === 'cellar'" @close="sheet = ''" />

    <!-- Новый уровень -->
    <BottomSheet :open="!!levelUp" label="Новый уровень" @close="levelUp = null">
      <div class="stack" style="align-items: center; text-align: center; gap: 12px; padding: 10px 0">
        <div style="position: relative; width: 120px; height: 120px; display: flex; align-items: center; justify-content: center">
          <Confetti />
          <div style="width: 96px; height: 96px; border-radius: 50%; background: var(--bg); border: 4px solid var(--coin); display: flex; flex-direction: column; align-items: center; justify-content: center; animation: svPopIn 2.4s ease-out 1">
            <span class="serif" style="font-size: 38px; line-height: 1">{{ levelUp }}</span><span style="font-size: 10px; font-weight: 700; color: var(--text-2)">УРОВЕНЬ</span>
          </div>
        </div>
        <h2 class="serif-title">Новый уровень!</h2>
        <div class="row" style="justify-content: center"><span class="g-cur"><span class="art-inline" v-html="ICON.coins" />+{{ 50 * (levelUp || 0) }}</span><span class="g-cur"><span class="art-inline" v-html="ICON.grozdi" />+3</span></div>
        <div v-if="unlocks.length" class="stack" style="gap: 6px; width: 100%">
          <div v-for="u in unlocks" :key="u" class="done-row"><i class="ph-fill ph-lock-open" />{{ u }}</div>
        </div>
      </div>
      <template #foot><button class="btn btn-primary" @click="levelUp = null">Отлично</button></template>
    </BottomSheet>

    <!-- Награды из сервиса -->
    <BottomSheet :open="showRewards" label="Награды из приложения" @close="game.ackRewards()">
      <div class="row" style="gap: 12px">
        <div style="width: 64px; height: 70px; flex: none" v-html="mascotArt({ mood: 'joy', apron: true })" />
        <div class="grow"><div style="font-size: 12px; font-weight: 700; color: var(--accent)">ПОКА ВАС НЕ БЫЛО</div><div class="serif" style="font-size: 22px; line-height: 1.2">Награды из приложения</div></div>
      </div>
      <div class="stack" style="gap: 8px">
        <div v-for="r in rewards" :key="r.kind + r.ref" class="done-row" style="align-items: flex-start">
          <i class="ph-fill ph-gift" style="color: var(--accent)" />
          <div class="grow">
            <b>{{ KIND_LABEL[r.kind] || r.kind }}</b><span v-if="r.reward.name" class="muted"> · {{ r.reward.name }}</span>
            <div class="small muted">{{ describe(r.reward) }}<template v-if="r.reward.unlocked?.length"> · открыт сорт {{ r.reward.unlocked.map((v: string) => VARIETIES[v]?.name).join(", ") }}</template></div>
          </div>
        </div>
      </div>
      <template #foot><button class="btn btn-primary" @click="game.ackRewards()">Забрать</button></template>
    </BottomSheet>

    <!-- Настройки -->
    <BottomSheet :open="sheet === 'settings'" label="Настройки игры" @close="sheet = ''">
      <div class="g-head"><h2>Настройки</h2></div>
      <div class="menu">
        <button role="switch" :aria-checked="state?.enabled !== false" @click="toggleEnabled">
          <i class="ph ph-plant" /><span class="grow">Игра «Моя винодельня»<span class="sub" style="display: block">выключенная игра нигде не показывается</span></span><span class="switch" :class="{ on: state?.enabled !== false }" />
        </button>
        <NuxtLink to="/winery/wordle"><i class="ph ph-puzzle-piece" /><span class="grow">Винордл</span><i class="ph ph-caret-right caret" /></NuxtLink>
      </div>
      <p class="small muted">Игра — про виноделие, а не про употребление: награды даются за сканы, полку и отзывы, а не за количество вина. Только 18+.</p>
    </BottomSheet>

    <GameToasts />
  </main>
</template>
