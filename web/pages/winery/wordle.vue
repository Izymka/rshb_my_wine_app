<script setup lang="ts">
// Винордл (1g) и результат (1h): угадать вино дня за 6 попыток по цвету, региону, сорту и
// винодельне. Без входа попытки хранятся в браузере, награды и серия — только у вошедших.
import { WORDLE_TRIES } from "~/utils/wordle.mjs";
import { agedArt, mascotArt } from "~/utils/game-art.mjs";
import { nextDayAt } from "~/utils/game.mjs";

interface Cell { t: string; s: "ok" | "near" | "no" }
interface View { number: number; solved: boolean; lost: boolean; attempt: number | null; rows: { id: number; name: string; cells: Cell[] }[]; hints: { label: string; value: string | null }[]; answer: null | { name: string; winery: string; region: string; color: string; grape: string } }
interface Resp { guest: boolean; options?: { id: number; name: string; winery: string }[]; view: View; stats?: { players: number; solvedPct: number | null } | null; streak?: number; reward?: { gained?: Record<string, number> } | null; state?: unknown }

const game = useGame();
const now = useGameClock();
const { goal } = useMetrika();
useHead({ title: "Винордл — Своё Вино" });
const GUEST_KEY = "wine-wordle:guest";

const data = ref<Resp | null>(null);
const options = ref<{ id: number; name: string; winery: string }[]>([]);
const guestGuesses = ref<number[]>([]);
const q = ref("");
const busy = ref(false);
const error = ref("");
const showResult = ref(false);
const gained = ref<Record<string, number> | null>(null);

onMounted(async () => {
  try {
    const r = await $fetch<Resp>("/api/game/wordle");
    options.value = r.options || [];
    data.value = r;
    if (r.guest) {
      try {
        const saved = JSON.parse(localStorage.getItem(GUEST_KEY) || "null");
        if (saved?.number === r.view.number && Array.isArray(saved.guesses) && saved.guesses.length) {
          guestGuesses.value = saved.guesses;
          await replayGuest();
        }
      } catch {}
    }
    if (data.value?.view.solved || data.value?.view.lost) showResult.value = true;
  } catch { error.value = "Не удалось загрузить Винордл. Проверьте сеть."; }
});

// Гость: сервер не хранит попытки, поэтому прошлый прогресс восстанавливаем, отправив последнюю
// попытку вместе с предыдущими.
async function replayGuest() {
  const list = guestGuesses.value;
  const r = await $fetch<Resp>("/api/game/wordle", { method: "POST", body: { guess: list[list.length - 1], guesses: list.slice(0, -1) } });
  data.value = { ...data.value!, view: r.view };
}

const view = computed(() => data.value?.view);
const playing = computed(() => view.value && !view.value.solved && !view.value.lost);
const tried = computed(() => new Set(view.value?.rows.map((r) => r.id) || []));
const sugg = computed(() => {
  const s = q.value.trim().toLowerCase();
  if (!s) return [];
  return options.value.filter((o) => !tried.value.has(o.id) && `${o.name} ${o.winery}`.toLowerCase().includes(s)).slice(0, 5);
});
const rows = computed(() => Array.from({ length: WORDLE_TRIES }, (_, i) => view.value?.rows[i] || null));
const left = computed(() => WORDLE_TRIES - (view.value?.rows.length || 0));

async function guess(id: number) {
  if (busy.value) return;
  busy.value = true;
  error.value = "";
  q.value = "";
  try {
    const guest = data.value?.guest;
    const r = await $fetch<Resp>("/api/game/wordle", { method: "POST", body: guest ? { guess: id, guesses: guestGuesses.value } : { guess: id } });
    data.value = { ...data.value!, ...r, options: undefined };
    if (guest) {
      guestGuesses.value = [...guestGuesses.value, id];
      try { localStorage.setItem(GUEST_KEY, JSON.stringify({ number: r.view.number, guesses: guestGuesses.value })); } catch {}
    }
    if (r.state) game.accept({ state: r.state, now: Date.now() + game.offset.value, speed: game.speed.value });
    if (r.reward?.gained) gained.value = r.reward.gained;
    if (r.view.solved || r.view.lost) {
      goal("game_wordle", { solved: r.view.solved, attempt: r.view.attempt || 0, guest: Boolean(guest) });
      setTimeout(() => { showResult.value = true; }, 600);
    }
  } catch (e) { error.value = apiError(e); }
  finally { busy.value = false; }
}

const COLORS: Record<string, string> = { ok: "#6f9a4f", near: "#e0ad3c", no: "#e3d9c8" };
const shareText = computed(() => {
  const v = view.value;
  if (!v) return "";
  const em: Record<string, string> = { ok: "🟩", near: "🟨", no: "⬜" };
  return `Винордл № ${v.number} — ${v.solved ? `${v.attempt}/${WORDLE_TRIES}` : `X/${WORDLE_TRIES}`}\n${v.rows.map((r) => r.cells.map((c) => em[c.s]).join("")).join("\n")}\n${location.origin}/winery/wordle`;
});
async function share() {
  goal("game_wordle_share");
  try {
    if (navigator.share) await navigator.share({ text: shareText.value });
    else { await navigator.clipboard.writeText(shareText.value); game.toast("Результат скопирован"); }
  } catch {}
}
const nextIn = computed(() => fmtLeft(nextDayAt(now.value || Date.now()) - (now.value || Date.now())));
</script>

<template>
  <main class="g-page" style="gap: 10px">
    <div class="top">
      <NuxtLink to="/winery" class="icon-btn" aria-label="К винодельне"><i class="ph ph-arrow-left" /></NuxtLink>
      <div class="grow"><div class="serif" style="font-size: 24px; line-height: 1.1">Винордл</div><div class="small muted" style="font-size: 12px">№ {{ view?.number || "…" }} · одно вино на всех</div></div>
      <button v-if="view && !playing" class="icon-btn" aria-label="Результат" @click="showResult = true"><i class="ph ph-trophy" /></button>
    </div>
    <p v-if="error" class="error" role="alert">{{ error }}</p>
    <template v-if="view">
      <div class="row" style="flex-wrap: wrap; gap: 6px">
        <span v-for="h in view.hints" :key="h.label" class="row" style="gap: 5px; padding: 5px 10px; border-radius: 999px; font-size: 12px" :style="{ background: h.value ? 'var(--sand)' : 'var(--seg)', color: h.value ? 'var(--text)' : 'var(--text-2)' }">
          <i :class="h.value ? 'ph-fill ph-lightbulb' : 'ph ph-lock-simple'" />{{ h.value ? `${h.label}: ${h.value}` : h.label }}
        </span>
      </div>
      <div class="g-grid4" style="gap: 6px; font-size: 11px; color: var(--text-2); font-weight: 600; text-align: center; padding-top: 4px"><span>Цвет</span><span>Регион</span><span>Сорт</span><span>Винодельня</span></div>
      <div class="stack" style="gap: 7px">
        <div v-for="(row, i) in rows" :key="i" class="stack" style="gap: 3px">
          <div style="font-size: 12px; font-weight: 600; min-height: 15px; line-height: 15px">{{ row?.name || "" }}</div>
          <div class="g-grid4" style="gap: 6px">
            <template v-if="row">
              <span v-for="(c, j) in row.cells" :key="j" style="height: 34px; border-radius: 10px; display: flex; align-items: center; justify-content: center; text-align: center; font-size: 11px; line-height: 1.1; font-weight: 600; padding: 0 3px; overflow: hidden"
                :style="{ background: COLORS[c.s], color: c.s === 'ok' ? '#fff' : c.s === 'near' ? 'var(--text)' : 'var(--text-2)' }">{{ c.t }}</span>
            </template>
            <template v-else><span v-for="j in 4" :key="j" style="height: 34px; border-radius: 10px; border: 1.5px dashed var(--line-2); background: #fff" /></template>
          </div>
        </div>
      </div>
      <div class="row" style="gap: 12px; font-size: 11px; color: var(--text-2)">
        <span class="row" style="gap: 4px"><span style="width: 10px; height: 10px; border-radius: 3px; background: #6f9a4f" />совпало</span>
        <span class="row" style="gap: 4px"><span style="width: 10px; height: 10px; border-radius: 3px; background: #e0ad3c" />тот же регион</span>
        <span class="row" style="gap: 4px"><span style="width: 10px; height: 10px; border-radius: 3px; background: #efe6d8" />мимо</span>
      </div>
      <div style="flex: 1" />
      <div v-if="playing" style="position: relative">
        <div v-if="sugg.length" style="position: absolute; left: 0; right: 0; bottom: 60px; background: #fff; border: 1px solid var(--line); border-radius: 18px; box-shadow: 0 12px 30px -12px rgba(60,40,20,.4); overflow: hidden; z-index: 5">
          <button v-for="s in sugg" :key="s.id" style="display: block; width: 100%; text-align: left; padding: 10px 14px; border: 0; border-bottom: 1px solid var(--line); background: #fff; cursor: pointer" @click="guess(s.id)">
            <div style="font-size: 14px; font-weight: 600">{{ s.name }}</div><div class="small muted" style="font-size: 12px">{{ s.winery }}</div>
          </button>
        </div>
        <label class="row" style="height: 52px; border-radius: 999px; border: 1.5px solid var(--accent); background: #fff; padding: 0 16px; gap: 10px">
          <i class="ph ph-magnifying-glass" style="font-size: 18px; color: var(--text-2)" />
          <input v-model="q" :disabled="busy" placeholder="Начните вводить: «Сап…»" style="flex: 1; min-width: 0; border: 0; outline: 0; background: none; font-size: 16px; color: var(--text)" @keydown.enter="sugg[0] && guess(sugg[0].id)" />
          <span class="small muted" style="font-size: 12px; white-space: nowrap">осталось {{ left }}</span>
        </label>
      </div>
      <button v-else-if="view.solved" class="row" style="background: var(--ok-bg); border-radius: 20px; padding: 12px 14px; gap: 12px; border: 0; cursor: pointer; text-align: left" @click="showResult = true">
        <span style="width: 44px; height: 48px; flex: none" v-html="mascotArt({ mood: 'joy', still: true })" />
        <div class="grow"><div style="font-size: 15px; font-weight: 700; color: var(--ok-ink)">Угадано с {{ view.attempt }}-й попытки!</div><div class="small" style="color: var(--ok-ink)">{{ data?.guest ? "Войдите, чтобы получать Грозди" : "смотреть результат" }}</div></div>
        <i class="ph-bold ph-arrow-right" style="color: var(--ok-ink)" />
      </button>
      <div v-else style="background: var(--accent-soft); border-radius: 20px; padding: 14px; font-size: 14px; color: var(--accent); font-weight: 600; text-align: center">Попытки закончились. Это был {{ view.answer?.name }} · {{ view.answer?.winery }}</div>
    </template>
    <p v-else-if="!error" class="muted">Загружаем вино дня…</p>

    <!-- 1h Результат -->
    <BottomSheet :open="showResult && !!view?.answer" label="Результат Винордла" @close="showResult = false">
      <div v-if="view?.answer" class="stack" style="align-items: center; text-align: center; gap: 14px">
        <div style="position: relative; width: 120px; height: 130px">
          <Confetti v-if="view.solved" />
          <div style="width: 100%; height: 100%; animation: svBob 3.2s ease-in-out infinite" v-html="mascotArt({ hat: true, mood: view.solved ? 'joy' : 'meh', wave: view.solved })" />
        </div>
        <div class="serif" style="font-size: 28px; line-height: 1.15">{{ view.solved ? `Угадано с ${view.attempt}-й попытки` : "В этот раз не вышло" }}</div>
        <div class="row" style="width: 100%; background: #fff; border: 1px solid var(--line); border-radius: 22px; padding: 14px; gap: 12px; text-align: left">
          <span style="width: 44px; height: 64px; flex: none" v-html="agedArt()" />
          <div class="grow"><div style="font-size: 16px; font-weight: 600">{{ view.answer.name }}</div><div class="small muted">{{ view.answer.winery }} · {{ view.answer.region }} · {{ view.answer.color.toLowerCase() }}</div></div>
        </div>
        <div class="stack" style="gap: 5px; align-items: center">
          <div v-for="r in view.rows" :key="r.id" class="row" style="gap: 5px"><span v-for="(c, j) in r.cells" :key="j" style="width: 26px; height: 26px; border-radius: 7px" :style="{ background: COLORS[c.s] }" /></div>
        </div>
        <div v-if="gained" class="row" style="gap: 8px; justify-content: center">
          <span v-if="gained.grozdi" class="g-cur" style="background: #fff; font-size: 15px"><span class="art-inline" style="width: 22px; height: 22px" v-html="ICON.grozdi" />+{{ gained.grozdi }}</span>
          <span v-if="gained.xp" class="g-cur" style="background: #fff; font-size: 15px"><span class="art-inline" style="width: 22px; height: 22px" v-html="ICON.xp" />+{{ gained.xp }}</span>
        </div>
        <NuxtLink v-else-if="data?.guest" :to="{ path: '/login', query: { redirect: '/winery' } }" class="small" style="font-weight: 600">Войдите — за Винордл дают Грозди для винодельни</NuxtLink>
        <div v-if="!data?.guest" class="small muted">Серия: {{ data?.streak || 0 }} {{ (data?.streak || 0) === 1 ? "день" : "дн." }}<template v-if="data?.stats?.solvedPct != null"> · {{ data.stats.solvedPct }}% игроков угадали сегодня</template></div>
      </div>
      <template #foot>
        <div class="stack" style="gap: 8px">
          <button class="btn btn-primary" @click="share"><i class="ph-bold ph-share-network" />Поделиться результатом</button>
          <div class="small muted" style="text-align: center">Следующее вино через <b style="color: var(--text); font-variant-numeric: tabular-nums">{{ nextIn }}</b></div>
        </div>
      </template>
    </BottomSheet>
    <GameToasts />
  </main>
</template>
