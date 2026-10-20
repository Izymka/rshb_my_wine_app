<script setup lang="ts">
// Задания и календарь (1f): три задания дня (одно — из сервиса), сундук дня, календарь входа на
// 7 дней без сброса за пропуск и недельный сундук за 5 активных дней.
import { BALANCE, CALENDAR, DECOR, QUESTS, VARIETIES, dayKey, nextDayAt, questProgress } from "~/utils/game.mjs";
import { chestArt, lanternArt, saplingArt } from "~/utils/game-art.mjs";
const game = useGame();
const { state, status, act } = game;
const now = useGameClock();
useHead({ title: "Задания — Моя винодельня" });
onMounted(() => { void game.load(); });
watch(status, (s) => { if (s === "guest" || s === "adult") navigateTo("/winery"); });

const quests = computed(() => {
  const s = state.value;
  if (!s?.daily) return [];
  return s.daily.quests.map((q: any, i: number) => {
    const t = QUESTS[q.kind];
    const p = questProgress(s, q);
    return { ...q, i, title: t.title(q.target), icon: t.icon, service: !!t.service, go: t.go, href: t.href, p, pct: `${Math.round((p / q.target) * 100)}%`, can: !q.claimed && p >= q.target };
  });
});
const chestSteps = computed(() => quests.value.map((q: any) => q.claimed));
const chestReady = computed(() => state.value && !state.value.daily.chest && quests.value.every((q: any) => q.claimed));
const resetIn = computed(() => fmtLeft(nextDayAt(now.value || Date.now()) - (now.value || Date.now())));

const today = computed(() => dayKey(now.value || Date.now()));
const calendar = computed(() => {
  const s = state.value;
  if (!s) return [];
  const claimedToday = s.calendar.last === today.value;
  // Сегодняшняя ячейка — следующая непрошедшая; если уже забрали, «сегодня» — только что взятая.
  const todayIdx = claimedToday ? (s.calendar.idx + CALENDAR.length - 1) % CALENDAR.length : s.calendar.idx;
  return CALENDAR.map((r: any, i: number) => {
    const got = i < s.calendar.idx || (claimedToday && i === todayIdx);
    const isToday = i === todayIdx;
    let label = "", art = ICON.coins;
    if (r.decor) { label = `${DECOR[r.decor].name} · редкий декор`; art = lanternArt(); }
    else if (r.sapling) { label = VARIETIES[r.sapling].name; art = saplingArt(r.sapling === "riesling" ? "#b5c35f" : "#86b05e"); }
    else if (r.grozdi) { label = String(r.grozdi); art = ICON.grozdi; }
    else label = String(r.coins);
    return { i, label, art, got: got && !(isToday && !claimedToday), isToday, can: isToday && !claimedToday, wide: i === 6 };
  });
});
const week = computed(() => state.value?.week);
</script>

<template>
  <main class="g-page">
    <GamePageTop title="Задания"><span class="small muted">обновятся через {{ resetIn }}</span></GamePageTop>
    <p v-if="!state" class="muted">Загружаем задания…</p>
    <div v-else class="stack" style="gap: 10px">
      <div class="row" style="background: var(--bg-cream); border-radius: 22px; padding: 14px; gap: 12px">
        <span style="width: 56px; height: 56px; flex: none" v-html="chestArt()" />
        <div class="grow">
          <div style="font-size: 15px; font-weight: 600">Сундук дня</div>
          <div class="small muted" style="font-size: 12px">{{ state.daily.chest ? "Открыт — завтра новый" : "Выполните все 3 задания" }}</div>
          <div class="row" style="gap: 4px; margin-top: 8px"><span v-for="(c, i) in chestSteps" :key="i" style="flex: 1; height: 6px; border-radius: 3px" :style="{ background: c ? 'var(--leaf)' : 'var(--line)' }" /></div>
        </div>
        <button v-if="chestReady" class="g-mini" @click="act({ type: 'chest' })">Открыть</button>
        <span v-else class="g-price"><span class="art-inline" style="width: 18px; height: 18px" v-html="ICON.grozdi" />{{ BALANCE.quests.chest.grozdi }}</span>
      </div>

      <div v-for="q in quests" :key="q.i" class="row" style="border: 1px solid var(--line); border-radius: 20px; padding: 12px 14px; gap: 12px; background: #fff">
        <span class="g-ico" :style="{ background: q.service ? 'var(--accent-soft)' : 'var(--bg-cream)', color: q.service ? 'var(--accent)' : 'var(--sand-ink)' }"><i :class="['ph', q.icon]" /></span>
        <div class="grow">
          <div style="font-size: 14px; font-weight: 600; line-height: 1.3">{{ q.title }}</div>
          <div v-if="q.service" style="font-size: 11px; font-weight: 700; color: var(--accent); margin-top: 2px">ЗАДАНИЕ ИЗ СЕРВИСА</div>
          <div class="row" style="gap: 8px; margin-top: 6px"><span class="g-bar grow" style="height: 6px"><div :style="{ width: q.pct }" /></span><span class="small muted" style="font-size: 12px">{{ q.p }}/{{ q.target }}</span></div>
        </div>
        <button v-if="q.can" class="g-mini" @click="act({ type: 'quest', index: q.i })">Забрать</button>
        <i v-else-if="q.claimed" class="ph-fill ph-check-circle" style="font-size: 26px; color: var(--ok)" />
        <NuxtLink v-else-if="q.service" :to="q.href" class="g-mini outline">{{ q.go }}</NuxtLink>
        <span v-else class="g-price" style="font-size: 13px"><span class="art-inline" v-html="ICON.xp" />{{ q.xp }}</span>
      </div>

      <div class="row" style="justify-content: space-between; align-items: baseline; padding: 14px 4px 0"><span style="font-size: 16px; font-weight: 600">Календарь входа</span><span class="small muted" style="font-size: 12px">день {{ (calendar.findIndex((d) => d.isToday) + 1) || 1 }} из 7</span></div>
      <div class="g-grid4">
        <button v-for="d in calendar" :key="d.i" class="g-item" :style="{ gridColumn: d.wide ? 'span 2' : 'auto', borderWidth: '1.5px', borderColor: d.can ? 'var(--accent)' : 'var(--line)', background: d.can ? 'var(--bg-cream)' : d.wide ? 'var(--sand)' : '#fff', cursor: d.can ? 'pointer' : 'default' }"
          :disabled="!d.can" @click="act({ type: 'calendar' })">
          <span style="font-size: 11px; font-weight: 700" :style="{ color: d.isToday ? 'var(--accent)' : 'var(--text-2)' }">{{ d.isToday ? "Сегодня" : `День ${d.i + 1}` }}</span>
          <span style="width: 34px; height: 34px" :style="{ opacity: d.got ? 0.45 : 1 }" v-html="d.art" />
          <span class="n">{{ d.label }}</span>
          <span v-if="d.got" style="position: absolute; top: 6px; right: 6px; color: var(--ok); font-size: 16px"><i class="ph-fill ph-check-circle" /></span>
        </button>
      </div>
      <div v-if="week" class="row" style="background: var(--bg-cream); border-radius: 18px; padding: 12px 14px; gap: 10px; font-size: 13px">
        <i class="ph-fill ph-calendar-check" style="font-size: 24px; color: var(--accent)" />
        <span class="grow">Недельный сундук: <b>{{ Math.min(week.days.length, BALANCE.week.days) }} из {{ BALANCE.week.days }}</b> активных дней. Пропуск не обнуляет прогресс</span>
        <button v-if="!week.claimed && week.days.length >= BALANCE.week.days" class="g-mini" @click="act({ type: 'week' })">Открыть</button>
        <i v-else-if="week.claimed" class="ph-fill ph-check-circle" style="font-size: 22px; color: var(--ok)" />
      </div>
    </div>
    <GameToasts />
  </main>
</template>
