<script setup lang="ts">
// Шторка здания (1c): давильня (3 грозди → сусло) или бродильный чан (сусло → молодое вино).
// Слоты, выбор сырья, ускорение за Грозди, улучшение скорости и второй слот.
import { BALANCE, VARIETIES, itemValue, skipCost, upgradeFactor } from "~/utils/game.mjs";
import { grapeRes, mustArt, pressArt, tankArt, youngArt } from "~/utils/game-art.mjs";
const props = defineProps<{ building: "press" | "tank" | null }>();
const emit = defineEmits<{ close: [] }>();
const { state, now, speed, act } = useGame();

const b = computed(() => (props.building && state.value ? state.value[props.building] : null));
const isPress = computed(() => props.building === "press");
const title = computed(() => (isPress.value ? "Давильня" : "Бродильный чан"));
const cfg = computed(() => BALANCE[props.building || "press"]);
const need = computed(() => (isPress.value ? BALANCE.press.input : 1));
const inKind = computed(() => (isPress.value ? "grape" : "must"));
const outKind = computed(() => (isPress.value ? "must" : "young"));
const youngCol = (v: string) => (VARIETIES[v].color === "white" ? "#d9cf7a" : "#5b1f35");
const art = (kind: string, v: string) => (kind === "grape" ? grapeRes(VARIETIES[v].col) : kind === "must" ? mustArt() : youngArt(youngCol(v)));

const jobs = computed(() => (b.value?.jobs || []).map((j: any, i: number) => {
  const total = j.readyAt - j.startedAt;
  return { ...j, i, ready: j.readyAt <= now.value, pct: total > 0 ? Math.min(1, (now.value - j.startedAt) / total) : 1, left: j.readyAt - now.value };
}));
const current = computed(() => jobs.value.find((j: any) => !j.ready) || jobs.value[0] || null);
const anyReady = computed(() => jobs.value.some((j: any) => j.ready));
const free = computed(() => (b.value ? b.value.slots - b.value.jobs.length : 0));
const inputs = computed(() => {
  if (!state.value) return [];
  return Object.keys(VARIETIES).map((v) => ({ v, have: state.value.items[`${inKind.value}:${v}`] || 0 })).filter((x) => x.have > 0);
});
const time = computed(() => (b.value ? cfg.value.time * upgradeFactor(b.value.level) : 0));
const upgradePrice = computed(() => (b.value ? cfg.value.upgrade[b.value.level] ?? null : null));

async function start(v: string) { await act({ type: "start", building: props.building, variety: v }); }
async function collect() { await act({ type: "collect", building: props.building }); }
async function skip() { if (current.value) await act({ type: "skip", building: props.building, index: current.value.i }); }
async function upgrade() { await act({ type: "upgrade", building: props.building }); }
async function buySlot() { await act({ type: "buy", item: `slot:${props.building}` }); }
</script>

<template>
  <BottomSheet :open="!!b" :label="title" @close="emit('close')">
    <template v-if="b">
      <div class="g-head"><h2>{{ title }}</h2><span class="small muted">ур. {{ b.level + 1 }} · {{ fmtDur(time, speed) }}</span></div>
      <div v-if="current" class="row" style="gap: 18px">
        <div style="position: relative; width: 112px; height: 112px; flex: none">
          <GameRing :value="current.pct" :size="112" :stroke="10" color="#8a5a92" />
          <div style="position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center">
            <template v-if="!current.ready"><span style="font-size: 20px; font-weight: 700; font-variant-numeric: tabular-nums">{{ fmtLeft(current.left) }}</span><span class="small muted" style="font-size: 11px">осталось</span></template>
            <template v-else><i class="ph-bold ph-check" style="font-size: 30px; color: var(--leaf)" /><span class="small muted" style="font-size: 11px">готово</span></template>
          </div>
        </div>
        <div class="grow stack" style="gap: 10px">
          <div class="row" style="gap: 8px"><span style="width: 36px; height: 36px; flex: none" v-html="art(inKind, current.variety)" /><div class="small" style="line-height: 1.25">{{ isPress ? `${need} грозди` : "Сусло" }}<br /><span class="muted">{{ VARIETIES[current.variety].name }}</span></div></div>
          <i class="ph-bold ph-arrow-down" style="color: var(--text-3); margin-left: 10px" />
          <div class="row" style="gap: 8px"><span style="width: 36px; height: 36px; flex: none" v-html="art(outKind, current.variety)" /><div class="small" style="line-height: 1.25">{{ isPress ? "Сусло" : "Молодое вино" }}<br /><span class="muted">{{ VARIETIES[current.variety].name }} · {{ itemValue(`${outKind}:${current.variety}`) }} монет</span></div></div>
        </div>
      </div>
      <div v-else class="row" style="gap: 14px">
        <div style="width: 96px; height: 108px; flex: none" v-html="isPress ? pressArt(false) : tankArt(false)" />
        <p class="small muted grow">{{ isPress ? `Положите ${need} грозди одного сорта — через ${fmtDur(time, speed)} будет сусло.` : `Залейте сусло — через ${fmtDur(time, speed)} получится молодое вино.` }}</p>
      </div>

      <div class="stack" style="gap: 8px">
        <div style="font-size: 14px; font-weight: 600">Слоты</div>
        <div class="grid-2">
          <div v-for="j in jobs" :key="j.i" class="g-slot" :class="j.ready ? 'done' : 'busy'">
            <span class="pic" v-html="art(outKind, j.variety)" />
            <div><b>{{ j.ready ? "Готово" : isPress ? "Давим" : "Брожение" }}</b><br />{{ j.ready ? VARIETIES[j.variety].name : `${Math.round(j.pct * 100)}%` }}</div>
          </div>
          <div v-for="n in free" :key="`f${n}`" class="g-slot" style="color: var(--text-2)"><i class="ph ph-circle-dashed" style="font-size: 26px" /><div>Свободно</div></div>
          <button v-if="b.slots < 2" class="g-slot add" @click="buySlot"><i class="ph ph-plus-circle" style="font-size: 26px" /><div>Второй слот<br /><span class="g-price" style="color: var(--text)"><span class="art-inline" style="width: 13px; height: 13px" v-html="ICON.coins" />{{ cfg.slotPrice }}</span></div></button>
        </div>
      </div>

      <div v-if="free > 0" class="stack" style="gap: 8px">
        <div style="font-size: 14px; font-weight: 600">{{ isPress ? "Что давим?" : "Что заливаем?" }}</div>
        <p v-if="!inputs.length" class="small muted">{{ isPress ? "На складе нет винограда — соберите урожай." : "Нет сусла — сначала отправьте виноград в давильню." }}</p>
        <button v-for="x in inputs" :key="x.v" class="g-option" :disabled="x.have < need" @click="start(x.v)">
          <span class="pic"><span class="art" v-html="art(inKind, x.v)" /></span>
          <span class="grow"><span class="t">{{ VARIETIES[x.v].name }}</span><span class="m" style="display: block">на складе {{ x.have }}{{ x.have < need ? ` — нужно ${need}` : "" }}</span></span>
          <span v-if="x.have >= need" class="g-mini">{{ isPress ? "Давить" : "Залить" }}</span>
        </button>
      </div>
    </template>
    <template v-if="b" #foot>
      <div class="stack" style="gap: 8px">
        <button v-if="anyReady" class="btn btn-primary" @click="collect"><i class="ph-bold ph-hand-grabbing" />Забрать</button>
        <button v-else-if="current" class="btn btn-primary" @click="skip">Готово сейчас<span class="g-btn-price"><span class="art-inline" v-html="ICON.grozdi" />{{ skipCost(current.readyAt, now) }}</span></button>
        <button v-if="upgradePrice" class="btn btn-outline" style="min-height: 48px; font-size: 15px" @click="upgrade">Улучшить: −10% времени<span class="g-price"><span class="art-inline" v-html="ICON.coins" />{{ upgradePrice }}</span></button>
      </div>
    </template>
  </BottomSheet>
</template>
