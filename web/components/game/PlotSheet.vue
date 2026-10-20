<script setup lang="ts">
// Шторка участка (1b): свободный — выбор саженца; растущий — сорт и таймер.
import { VARIETIES, plotStage, varietyAccess } from "~/utils/game.mjs";
import { grapeRes, plotArt } from "~/utils/game-art.mjs";
const props = defineProps<{ index: number | null }>();
const emit = defineEmits<{ close: [] }>();
const { state, now, speed, act } = useGame();
const { goal } = useMetrika();

const plot = computed(() => (props.index === null ? null : state.value?.plots[props.index]));
const stage = computed(() => (plot.value ? plotStage(plot.value, now.value, speed.value) : "empty"));
const STAGES: Record<string, string> = { sapling: "саженец", leaves: "листья", green: "зелёные грозди", ripe: "созрел", overripe: "перезрел", problem: "скворцы!" };
const pick = ref<string>("chardonnay");

const options = computed(() => {
  if (!state.value) return [];
  return Object.entries(VARIETIES).map(([key, v]) => {
    const access = varietyAccess(state.value, key);
    const saplings = state.value.saplings[key] || 0;
    return {
      key, name: v.name, art: grapeRes(v.col), open: access.open, lock: access.lock, saplings, price: v.price,
      scan: access.source === "scan" || access.source === "review",
      meta: `${v.color === "white" ? "Белое" : "Красное"} · ${fmtDur(v.grow, speed.value)} · урожай ${v.yld}`,
    };
  }).sort((a, b) => Number(b.open) - Number(a.open) || b.saplings - a.saplings);
});
watch(() => props.index, () => {
  const first = options.value.find((o) => o.open && o.saplings) || options.value.find((o) => o.open);
  if (first) pick.value = first.key;
}, { immediate: true });
const chosen = computed(() => options.value.find((o) => o.key === pick.value));

async function plant() {
  if (props.index === null || !chosen.value) return;
  const ok = await act({ type: "plant", plot: props.index, variety: pick.value });
  if (ok) { goal("game_plant", { variety: pick.value }); emit("close"); }
}
</script>

<template>
  <BottomSheet :open="index !== null && !!plot" :label="`Участок ${(index ?? 0) + 1}`" @close="emit('close')">
    <template v-if="plot">
      <div class="g-head"><h2>Участок {{ (index ?? 0) + 1 }}</h2><span class="small muted">{{ plot.variety ? STAGES[stage] : "свободен" }}</span></div>
      <template v-if="!plot.variety">
        <p class="muted small" style="margin-top: -12px">Что посадим?</p>
        <div class="stack" style="gap: 8px">
          <button v-for="o in options" :key="o.key" class="g-option" :class="{ on: pick === o.key && o.open }" :disabled="!o.open" @click="pick = o.key">
            <span class="pic"><span class="art" v-html="o.art" /></span>
            <span class="grow">
              <span class="t">{{ o.name }}<span v-if="o.scan" class="g-tag">ИЗ СКАНА</span></span>
              <span class="m" style="display: block">{{ o.meta }}</span>
            </span>
            <span v-if="o.open && o.saplings" class="small" style="font-weight: 700">саженцев: {{ o.saplings }}</span>
            <span v-else-if="o.open" class="g-price"><span class="art-inline" v-html="ICON.coins" />{{ o.price }}</span>
            <span v-else class="lock"><i class="ph-fill ph-lock-simple" style="font-size: 14px; flex: none" />{{ o.lock }}</span>
          </button>
          <NuxtLink to="/" class="row small muted" style="padding: 6px 4px; gap: 10px"><i class="ph ph-scan" style="font-size: 20px; color: var(--accent)" /><span class="grow">Автохтоны открываются сканом этикетки или отзывом</span><b style="color: var(--accent)">Как?</b></NuxtLink>
        </div>
      </template>
      <template v-else>
        <div class="row" style="gap: 16px">
          <div style="width: 120px; height: 106px; flex: none" v-html="plotArt(stage, VARIETIES[plot.variety].col)" />
          <div class="grow stack" style="gap: 4px">
            <b>{{ VARIETIES[plot.variety].name }}</b>
            <span class="small muted" v-if="now < plot.readyAt">Созреет через {{ fmtLeft(plot.readyAt - now) }}</span>
            <span class="small muted" v-else>Урожай готов — нажмите на участок</span>
            <span class="small muted">Урожай: {{ VARIETIES[plot.variety].yld }} гроздей</span>
          </div>
        </div>
        <p class="small muted">Спелый виноград ждёт двое суток, потом перезревает и даёт вдвое меньше. Скворцов прогоняйте сразу — иначе съедят пятую часть урожая.</p>
      </template>
    </template>
    <template v-if="plot && !plot.variety" #foot>
      <button class="btn btn-primary" :disabled="!chosen?.open" @click="plant">
        Посадить {{ chosen?.name }}
        <span v-if="chosen && !chosen.saplings" class="g-btn-price"><span class="art-inline" v-html="ICON.coins" />{{ chosen.price }}</span>
        <span v-else class="g-btn-price" style="padding: 3px 10px">саженец</span>
      </button>
    </template>
  </BottomSheet>
</template>
