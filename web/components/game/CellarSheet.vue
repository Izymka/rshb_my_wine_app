<script setup lang="ts">
// Погреб: бочки выдержки. Молодое вино → выдержанное за 1 / 3 / 7 дней (1 / 2 / 3 бутылки).
import { BALANCE, VARIETIES, skipCost } from "~/utils/game.mjs";
import { barrelArt, youngArt } from "~/utils/game-art.mjs";
const props = defineProps<{ open: boolean }>();
const emit = defineEmits<{ close: [] }>();
const { state, now, speed, act } = useGame();
const days = ref(3);
const choosing = ref<number | null>(null);

const barrels = computed(() => (state.value?.barrels || []).map((b: any, i: number) => {
  const st = !b.variety ? "empty" : b.readyAt <= now.value ? "ready" : "aging";
  const left = b.readyAt - now.value;
  return { ...b, i, st, left, label: st === "aging" ? fmtLeft(left).replace(/ \d+ ч$/, "") : "" };
}));
const young = computed(() => (state.value ? Object.keys(VARIETIES).map((v) => ({ v, have: state.value.items[`young:${v}`] || 0 })).filter((x) => x.have) : []));
const youngCol = (v: string) => (VARIETIES[v].color === "white" ? "#d9cf7a" : "#5b1f35");

async function age(v: string) {
  if (choosing.value === null) return;
  if (await act({ type: "age", barrel: choosing.value, variety: v, days: days.value })) choosing.value = null;
}
</script>

<template>
  <BottomSheet :open="open" label="Погреб" @close="choosing = null; emit('close')">
    <div class="g-head"><h2>Погреб</h2><span class="small muted">бочек: {{ barrels.length }} из {{ BALANCE.barrel.max }}</span></div>
    <p class="small muted" style="margin-top: -10px">Чем дольше выдержка, тем больше выдержанного вина из одной бочки: 1 день — 1, 3 дня — 2, 7 дней — 3.</p>
    <div class="stack" style="gap: 8px">
      <div v-for="b in barrels" :key="b.i" class="g-option" style="cursor: default" :class="{ on: choosing === b.i }">
        <span style="width: 70px; height: 56px; flex: none" v-html="barrelArt(b.st, b.label)" />
        <span class="grow">
          <span class="t">Бочка {{ b.i + 1 }}</span>
          <span class="m" style="display: block">
            <template v-if="b.st === 'empty'">пустая</template>
            <template v-else-if="b.st === 'aging'">{{ VARIETIES[b.variety].name }} · {{ fmtDur(b.days * 86400, 1) }} · осталось {{ fmtLeft(b.left) }}</template>
            <template v-else>{{ VARIETIES[b.variety].name }} готово · {{ BALANCE.barrel.days[b.days] }} бут.</template>
          </span>
        </span>
        <button v-if="b.st === 'empty'" class="g-mini outline" @click="choosing = choosing === b.i ? null : b.i">Заложить</button>
        <button v-else-if="b.st === 'ready'" class="g-mini" @click="act({ type: 'uncork', barrel: b.i })">Откупорить</button>
        <button v-else class="g-mini outline" @click="act({ type: 'skip', building: 'barrel', index: b.i })"><span class="g-price"><span class="art-inline" v-html="ICON.grozdi" />{{ skipCost(b.readyAt, now) }}</span></button>
      </div>
    </div>
    <div v-if="choosing !== null" class="stack" style="gap: 10px">
      <div class="segment">
        <button v-for="d in [1, 3, 7]" :key="d" :class="{ on: days === d }" @click="days = d">{{ fmtDur(d * 86400, speed) }} · {{ BALANCE.barrel.days[d] }} бут.</button>
      </div>
      <p v-if="!young.length" class="small muted">Нет молодого вина — сделайте его в бродильном чане.</p>
      <button v-for="x in young" :key="x.v" class="g-option" @click="age(x.v)">
        <span class="pic"><span class="art" v-html="youngArt(youngCol(x.v))" /></span>
        <span class="grow"><span class="t">Молодое {{ VARIETIES[x.v].name }}</span><span class="m" style="display: block">на складе {{ x.have }}</span></span>
        <span class="g-mini">В бочку</span>
      </button>
    </div>
    <NuxtLink v-if="barrels.length < BALANCE.barrel.max" to="/winery/shop?tab=buildings" class="row small" style="gap: 8px; font-weight: 600"><i class="ph ph-plus-circle" style="font-size: 20px" />Ещё бочка — в магазине</NuxtLink>
  </BottomSheet>
</template>
