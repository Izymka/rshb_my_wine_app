<script setup lang="ts">
// Магазин (1i): здания, саженцы, участки, декор. Грозди здесь только тратятся — купить их за
// деньги нельзя, они приходят за действия в сервисе и Винордл.
import { DECOR, VARIETIES, shopCatalog } from "~/utils/game.mjs";
import { barrelArt, benchArt, bottlingArt, flowerBarrelArt, lanternArt, plotArt, pressArt, saplingArt, scarecrowArt, tankArt } from "~/utils/game-art.mjs";
const game = useGame();
const { state, status, speed, act } = game;
const route = useRoute();
const { goal } = useMetrika();
useHead({ title: "Магазин — Моя винодельня" });
onMounted(() => { if (!state.value) void game.load(); });
watch(status, (s) => { if (s === "guest" || s === "adult") navigateTo("/winery"); });

const TABS = [["buildings", "Здания"], ["saplings", "Саженцы"], ["plots", "Участки"], ["decor", "Декор"]] as const;
const tab = ref<string>(TABS.some(([k]) => k === route.query.tab) ? String(route.query.tab) : "buildings");
const BUILD: Record<string, { name: string; desc: string; art: () => string; isNew?: boolean }> = {
  barrel: { name: "Ещё бочка", desc: "Несколько выдержек одновременно", art: () => barrelArt("empty"), isNew: true },
  "slot:press": { name: "Слот давильни", desc: "Давить два сорта сразу", art: () => pressArt(false) },
  "slot:tank": { name: "Второй чан", desc: "Два брожения одновременно", art: () => tankArt(false) },
  bottling: { name: "Линия розлива", desc: "1 вино → 6 бутылок за 1 ч", art: bottlingArt },
  blending: { name: "Купажный стол", desc: "Купажи по рецептам сомелье", art: () => tankArt(false) },
};
const DECOR_ART: Record<string, () => string> = { lantern: lanternArt, bench: benchArt, flowers: flowerBarrelArt, scarecrow: scarecrowArt };

const items = computed(() => {
  if (!state.value) return [];
  const cat = shopCatalog(state.value);
  if (tab.value === "saplings") return cat.saplings.map((i: any) => {
    const v = VARIETIES[i.variety];
    return { ...i, name: `Саженец · ${v.name}`, desc: `${v.color === "white" ? "Белое" : "Красное"}, растёт ${fmtDur(v.grow, speed.value)}`, art: saplingArt(v.color === "white" ? "#c8c46a" : "#86b05e"), lock: i.lock ? (v.scanOnly ? "Скан или отзыв" : i.lock) : null, have: state.value.saplings[i.variety] || 0 };
  });
  if (tab.value === "plots") return cat.plots.map((i: any) => ({ ...i, name: `Участок № ${i.plot}`, desc: i.plot === 7 ? "Рядом с погребом" : "У пруда", art: plotArt("empty") }));
  if (tab.value === "decor") return cat.decor.map((i: any) => ({ ...i, name: DECOR[i.decor].name, desc: DECOR[i.decor].desc, isNew: DECOR[i.decor].isNew, art: DECOR_ART[i.decor]() }));
  return cat.buildings.map((i: any) => ({ ...i, ...BUILD[i.id], art: BUILD[i.id].art() }));
});
async function buy(id: string) {
  if (await act({ type: "buy", item: id })) { goal("game_buy", { item: id }); game.toast("Куплено"); }
}
</script>

<template>
  <main class="g-page">
    <GamePageTop title="Магазин">
      <template v-if="state">
        <span class="g-cur" style="font-size: 13px"><span class="art-inline" style="width: 18px; height: 18px" v-html="ICON.coins" />{{ fmtNum(state.coins) }}</span>
        <span class="g-cur" style="font-size: 13px"><span class="art-inline" style="width: 18px; height: 18px" v-html="ICON.grozdi" />{{ state.grozdi }}</span>
      </template>
    </GamePageTop>
    <div class="segment" role="tablist">
      <button v-for="[k, label] in TABS" :key="k" role="tab" :aria-selected="tab === k" :class="{ on: tab === k }" @click="tab = k">{{ label }}</button>
    </div>
    <p v-if="!state" class="muted">Загружаем магазин…</p>
    <div v-else class="grid-2" style="gap: 10px; align-content: start">
      <div v-for="it in items" :key="it.id" class="g-card" style="padding: 12px; gap: 8px" :style="{ opacity: it.lock ? 0.7 : 1 }">
        <div style="height: 96px; border-radius: 16px; background: var(--bg-cream); display: flex; align-items: center; justify-content: center; position: relative">
          <span style="width: 84px; height: 84px" v-html="it.art" />
          <span v-if="it.isNew && !it.lock" style="position: absolute; top: 6px; left: 6px; font-size: 10px; font-weight: 700; background: var(--accent); color: #fff; padding: 2px 6px; border-radius: 999px">НОВОЕ</span>
          <span v-if="it.have" style="position: absolute; top: 6px; right: 8px; font-size: 11px; font-weight: 700; color: var(--accent)">×{{ it.have }}</span>
        </div>
        <div style="font-size: 14px; font-weight: 600; line-height: 1.25">{{ it.name }}</div>
        <div class="small muted" style="font-size: 12px; line-height: 1.3; min-height: 31px">{{ it.desc }}</div>
        <button v-if="!it.lock" class="btn btn-sand" style="min-height: 38px; font-size: 14px; font-weight: 700; gap: 5px" @click="buy(it.id)"><span class="art-inline" style="width: 18px; height: 18px" v-html="it.cur === 'grozdi' ? ICON.grozdi : ICON.coins" />{{ it.price }}</button>
        <div v-else style="height: 38px; border-radius: 999px; background: var(--seg); color: var(--text-2); font-size: 13px; font-weight: 600; display: flex; align-items: center; justify-content: center; gap: 5px"><i class="ph-fill ph-lock-simple" />{{ it.lock }}</div>
      </div>
    </div>
    <p v-if="tab === 'saplings'" class="small muted" style="padding: 0 4px">Автохтонные сорта не продаются: их саженец приходит за скан этикетки или отзыв о вине из этого сорта.</p>
    <GameToasts />
  </main>
</template>
