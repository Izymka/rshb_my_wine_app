<script setup lang="ts">
// Склад (1d): виноград, полуфабрикаты, вино и саженцы. Тап по продукту — продажа за монеты.
import { BALANCE, VARIETIES, itemName, itemValue, shopCatalog, storageUsed } from "~/utils/game.mjs";
import { agedArt, grapeRes, mustArt, saplingArt, youngArt } from "~/utils/game-art.mjs";
const game = useGame();
const { state, status, act } = game;
useHead({ title: "Склад — Моя винодельня" });
onMounted(() => { if (!state.value) void game.load(); });
watch(status, (s) => { if (s === "guest" || s === "adult") navigateTo("/winery"); });

const youngCol = (v: string) => (VARIETIES[v].color === "white" ? "#d9cf7a" : "#5b1f35");
function art(item: string) {
  const [kind, v] = item.split(":");
  if (kind === "grape") return grapeRes(VARIETIES[v].col);
  if (kind === "must") return mustArt();
  if (kind === "young") return youngArt(youngCol(v));
  return agedArt();
}
const used = computed(() => (state.value ? storageUsed(state.value) : 0));
const sections = computed(() => {
  if (!state.value) return [];
  const items = Object.entries(state.value.items as Record<string, number>);
  const pick = (kinds: string[]) => items.filter(([k]) => kinds.includes(k.split(":")[0]))
    .map(([item, qty]) => ({ item, qty, name: itemName(item).replace(/^Виноград /, ""), art: art(item), scan: false }));
  const saplings = Object.entries(state.value.saplings as Record<string, number>).map(([v, qty]) => ({
    item: `sapling:${v}`, qty, name: VARIETIES[v].name, art: saplingArt(VARIETIES[v].color === "white" ? "#c8c46a" : "#86b05e"),
    scan: ["scan", "review"].includes(state.value.unlocked[v]),
  }));
  return [
    { title: "Виноград", items: pick(["grape"]) },
    { title: "В работе", items: pick(["must", "young"]) },
    { title: "Вино", items: pick(["aged"]) },
    { title: "Саженцы", items: saplings },
  ].filter((s) => s.items.length);
});
const expand = computed(() => (state.value ? shopCatalog(state.value).storage : null));

const selling = ref<string | null>(null);
const count = ref(1);
const sel = computed(() => (selling.value && state.value ? { item: selling.value, have: state.value.items[selling.value] || 0, price: itemValue(selling.value), name: itemName(selling.value) } : null));
function open(item: string) {
  if (item.startsWith("sapling:")) { navigateTo("/winery"); return; }
  selling.value = item;
  count.value = 1;
}
async function sell() {
  if (!sel.value) return;
  if (await act({ type: "sell", item: sel.value.item, count: count.value })) selling.value = null;
}
</script>

<template>
  <main class="g-page">
    <GamePageTop title="Склад" />
    <p v-if="!state" class="muted">Загружаем склад…</p>
    <template v-else>
      <div class="stack" style="gap: 6px; padding: 0 4px">
        <div class="row small" style="justify-content: space-between"><b>Заполнено {{ used }} из {{ state.storage }}</b><span class="muted">{{ Math.round((used / state.storage) * 100) }}%</span></div>
        <div class="g-bar"><div :style="{ width: `${Math.min(100, (used / state.storage) * 100)}%`, background: used >= state.storage ? 'var(--accent)' : undefined }" /></div>
      </div>
      <p v-if="!sections.length" class="g-empty">Склад пуст. Соберите урожай на поле.</p>
      <section v-for="sec in sections" :key="sec.title" class="stack" style="gap: 8px">
        <div style="font-size: 14px; font-weight: 600; padding-top: 6px">{{ sec.title }}</div>
        <div class="g-grid4">
          <button v-for="it in sec.items" :key="it.item" class="g-item" @click="open(it.item)">
            <span class="pic" v-html="it.art" />
            <span class="n">{{ it.name }}</span>
            <span class="q">×{{ it.qty }}</span>
            <span v-if="it.scan" class="s"><i class="ph-bold ph-scan" /></span>
          </button>
        </div>
      </section>
      <p class="small muted" style="padding: 0 4px">Нажмите на продукт, чтобы продать. Заказы платят больше, чем продажа со склада.</p>
      <div style="flex: 1" />
      <button v-if="expand?.price" class="btn btn-outline" @click="act({ type: 'buy', item: 'storage' })">
        Расширить до {{ expand.next }} мест<span class="g-price"><span class="art-inline" v-html="ICON.coins" />{{ expand.price }}</span>
      </button>
      <p v-else class="small muted" style="text-align: center">Склад расширен до максимума — {{ BALANCE.storage.start + BALANCE.storage.step * BALANCE.storage.prices.length }} мест</p>
    </template>

    <BottomSheet :open="!!sel" :label="sel?.name || 'Продажа'" @close="selling = null">
      <template v-if="sel">
        <div class="row" style="gap: 14px">
          <span style="width: 56px; height: 56px; flex: none" v-html="art(sel.item)" />
          <div class="grow"><div class="serif" style="font-size: 22px; line-height: 1.2">{{ sel.name }}</div><div class="small muted">на складе {{ sel.have }} · {{ sel.price }} монет за штуку</div></div>
        </div>
        <div class="row" style="justify-content: center; gap: 18px">
          <button class="icon-btn" aria-label="Меньше" :disabled="count <= 1" @click="count--"><i class="ph ph-minus" /></button>
          <span class="serif" style="font-size: 32px; min-width: 48px; text-align: center">{{ count }}</span>
          <button class="icon-btn" aria-label="Больше" :disabled="count >= sel.have" @click="count++"><i class="ph ph-plus" /></button>
          <button class="chip" @click="count = sel.have">все</button>
        </div>
      </template>
      <template v-if="sel" #foot>
        <button class="btn btn-primary" @click="sell">Продать<span class="g-btn-price"><span class="art-inline" v-html="ICON.coins" />{{ sel.price * count }}</span></button>
      </template>
    </BottomSheet>
    <GameToasts />
  </main>
</template>
