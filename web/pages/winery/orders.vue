<script setup lang="ts">
// Доска заказов (1e): гости, рестораны и праздники просят продукцию. Заказ с полки — вино, которое
// игрок отметил «Хочу попробовать»: награда ×1,5, а если сорт закрыт — подсказка открыть сканом.
import { BALANCE, VARIETIES, itemName, levelOf, parseItem, varietyAccess } from "~/utils/game.mjs";
import { agedArt, grapeRes, mustArt, youngArt } from "~/utils/game-art.mjs";
import { winePath } from "~/utils/shelf.mjs";
const game = useGame();
const { state, status, speed, act } = game;
const now = useGameClock();
const { goal } = useMetrika();
useHead({ title: "Заказы — Моя винодельня" });
onMounted(() => { void game.load(); });
watch(status, (s) => { if (s === "guest" || s === "adult") navigateTo("/winery"); });

function art(item: string) {
  const { kind, variety } = parseItem(item);
  if (kind === "grape") return grapeRes(VARIETIES[variety].col);
  if (kind === "must") return mustArt();
  if (kind === "young") return youngArt(VARIETIES[variety].color === "white" ? "#d9cf7a" : "#5b2342");
  return agedArt();
}
const orders = computed(() => {
  const s = state.value;
  if (!s) return [];
  return s.orders.slots.map((slot: any, i: number) => {
    const o = slot.order;
    if (!o) return { i, empty: true, left: Math.max(0, slot.cooldownUntil - now.value), done: Boolean(slot.doneAt) };
    const have = s.items[o.item] || 0;
    const variety = parseItem(o.item).variety;
    const locked = !varietyAccess(s, variety).open;
    return { i, ...o, have, ok: have >= o.count && !locked, locked, art: art(o.item), need: itemName(o.item), left: fmtLeft(o.expiresAt - now.value) };
  });
});
const lockedSlots = computed(() => (state.value && levelOf(state.value.xp) < BALANCE.orders.extraLevel ? BALANCE.orders.extra : 0));
async function send(i: number) {
  const o = orders.value[i];
  if (await act({ type: "fulfill", slot: i })) goal("game_order", { shelf: Boolean(o.shelf), item: o.item });
}
</script>

<template>
  <main class="g-page">
    <GamePageTop title="Заказы">
      <span v-if="state" class="g-cur"><span class="art-inline" v-html="ICON.coins" />{{ fmtNum(state.coins) }}</span>
    </GamePageTop>
    <p class="small muted" style="padding: 0 4px">Выполненный заказ сменится новым через {{ fmtDur(BALANCE.orders.cooldown, speed) }}</p>
    <p v-if="!state" class="muted">Загружаем заказы…</p>
    <div v-else class="stack" style="gap: 10px">
      <template v-for="o in orders" :key="o.i">
        <div v-if="o.empty" class="g-card" style="background: #f4f8f2; border-color: #cfe0c9">
          <div class="g-ok" style="background: none"><i class="ph-bold ph-check" />{{ o.done ? "Отправлено" : "Ищем заказчика" }} · новый через {{ fmtLeft(o.left) }}</div>
        </div>
        <div v-else class="g-card">
          <div class="row" style="gap: 10px">
            <span class="g-ico"><i :class="['ph', o.icon]" /></span>
            <div class="grow"><div style="font-size: 15px; font-weight: 600; overflow-wrap: anywhere">{{ o.who }}</div><div class="small muted row" style="gap: 4px; font-size: 12px"><i class="ph ph-clock" />осталось {{ o.left }}</div></div>
            <span v-if="o.mult" class="badge" style="background: var(--sand); color: var(--sand-ink)">награда ×{{ String(o.mult).replace(".", ",") }}</span>
          </div>
          <NuxtLink v-if="o.shelf" :to="winePath(o.shelf.slug)" class="row" style="gap: 8px; background: var(--bg-cream); border-radius: 12px; padding: 8px 10px; font-size: 12px; color: var(--text)">
            <i class="ph-fill ph-books" style="color: var(--accent); font-size: 16px" /><span class="grow"><b>Из вашей полки:</b> {{ o.shelf.name }}<template v-if="o.shelf.winery"> · {{ o.shelf.winery }}</template></span><i class="ph ph-arrow-up-right" style="color: var(--accent)" />
          </NuxtLink>
          <div class="row" style="gap: 10px">
            <span style="width: 40px; height: 40px; flex: none" v-html="o.art" />
            <div class="grow small">{{ o.need }}<div style="font-weight: 700" :style="{ color: o.have >= o.count ? 'var(--ok)' : 'var(--text-2)' }">есть {{ o.have }} из {{ o.count }}</div></div>
            <div class="stack" style="gap: 3px; align-items: flex-end; font-size: 13px">
              <span class="g-price"><span class="art-inline" v-html="ICON.coins" />{{ o.coins }}</span>
              <span v-if="o.grozdi" class="g-price muted"><span class="art-inline" v-html="ICON.grozdi" />+{{ o.grozdi }}</span>
              <span v-else-if="o.xp" class="g-price muted"><span class="art-inline" v-html="ICON.xp" />+{{ o.xp }}</span>
            </div>
          </div>
          <button v-if="o.ok" class="btn btn-primary" style="min-height: 44px; font-size: 15px" @click="send(o.i)">Отправить заказ</button>
          <NuxtLink v-else-if="o.locked" to="/" class="btn btn-accent-outline" style="min-height: 44px"><i class="ph-bold ph-scan" />Открыть сорт сканом</NuxtLink>
          <div v-else class="small muted row" style="gap: 6px; font-size: 12px"><i class="ph ph-info" />Не хватает {{ o.count - o.have }} — {{ o.item.startsWith("grape") ? "соберите урожай" : o.item.startsWith("must") ? "выжмите в давильне" : o.item.startsWith("young") ? "сделайте в чане" : "выдержите в погребе" }}</div>
        </div>
      </template>
      <div v-if="lockedSlots" class="g-empty"><i class="ph ph-hourglass-medium" style="font-size: 18px" />Ещё {{ lockedSlots }} заказа откроются на {{ BALANCE.orders.extraLevel }} уровне</div>
    </div>
    <GameToasts />
  </main>
</template>
