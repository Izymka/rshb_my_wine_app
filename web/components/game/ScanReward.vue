<script setup lang="ts">
// Награда в винодельне после распознанного скана (1k): новый сорт и саженец, Грозди, счётчик
// наград за сутки; при исчерпанном лимите — спокойная «Задумчивая» Виноградинка (1m). Видна
// только вошедшим взрослым с включённой игрой; сбой игры не мешает карточке вина.
import { VARIETIES } from "~/utils/game.mjs";
import { mascotArt, saplingArt } from "~/utils/game-art.mjs";
const props = defineProps<{ slug: string | null }>();
const { user } = useAuth();
const { enabledLocal, readEnabled } = useGame();
const { goal } = useMetrika();
const claimed = useState<string[]>("game.scanClaimed", () => []);

interface Reward { status: "granted" | "repeat" | "limit" | "disabled"; used: number; limit: number; name: string; reward: null | { gained: Record<string, number>; unlocked: string[]; saplings: Record<string, number> } }
const res = ref<Reward | null>(null);
const open = ref(false);

async function claim(slug: string) {
  readEnabled();
  if (!user.value?.adult || !enabledLocal.value || claimed.value.includes(slug)) return;
  claimed.value = [...claimed.value, slug];
  try {
    res.value = await $fetch<Reward>("/api/game/reward", { method: "POST", body: { kind: "scan", slug } });
    if (res.value.status === "granted" || res.value.status === "limit") {
      open.value = true;
      goal("game_scan_reward", { status: res.value.status, unlocked: res.value.reward?.unlocked.length || 0 });
    }
  } catch {}
}
onMounted(() => { if (props.slug) setTimeout(() => claim(props.slug!), 900); });
watch(() => props.slug, (s) => { if (s) void claim(s); });

const sapling = computed(() => {
  const v = Object.keys(res.value?.reward?.saplings || {})[0];
  return v ? { v, name: VARIETIES[v].name, art: saplingArt(VARIETIES[v].color === "white" ? "#c8c46a" : "#86b05e") } : null;
});
const title = computed(() => {
  const u = res.value?.reward?.unlocked || [];
  return u.length ? `Новый сорт — ${u.map((v) => VARIETIES[v].name).join(" и ")}!` : sapling.value ? `Саженец ${sapling.value.name}` : "Грозди для винодельни";
});
</script>

<template>
  <BottomSheet :open="open && !!res" label="Награда в винодельне" @close="open = false">
    <template v-if="res?.status === 'granted' && res.reward">
      <div class="row" style="gap: 12px">
        <div style="width: 64px; height: 70px; flex: none; animation: svBob 3.2s ease-in-out infinite" v-html="mascotArt({ mood: 'joy', apron: true })" />
        <div class="grow"><div style="font-size: 12px; font-weight: 700; color: var(--accent)">НАГРАДА В ВИНОДЕЛЬНЕ</div><div class="serif" style="font-size: 22px; line-height: 1.2">{{ title }}</div></div>
      </div>
      <div class="grid-2">
        <div v-if="sapling" class="row" style="background: var(--bg-cream); border-radius: 18px; padding: 12px; gap: 10px"><span style="width: 40px; height: 40px; flex: none" v-html="sapling.art" /><div class="small" style="line-height: 1.3"><b>Саженец</b><br />{{ sapling.name }} ×1</div></div>
        <div v-if="res.reward.gained.grozdi" class="row" style="background: var(--bg-cream); border-radius: 18px; padding: 12px; gap: 10px"><span style="width: 36px; height: 36px; flex: none" v-html="ICON.grozdi" /><div class="small" style="line-height: 1.3"><b>+{{ res.reward.gained.grozdi }}</b><br />Гроздей</div></div>
      </div>
      <div class="small muted row" style="gap: 6px; font-size: 12px"><i class="ph ph-gift" />Наград за сканы сегодня: {{ res.used }} из {{ res.limit }}</div>
    </template>
    <template v-else-if="res?.status === 'limit'">
      <div class="row" style="background: var(--bg-cream); border-radius: 16px; padding: 10px 12px; gap: 10px">
        <span style="width: 44px; height: 48px; flex: none" v-html="mascotArt({ mood: 'meh', still: true })" />
        <div class="grow small" style="line-height: 1.35"><b>На сегодня наград за сканы хватит</b><br /><span class="muted">Вино можно сохранить на полку. Новые награды — завтра</span></div>
      </div>
    </template>
    <template #foot>
      <div class="row" style="gap: 8px">
        <button class="btn btn-outline" style="flex: 1; min-height: 52px" @click="open = false">К вину</button>
        <NuxtLink v-if="res?.status === 'granted'" to="/winery" class="btn btn-primary" style="flex: 1.4; min-height: 52px" @click="goal('game_scan_reward_plant')"><i class="ph-bold ph-plant" />Посадить</NuxtLink>
      </div>
    </template>
  </BottomSheet>
</template>
