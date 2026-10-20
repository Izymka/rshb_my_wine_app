<script setup lang="ts">
// Нижняя навигация: «Сканер / Мои вина / Винодельня / Профиль». Бейдж — вина, которые пора оценить.
import { dueWines } from "~/utils/shelf.mjs";
const route = useRoute();
const { entries } = useShelf();
const { busy } = useScan();
const found = useState("scan.found", () => false);
const now = ref(0);
const { goal } = useMetrika();
onMounted(() => { now.value = Date.now(); });
const due = computed(() => (now.value ? dueWines(entries.value, now.value).length : 0));
const visible = computed(() => ["/", "/wines", "/winery", "/profile"].includes(route.path) && !busy.value && !found.value);
// Вкладка игры скрыта, если игрок выключил «Мою винодельню» в настройках.
const { enabledLocal, readEnabled } = useGame();
onMounted(readEnabled);
const tabs = computed(() => [
  { to: "/", label: "Сканер", icon: "ph-scan" },
  { to: "/wines", label: "Мои вина", icon: "ph-books" },
  ...(enabledLocal.value ? [{ to: "/winery", label: "Винодельня", icon: "ph-plant" }] : []),
  { to: "/profile", label: "Профиль", icon: "ph-user-circle" },
]);
</script>

<template>
  <div v-if="visible" class="tabbar">
    <nav aria-label="Разделы">
      <NuxtLink v-for="t in tabs" :key="t.to" :to="t.to" :class="{ on: route.path === t.to }" :aria-current="route.path === t.to ? 'page' : undefined" @click="goal('tab_click', { tab: t.to, from: route.path, due })">
        <i :class="[route.path === t.to ? 'ph-fill' : 'ph', t.icon]" />{{ t.label }}
        <span v-if="t.to === '/wines' && due" class="count" :aria-label="`ждут оценки: ${due}`">{{ due }}</span>
      </NuxtLink>
    </nav>
  </div>
</template>
