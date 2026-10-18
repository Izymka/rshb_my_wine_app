<script setup lang="ts">
// Мои вина (1e): полка-иллюстрация, напоминание оценить, поиск, фильтры со счётчиками,
// превью с оценкой, блюдами и заметкой. Полка живёт в браузере и работает без сети;
// после входа она ещё и синхронизируется с аккаунтом.
import { STATUS, REACTIONS, shelfDate, winePath, dueWines } from "~/utils/shelf.mjs";
import { DISH_ICONS, plural } from "~/utils/wine.mjs";
const { entries, ready, error } = useShelf();
const { user } = useAuth();
const filter = ref<"all" | keyof typeof STATUS>("all");
const search = ref("");
const now = ref(0);
onMounted(() => { now.value = Date.now(); });
const shown = computed(() => entries.value.filter((e) => (filter.value === "all" || e.status === filter.value)
  && `${e.card.name} ${e.card.winery} ${e.note}`.toLowerCase().includes(search.value.trim().toLowerCase())));
const due = computed(() => (now.value ? dueWines(entries.value, now.value) : []));
const liked = computed(() => entries.value.filter((e) => e.reaction === "liked").length);
const filters = computed(() => [
  { key: "all" as const, label: "Все", count: entries.value.length },
  ...Object.entries(STATUS).map(([key, label]) => ({ key: key as keyof typeof STATUS, label, count: entries.value.filter((e) => e.status === key).length })),
]);
useHead({ title: "Мои вина — Своё Вино" });
</script>

<template>
  <main class="page with-tabs">
    <header class="topbar">
      <div class="grow">
        <h1 class="page-title">Мои вина</h1>
        <div v-if="entries.length" class="small muted" style="margin-top: 2px">
          {{ entries.length }} {{ plural(entries.length, ["вино", "вина", "вин"]) }}<template v-if="liked"> · {{ liked }} {{ plural(liked, ["понравилось", "понравились", "понравились"]) }}</template>
        </div>
      </div>
      <NuxtLink class="icon-btn accent" to="/" aria-label="Сканировать"><i class="ph-bold ph-scan" /></NuxtLink>
    </header>

    <p v-if="error" class="error" role="alert">{{ error }}</p>
    <p v-if="!ready" role="status" class="muted">Загружаем полку…</p>
    <template v-else>
      <div class="shelf-art" aria-hidden="true">
        <WineBottle v-for="e in entries.slice(0, 5)" :key="e.slug" :slug="e.slug" :card="e.card" :height="100" />
        <div class="grow" />
        <Mascot :size="52" :mood="entries.length ? 'happy' : 'wow'" />
      </div>

      <NuxtLink v-if="due.length" class="remind-card" :to="`${winePath(due[0].slug)}#review`" style="color: var(--text)">
        <i class="ph-fill ph-bell-ringing" />
        <div class="grow">
          <div style="font-size: 15px; font-weight: 600">Уже попробовали {{ due[0].card.name }}?</div>
          <div class="small muted" style="margin-top: 2px">Сохраните впечатление, пока свежо<template v-if="due.length > 1"> · ещё {{ due.length - 1 }}</template></div>
        </div>
        <span class="go">Оценить</span>
      </NuxtLink>

      <section v-if="!entries.length" class="card cream stack" style="align-items: flex-start">
        <h2 class="serif-title">Здесь будут ваши вина</h2>
        <p class="muted">Отсканируйте этикетку и сохраните вино. Его название, ваша оценка и заметка останутся под рукой.</p>
        <NuxtLink class="btn btn-primary" to="/"><i class="ph-bold ph-camera" />Сканировать первое вино</NuxtLink>
      </section>

      <template v-else>
        <div class="stack">
          <label class="search">
            <i class="ph ph-magnifying-glass" />
            <input v-model="search" type="search" placeholder="Название, винодельня или заметка" aria-label="Найти на полке" />
          </label>
          <div class="chips scroll" role="group" aria-label="Фильтр">
            <button v-for="f in filters" :key="f.key" class="chip" :class="{ on: filter === f.key }" :aria-pressed="filter === f.key" @click="filter = f.key">
              {{ f.label }}<span class="count">{{ f.count }}</span>
            </button>
          </div>
        </div>

        <div class="stack">
          <NuxtLink v-for="e in shown" :key="e.slug" :to="winePath(e.slug)" class="wine-row">
            <div class="thumb"><WineBottle :slug="e.slug" :card="e.card" :height="76" /></div>
            <div class="grow stack" style="gap: 3px">
              <div class="row" style="gap: 6px"><span class="st" :class="e.status">{{ STATUS[e.status] }}</span><span style="font-size: 11px; color: var(--text-3)">{{ shelfDate(e.addedAt) }}</span></div>
              <div class="name">{{ e.card.name }}</div>
              <div class="sub">{{ e.card.winery }}<template v-if="e.card.category"> · {{ e.card.category }}</template></div>
              <div v-if="e.reaction || e.dishes.length" class="row" style="gap: 8px; margin-top: 2px">
                <span v-if="e.reaction" class="row" style="gap: 4px; font-size: 12px; font-weight: 600" :style="{ color: e.reaction === 'liked' ? 'var(--ok)' : 'var(--accent)' }">
                  <i :class="['ph-fill', e.reaction === 'liked' ? 'ph-thumbs-up' : 'ph-thumbs-down']" style="font-size: 14px" />{{ REACTIONS[e.reaction] }}
                </span>
                <i v-for="d in e.dishes" :key="d" :class="['ph', DISH_ICONS[d]]" :title="d" style="font-size: 16px; color: var(--accent)" />
              </div>
              <div v-if="e.note" class="note">«{{ e.note }}»</div>
            </div>
            <i class="ph ph-caret-right caret" />
          </NuxtLink>
          <p v-if="!shown.length" class="muted small" style="text-align: center; padding: 24px 12px">Таких вин на полке пока нет</p>
          <div class="device-note">
            <i :class="['ph', user ? 'ph-cloud-check' : 'ph-device-mobile']" style="font-size: 15px" />
            {{ user ? "Полка сохранена в аккаунте и работает без сети" : "Полка хранится на этом устройстве и работает без сети" }}
          </div>
        </div>
      </template>
    </template>
  </main>
</template>
