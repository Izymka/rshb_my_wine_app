<script setup lang="ts">
import { STATUS, REACTIONS, shelfDate, winePath } from '~/utils/shelf.mjs';
const { entries, ready, error } = useShelf();
const filter = ref('all');
const search = ref('');
const now = ref(Date.now());
onMounted(() => { now.value = Date.now(); });
const shown = computed(() => entries.value.filter(e => (filter.value === 'all' || e.status === filter.value)
  && `${e.card.name} ${e.card.winery} ${e.note}`.toLowerCase().includes(search.value.toLowerCase())));
const due = computed(() => entries.value.filter(e => !e.reaction && e.remindAt !== null && e.remindAt <= now.value));
useHead({ title: 'Мои вина — Своё Вино' });
</script>
<template>
  <main class="page">
    <header class="topbar"><img src="/icon.png" alt="" /><div class="grow"><h1 class="brand">Мои вина</h1><div class="sub">Ваша личная винная полка</div></div><NuxtLink to="/">Сканировать</NuxtLink></header>
    <p class="small muted">Полка хранится в этом браузере на этом устройстве. Карточки и заметки доступны без сети.</p>
    <p v-if="error" class="error" role="alert">{{ error }}</p>
    <p v-if="!ready" role="status">Загружаем полку…</p>
    <template v-else>
      <section v-if="due.length" class="card cream">
        <h2>Уже попробовали?</h2><p>Сохраните впечатление, чтобы вспомнить его при следующем выборе.</p>
        <NuxtLink v-for="e in due" :key="e.slug" class="item" :to="`${winePath(e.slug)}#review`">{{ e.card.name }} · Оценить →</NuxtLink>
      </section>
      <div v-if="entries.length" class="shelf-tools">
        <label for="wine-search" class="small">Найти на полке</label><input id="wine-search" v-model="search" type="search" placeholder="Название, винодельня или заметка" />
        <div class="vintage"><button class="chip" :class="{ on: filter === 'all' }" :aria-pressed="filter === 'all'" @click="filter = 'all'">Все · {{ entries.length }}</button><button v-for="(label, value) in STATUS" :key="value" class="chip" :class="{ on: filter === value }" :aria-pressed="filter === value" @click="filter = value">{{ label }}</button></div>
      </div>
      <section v-if="!entries.length" class="card cream"><h1 class="title">Здесь будут ваши вина</h1><p>Отсканируйте этикетку и сохраните вино. Его название, ваша оценка и заметка останутся под рукой.</p><NuxtLink class="btn btn-primary" to="/">Сканировать первое вино</NuxtLink></section>
      <p v-else-if="!shown.length" class="muted">Таких вин на полке пока нет. Попробуйте другой фильтр.</p>
      <NuxtLink v-for="e in shown" :key="e.slug" :to="winePath(e.slug)" class="item shelf-item">
        <img class="bottle sm" :src="imageUrl(e.slug)" alt="" @error="($event.target as HTMLImageElement).style.visibility = 'hidden'" />
        <div class="grow"><span class="tag">{{ STATUS[e.status] }}</span><div class="name">{{ e.card.name }}</div><div class="small muted">{{ e.card.winery }}</div><div class="small">{{ e.reaction ? REACTIONS[e.reaction] : 'Без оценки' }} · {{ shelfDate(e.addedAt) }}</div><p v-if="e.note" class="shelf-excerpt small">{{ e.note }}</p></div><span aria-hidden="true">›</span>
      </NuxtLink>
      <ShelfPushSettings v-if="entries.length" />
    </template>
  </main>
</template>
