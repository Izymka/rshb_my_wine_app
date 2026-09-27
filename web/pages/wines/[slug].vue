<script setup lang="ts">
import { winePath, similarKnown } from '~/utils/shelf.mjs';
const route = useRoute();
const slug = computed(() => String(route.params.slug));
const { ready, entries, find, remove } = useShelf();
const entry = computed(() => find(slug.value));
const known = computed(() => entries.value.flatMap(e => [{ ...e.card, slug: e.slug }, ...e.analogues]));
const card = computed(() => entry.value?.card || known.value.find(c => c.slug === slug.value));
const analogues = computed(() => entry.value?.analogues.length ? entry.value.analogues : card.value ? similarKnown(slug.value, card.value, known.value) : []);
const confirmRemove = ref(false);
watch(slug, () => { confirmRemove.value = false; });
watch([ready, () => route.hash, slug], async () => {
  if (import.meta.client && ready.value && route.hash === '#review') {
    await nextTick();
    document.getElementById('review')?.scrollIntoView({ block: 'start' });
  }
}, { flush: 'post' });
async function deleteWine() { if (remove(slug.value)) await navigateTo('/wines'); }
useHead({ title: computed(() => `${card.value?.name || 'Мои вина'} — Своё Вино`) });
</script>
<template>
  <main class="page">
    <header class="topbar"><NuxtLink to="/wines">← Мои вина</NuxtLink><div class="grow" /><NuxtLink to="/">Сканировать</NuxtLink></header>
    <p v-if="!ready">Загружаем карточку…</p>
    <template v-else-if="card">
      <section class="card"><div class="row"><img class="bottle" :src="imageUrl(slug)" alt="" @error="($event.target as HTMLImageElement).style.visibility = 'hidden'" /><div class="grow"><h1 class="title">{{ card.name }}</h1><p class="muted">{{ card.winery }}<span v-if="card.region"> · {{ card.region }}</span></p></div></div><div v-if="card.category || card.grapes" class="tags"><span v-if="card.category" class="tag">{{ card.category }}</span><span v-if="card.grapes" class="tag">{{ card.grapes }}</span></div><p v-if="card.vintage">Год урожая: {{ card.vintage }}</p><p v-if="card.description">{{ card.description }}</p></section>
      <ShelfEditor :key="slug" :slug="slug" :card="card" :analogues="analogues" />
      <section v-if="entry?.reaction === 'liked'" class="card"><h2>Похожее на понравившееся</h2><p class="small muted">По характеристикам сохранённых карточек. Вашу оценку других вин учитываем: «Не моё» сюда не попадёт.</p>
        <div v-if="analogues.filter(a => find(a.slug)?.reaction !== 'disliked').length" class="list"><NuxtLink v-for="a in analogues.filter(a => find(a.slug)?.reaction !== 'disliked')" :key="a.slug" class="item" :to="winePath(a.slug)"><div><b>{{ a.name }}</b><div class="small muted">{{ a.winery }} · {{ a.grapes || a.category }}</div></div><span>›</span></NuxtLink></div>
        <p v-else class="muted">Пока нет подходящих сохранённых аналогов. Сканируйте другие вина, чтобы расширить подборку.</p>
      </section>
      <Sommelier :key="slug" :slug="slug" :card="card" />
      <button v-if="entry && !confirmRemove" class="btn btn-ghost" @click="confirmRemove = true">Удалить с полки</button>
      <section v-if="entry && confirmRemove" class="card"><p>Удалить вино, оценку и заметку? Напоминание тоже будет отменено.</p><div class="row"><button class="btn btn-ghost" @click="confirmRemove = false">Оставить</button><button class="btn btn-primary" @click="deleteWine">Удалить</button></div></section>
    </template>
    <section v-else class="card"><h1>Вина нет на этой полке</h1><p>Возможно, оно удалено или сохранено на другом устройстве.</p><NuxtLink to="/wines">Вернуться к моим винам</NuxtLink></section>
  </main>
</template>
