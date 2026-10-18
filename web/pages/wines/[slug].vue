<script setup lang="ts">
// Сохранённое вино. Если его нет на этой полке (ссылкой поделились), карточку подставляет
// сообщество — из облачных полок других пользователей, а аналог, открытый с карточки, —
// память просмотренных карточек (useSeenCards).
import { similarKnown } from "~/utils/shelf.mjs";
import type { Card } from "~/composables/useScan";
const route = useRoute();
const slug = computed(() => String(route.params.slug));
const { ready, entries, find } = useShelf();
const { ready: authReady } = useAuth();
const { cards: seenCards, seen } = useSeenCards();
const { data: community, refresh } = useCommunity(() => slug.value);
const entry = computed(() => find(slug.value));
const known = computed(() => [...entries.value.flatMap((e) => [{ ...e.card, slug: e.slug }, ...e.analogues]), ...seenCards.value]);
// Полная карточка с описанием лучше урезанной карточки аналога.
const card = computed<Card | undefined>(() => entry.value?.card || seen(slug.value)
  || (community.value?.card as Card | null) || known.value.find((c) => c.slug === slug.value) || undefined);
const analogues = computed(() => entry.value?.analogues.length ? entry.value.analogues
  : card.value ? similarKnown(slug.value, card.value, known.value) : []);
watch(authReady, (r) => { if (r && !entry.value) void refresh(); }, { immediate: true });
useHead({ title: computed(() => `${card.value?.name || "Мои вина"} — Своё Вино`) });
</script>

<template>
  <main class="page">
    <p v-if="!ready" class="muted">Загружаем карточку…</p>
    <WineDetails v-else-if="card" :key="slug" :slug="slug" :card="card" :analogues="analogues" />
    <template v-else>
      <div class="navbar"><NuxtLink class="icon-btn" to="/wines" aria-label="Мои вина"><i class="ph ph-arrow-left" /></NuxtLink></div>
      <section class="card cream stack" style="align-items: center; text-align: center">
        <Mascot :size="72" mood="meh" />
        <h1 class="serif-title">Вина нет на этой полке</h1>
        <p class="muted">Возможно, оно удалено или сохранено на другом устройстве.</p>
        <NuxtLink class="btn btn-primary" to="/wines">Вернуться к моим винам</NuxtLink>
      </section>
    </template>
  </main>
</template>
