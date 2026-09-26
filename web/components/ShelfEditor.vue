<script setup lang="ts">
import type { Card, Analogue } from '~/composables/useScan';
import { STATUS, REACTIONS, DISHES, DAY, shelfDate, winePath } from '~/utils/shelf.mjs';
const props = defineProps<{ slug: string; card: Card; analogues?: Analogue[] }>();
const { ready, error, find, add, update } = useShelf();
const entry = computed(() => find(props.slug));
const note = ref('');
const saved = ref(false);
watch(() => entry.value?.note, value => { note.value = value || ''; }, { immediate: true });
watch(() => props.slug, () => { saved.value = false; });
function saveNote() { saved.value = update(props.slug, { note: note.value.trim() }); }
function dish(value: string) {
  const current = entry.value?.dishes || [];
  update(props.slug, { dishes: current.includes(value) ? current.filter(d => d !== value) : [...current, value] });
}
</script>
<template>
  <section class="card shelf-editor" aria-label="Моя винная полка">
    <p v-if="error" class="error" role="alert">{{ error }}</p>
    <template v-if="ready && !entry">
      <h2>Запомните это вино</h2>
      <p class="muted">Сохраните на полку, чтобы вернуться к нему без камеры.</p>
      <button class="btn btn-primary" @click="add(slug, card, analogues || [])">Сохранить в мои вина</button>
    </template>
    <template v-else-if="entry">
      <div class="shelf-saved" role="status">
        <b>✓ Уже на вашей полке</b>
        <div class="small">Добавлено {{ shelfDate(entry.addedAt) }} · {{ STATUS[entry.status] }}</div>
        <div>Ваша оценка: {{ entry.reaction ? REACTIONS[entry.reaction] : 'пока нет' }}</div>
      </div>
      <fieldset><legend>На моей полке</legend><div class="vintage">
        <button v-for="(label, value) in STATUS" :key="value" class="chip" :class="{ on: entry.status === value }" :aria-pressed="entry.status === value" @click="update(slug, { status: value })">{{ label }}</button>
      </div></fieldset>
      <fieldset id="review"><legend>Как вам вино?</legend><div class="vintage">
        <button v-for="(label, value) in REACTIONS" :key="value" class="chip" :class="{ on: entry.reaction === value }" :aria-pressed="entry.reaction === value" @click="update(slug, { reaction: entry.reaction === value ? null : value })">{{ label }}</button>
      </div></fieldset>
      <fieldset><legend>С чем пробовали? <span class="small muted">Необязательно</span></legend><div class="vintage">
        <button v-for="d in DISHES" :key="d" class="chip" :class="{ on: entry.dishes.includes(d) }" :aria-pressed="entry.dishes.includes(d)" @click="dish(d)">{{ d }}</button>
      </div></fieldset>
      <form class="shelf-note" @submit.prevent="saveNote">
        <label :for="`note-${slug}`">Короткая заметка</label>
        <textarea :id="`note-${slug}`" v-model="note" maxlength="500" rows="3" placeholder="Что запомнилось во вкусе или сочетании?" @input="saved = false" />
        <div class="row"><button class="chip" type="submit">Сохранить заметку</button><span class="small muted" role="status">{{ saved ? 'Сохранено' : `${note.length}/500` }}</span></div>
      </form>
      <div v-if="!entry.reaction" class="reminder-box">
        <p>{{ entry.remindAt ? `Напоминание об оценке: ${shelfDate(entry.remindAt)}` : 'Напоминание отключено' }}</p>
        <div class="vintage">
          <button class="chip" @click="update(slug, { remindAt: Date.now() + 7 * DAY })">Через неделю</button>
          <button v-if="entry.remindAt" class="chip" @click="update(slug, { remindAt: null })">Не напоминать</button>
        </div>
        <ShelfPushSettings />
      </div>
      <NuxtLink class="small" :to="winePath(slug)">Открыть сохранённую карточку →</NuxtLink>
    </template>
    <p v-else class="muted">Загружаем полку…</p>
  </section>
</template>
