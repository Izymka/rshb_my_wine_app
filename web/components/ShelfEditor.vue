<script setup lang="ts">
import type { Card, Analogue } from '~/composables/useScan';
import { STATUS, REACTIONS, DISHES, DAY, shelfDate, winePath } from '~/utils/shelf.mjs';
const props = defineProps<{ slug: string; card: Card; analogues?: Analogue[] }>();
const { ready, error, find, add, update } = useShelf();
const route = useRoute();
const entry = computed(() => find(props.slug));
// На странице этого же вина ссылка вела бы сама на себя.
const onOwnPage = computed(() => route.path === winePath(props.slug));
const DISHES_SHOWN = 6;
const allDishes = ref(false);
const shownDishes = computed(() => allDishes.value ? DISHES
  : DISHES.filter((d, i) => i < DISHES_SHOWN || entry.value?.dishes.includes(d)));
const hiddenDishes = computed(() => DISHES.length - shownDishes.value.length);
const reschedule = ref(false);
const note = ref('');
let noteTimer: ReturnType<typeof setTimeout> | undefined;
watch(() => entry.value?.note, value => { if (!noteTimer) note.value = value || ''; }, { immediate: true });
function saveNote() {
  clearTimeout(noteTimer);
  noteTimer = undefined;
  const value = note.value.trim();
  if (entry.value && entry.value.note !== value) update(props.slug, { note: value });
}
function queueNote() { clearTimeout(noteTimer); noteTimer = setTimeout(saveNote, 1000); }
onBeforeUnmount(() => { if (noteTimer) saveNote(); });
function dish(value: string) {
  const current = entry.value?.dishes || [];
  update(props.slug, { dishes: current.includes(value) ? current.filter(d => d !== value) : [...current, value] });
}
function remind(at: number | null) { update(props.slug, { remindAt: at }); reschedule.value = false; }
const shortDate = (value: number) => new Date(value).toLocaleDateString('ru-RU', { day: '2-digit', month: '2-digit' });
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
      <fieldset id="review" class="shelf-review"><legend>Как вам вино?</legend>
        <div class="reaction-row">
          <button v-for="(label, value) in REACTIONS" :key="value" class="reaction" :class="{ on: entry.reaction === value }" :aria-pressed="entry.reaction === value" @click="update(slug, { reaction: entry.reaction === value ? null : value })">{{ label }}</button>
        </div>
        <template v-if="!entry.reaction">
          <p class="shelf-hint">
            {{ entry.remindAt ? `Напомним оценить ${shortDate(entry.remindAt)}` : 'Напоминание отключено' }} ·
            <button class="link" :aria-expanded="reschedule" @click="reschedule = !reschedule">{{ entry.remindAt ? 'перенести' : 'напомнить' }}</button>
          </p>
          <div v-if="reschedule" class="reminder-box">
            <div class="vintage">
              <button class="chip" @click="remind(Date.now() + 3 * DAY)">Через 3 дня</button>
              <button class="chip" @click="remind(Date.now() + 7 * DAY)">Через неделю</button>
              <button v-if="entry.remindAt" class="chip" @click="remind(null)">Не напоминать</button>
            </div>
            <ShelfPushSettings />
          </div>
        </template>
      </fieldset>
      <fieldset><legend>С чем пили <span class="optional">— необязательно</span></legend><div class="vintage">
        <button v-for="d in shownDishes" :key="d" class="chip" :class="{ on: entry.dishes.includes(d) }" :aria-pressed="entry.dishes.includes(d)" @click="dish(d)">{{ d }}</button>
        <button v-if="hiddenDishes" class="link more" @click="allDishes = true">Ещё {{ hiddenDishes }}</button>
      </div></fieldset>
      <div class="shelf-note">
        <label :for="`note-${slug}`">Заметка</label>
        <textarea :id="`note-${slug}`" v-model="note" maxlength="500" rows="4" placeholder="Что запомнилось во вкусе или сочетании?" @input="queueNote" @blur="saveNote" />
        <span class="shelf-hint">Сохраняется автоматически · до 500 символов</span>
      </div>
      <NuxtLink v-if="!onOwnPage" class="small" :to="winePath(slug)">Открыть сохранённую карточку →</NuxtLink>
    </template>
    <p v-else class="muted">Загружаем полку…</p>
  </section>
</template>
