<script setup lang="ts">
// Шторка отзыва (1d): оценка, блюда, заметка — сохраняются на полке сразу, как раньше. Кнопка
// внизу решает только судьбу публикации. Без входа публикация ведёт на шлюз (2a), где можно
// войти или оставить отзыв только на полке — данные при этом не теряются.
import { REACTIONS, DISHES, DAY } from "~/utils/shelf.mjs";
import { DISH_ICONS } from "~/utils/wine.mjs";
import type { Card } from "~/composables/useScan";

const props = defineProps<{ open: boolean; slug: string; card: Card }>();
const emit = defineEmits<{ close: []; changed: [] }>();
const route = useRoute();
const { find, update } = useShelf();
const auth = useAuth();
const { data: community } = useCommunity(() => props.slug);
const { publish, unpublish, setPending } = useReviews();
const entry = computed(() => find(props.slug));
const { goal } = useMetrika();

const view = ref<"form" | "gate" | "success">("form");
const share = ref(true);
const anonymous = ref(false);
const shared = ref(false);
const busy = ref(false);
const error = ref("");
const note = ref("");
const reschedule = ref(false);
let noteTimer: ReturnType<typeof setTimeout> | undefined;

const canShare = computed(() => auth.enabled.value);
const published = computed(() => Boolean(community.value?.mine));

watch(() => props.open, (open, was) => {
  // Сохраняем только закрытие открытой шторки: при монтировании поле ещё пустое.
  if (!open) { if (was) saveNote(); return; }
  view.value = "form";
  error.value = "";
  reschedule.value = false;
  note.value = entry.value?.note || "";
  share.value = canShare.value && (published.value || (auth.user.value?.defaultShare ?? true));
  anonymous.value = community.value?.mine?.anonymous ?? false;
}, { immediate: true });

function saveNote() {
  clearTimeout(noteTimer);
  noteTimer = undefined;
  const value = note.value.trim();
  if (!entry.value || entry.value.note === value) return;
  if (update(props.slug, { note: value })) goal("review_note", { slug: props.slug, length: value.length });
}
function queueNote() { clearTimeout(noteTimer); noteTimer = setTimeout(saveNote, 800); }
onBeforeUnmount(() => { if (noteTimer) saveNote(); });

function react(value: "liked" | "disliked") {
  const reaction = entry.value?.reaction === value ? null : value;
  update(props.slug, { reaction });
  goal("review_reaction", { slug: props.slug, reaction: reaction || "none" });
}
function dish(value: string) {
  const current = entry.value?.dishes || [];
  const on = !current.includes(value);
  update(props.slug, { dishes: on ? [...current, value] : current.filter((d) => d !== value) });
  goal("review_dish", { slug: props.slug, dish: value, on });
}
function remind(at: number | null, days: number | null) {
  update(props.slug, { remindAt: at });
  reschedule.value = false;
  goal("reminder_set", { slug: props.slug, days: days ?? "off" });
}
function toggleShare() {
  share.value = !share.value;
  goal("review_share_toggle", { on: share.value });
}
function setAnonymous(value: boolean) {
  anonymous.value = value;
  goal("review_anonymous", { on: value });
}
const shortDate = (value: number) => new Date(value).toLocaleDateString("ru-RU", { day: "2-digit", month: "2-digit" });

const mood = computed(() => (entry.value?.reaction === "disliked" ? "meh" : entry.value?.reaction === "liked" ? "happy" : "wow"));
const willPublish = computed(() => canShare.value && share.value);

async function submit() {
  saveNote();
  error.value = "";
  goal("review_submit", { slug: props.slug, publish: willPublish.value, reaction: entry.value?.reaction || "none" });
  if (willPublish.value) {
    if (!entry.value?.reaction) {
      error.value = "Отметьте, понравилось ли вино, — без оценки отзыв не опубликовать.";
      goal("review_error", { reason: "no_reaction" });
      return;
    }
    if (!auth.user.value) { view.value = "gate"; goal("review_login_gate", { slug: props.slug }); return; }
    if (!auth.complete.value) {
      goal("review_login_redirect", { slug: props.slug, reason: "profile_incomplete" });
      setPending(props.slug, anonymous.value);
      await navigateTo({ path: "/login", query: { redirect: route.fullPath } });
      return;
    }
  }
  busy.value = true;
  try {
    if (willPublish.value) await publish(props.slug, anonymous.value);
    else if (published.value) await unpublish(props.slug);
    goal(willPublish.value ? "review_published" : published.value ? "review_unpublished" : "review_saved",
      { slug: props.slug, anonymous: willPublish.value ? anonymous.value : undefined });
    shared.value = willPublish.value;
    view.value = "success";
    emit("changed");
  } catch (e) {
    error.value = e instanceof Error && !("data" in e) ? e.message : apiError(e);
    goal("review_error", { reason: "request", publish: willPublish.value });
  } finally {
    busy.value = false;
  }
}

async function signIn() {
  goal("review_login_redirect", { slug: props.slug, reason: "gate" });
  setPending(props.slug, anonymous.value);
  emit("close");
  await navigateTo({ path: "/login", query: { redirect: route.fullPath } });
}
function shelfOnly() {
  goal("review_saved", { slug: props.slug, via: "gate" });
  shared.value = false;
  view.value = "success";
}
</script>

<template>
  <BottomSheet :open="open" :label="view === 'gate' ? 'Вход' : 'Отзыв о вине'" @close="emit('close')">
    <!-- 2a: вход при публикации -->
    <template v-if="view === 'gate'">
      <div class="stack" style="align-items: center; text-align: center; gap: 10px">
        <Mascot :size="72" mood="wow" />
        <h2 style="font-size: 25px; line-height: 1.2">Войдите, чтобы поделиться отзывом</h2>
        <p class="muted" style="font-size: 14px; max-width: 300px">Оценка и заметка уже сохранены на полке — вход займёт меньше минуты</p>
      </div>
      <div class="stack" style="gap: 12px">
        <div class="feature"><span><i class="ph ph-chats-circle" /></span>Отзыв появится в карточке вина от вашего имени</div>
        <div class="feature"><span><i class="ph ph-arrows-clockwise" /></span>Полка сохранится в аккаунте и откроется на любом устройстве</div>
      </div>
      <div class="stack" style="gap: 8px">
        <button class="btn btn-primary" @click="signIn"><i class="ph-bold ph-device-mobile" />Войти по номеру телефона</button>
        <div class="grid-2">
          <button class="btn btn-outline" disabled title="Скоро">Яндекс ID</button>
          <button class="btn btn-outline" disabled title="Скоро">VK ID</button>
        </div>
        <p class="hint" style="text-align: center">Вход через Яндекс ID и VK ID появится позже</p>
        <button class="btn btn-text" style="color: var(--text-2); font-weight: 500" @click="shelfOnly">Сохранить только на полке</button>
      </div>
    </template>

    <!-- 1d: отзыв -->
    <template v-else-if="entry">
      <div class="row" style="gap: 14px">
        <Mascot :size="64" :mood="mood" />
        <div class="grow">
          <h2 style="font-size: 23px; line-height: 1.2">Как вам вино?</h2>
          <div class="small muted" style="margin-top: 2px">{{ card.name }}<template v-if="card.winery"> · {{ card.winery }}</template></div>
        </div>
      </div>

      <div class="reaction-row">
        <button v-for="(label, value) in REACTIONS" :key="value" class="reaction" :class="{ on: entry.reaction === value }" :aria-pressed="entry.reaction === value" @click="react(value)">
          <i :class="[entry.reaction === value ? 'ph-fill' : 'ph', value === 'liked' ? 'ph-thumbs-up' : 'ph-thumbs-down']" />{{ label }}
        </button>
      </div>

      <div v-if="!entry.reaction" class="small muted">
        {{ entry.remindAt ? `Напомним оценить ${shortDate(entry.remindAt)}` : "Напоминание отключено" }} ·
        <button class="link" :aria-expanded="reschedule" @click="reschedule = !reschedule">{{ entry.remindAt ? "перенести" : "напомнить" }}</button>
        <div v-if="reschedule" class="reminder-box" style="margin-top: 10px">
          <div class="chips">
            <button class="chip" @click="remind(Date.now() + 3 * DAY, 3)">Через 3 дня</button>
            <button class="chip" @click="remind(Date.now() + 7 * DAY, 7)">Через неделю</button>
            <button v-if="entry.remindAt" class="chip" @click="remind(null, null)">Не напоминать</button>
          </div>
          <ShelfPushSettings />
        </div>
      </div>

      <div class="stack">
        <div style="font-size: 16px; font-weight: 600">С чем пили <span class="muted" style="font-weight: 400; font-size: 14px">— необязательно</span></div>
        <div class="chips">
          <button v-for="d in DISHES" :key="d" class="chip" :class="{ on: entry.dishes.includes(d) }" :aria-pressed="entry.dishes.includes(d)" @click="dish(d)">
            <i :class="['ph', DISH_ICONS[d]]" />{{ d }}
          </button>
        </div>
      </div>

      <div class="stack" style="gap: 8px">
        <label :for="`note-${slug}`" style="font-size: 16px; font-weight: 600">Заметка</label>
        <textarea :id="`note-${slug}`" v-model="note" class="textarea" maxlength="500" rows="3" placeholder="Что запомнилось во вкусе или сочетании?" @input="queueNote" @blur="saveNote" />
        <div class="hint">Сохраняется автоматически · {{ note.length }}/500</div>
      </div>

      <div v-if="canShare" class="share-box">
        <button class="toggle-row" role="switch" :aria-checked="share" @click="toggleShare">
          <i class="ph ph-users-three" />
          <div class="grow"><div style="font-size: 15px; font-weight: 600">Поделиться отзывом</div><div class="small muted" style="font-size: 12px; margin-top: 2px">{{ published ? "Сейчас опубликован в карточке вина" : "Появится в карточке вина для других" }}</div></div>
          <span class="switch" :class="{ on: share }" />
        </button>
        <div v-if="share && auth.user.value" class="segment sm" style="background: #f3ead6">
          <button :class="{ on: !anonymous }" @click="setAnonymous(false)">Как {{ auth.user.value.name || "вы" }}</button>
          <button :class="{ on: anonymous }" @click="setAnonymous(true)">Анонимно</button>
        </div>
      </div>
      <p v-if="error" class="error" role="alert">{{ error }}</p>
    </template>

    <template v-if="view === 'form' && entry" #foot>
      <button class="btn btn-primary" :disabled="busy" @click="submit">
        <span v-if="busy" class="spinner light" />
        <i v-else :class="willPublish ? 'ph-bold ph-paper-plane-tilt' : 'ph-bold ph-check'" />
        {{ willPublish ? "Сохранить и опубликовать" : published ? "Снять с публикации и сохранить" : "Сохранить на полке" }}
      </button>
    </template>

    <template v-if="view === 'success'" #overlay>
      <div class="success" role="status">
        <div style="position: relative; display: flex; justify-content: center"><Confetti v-if="shared" /><Mascot :size="110" mood="happy" /></div>
        <div class="title">{{ shared ? "Отзыв опубликован" : "Сохранено на полке" }}</div>
        <p class="muted" style="font-size: 15px; max-width: 280px">{{ shared ? "Его увидят в карточке вина. Изменить или скрыть можно в любой момент." : "Оценка и заметка видны только вам." }}</p>
        <button class="btn btn-sand" style="width: auto; padding: 0 32px; min-height: 50px; margin-top: 8px" @click="emit('close')">Готово</button>
      </div>
    </template>
  </BottomSheet>
</template>
