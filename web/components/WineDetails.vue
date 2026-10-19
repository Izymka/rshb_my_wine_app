<script setup lang="ts">
// Карточка вина (1c): бутылка крупно, чипы, описание и ароматы, «На моей полке» компактным
// сегментом, отзыв — в шторке, сообщество, сомелье, аналоги; «Удалить» — тихой ссылкой внизу.
// Общая для ответа сканера (/result) и сохранённого вина (/wines/:slug).
import { STATUS, REACTIONS, shelfDate, winePath } from "~/utils/shelf.mjs";
import { DISH_ICONS, aromas, servingTemp, wineTags } from "~/utils/wine.mjs";
import type { Card, Analogue } from "~/composables/useScan";

const props = defineProps<{ slug: string; card: Card; analogues?: Analogue[]; back?: string }>();
const route = useRoute();
const router = useRouter();
const { ready, find, add, update, remove } = useShelf();
const { enabled, user } = useAuth();
const { data: community, refresh: refreshCommunity } = useCommunity(() => props.slug);
const { remember } = useSeenCards();
const { data: catalog } = useCatalog(() => props.slug);
const portal = computed(() => catalog.value?.portal || null);
const entry = computed(() => find(props.slug));
const metrika = useMetrika();
const { goal } = metrika;
// Ответ сканера открывается с back="/", сохранённое вино и аналоги — по ссылке.
const source = computed(() => (props.back === "/" ? "scan" : "link"));
const item = () => ({ slug: props.slug, card: props.card });

const tags = computed(() => {
  const list = wineTags(props.card, props.slug);
  if (portal.value?.alcohol) list.push(`${String(portal.value.alcohol).replace(".", ",")}%`);
  return list;
});
// Как на портале: 4,93, а не округлённые 5,0; целое — с запятой, 5,0.
const ratingText = computed(() => {
  const r = portal.value?.rating || 0;
  return (Number.isInteger(r) ? r.toFixed(1) : String(Number(r.toFixed(2)))).replace(".", ",");
});
// Народный рейтинг портала звёздами: полная, половина (от ,25 до ,75) или пустая.
const stars = computed(() => {
  const r = portal.value?.rating;
  if (!r) return [];
  return [1, 2, 3, 4, 5].map((i) => (r >= i - 0.25 ? "ph-fill ph-star" : r >= i - 0.75 ? "ph-fill ph-star-half" : "ph ph-star"));
});
const notes = computed(() => aromas(props.card.description));
// Температура портала точнее общего правила по стилю вина.
const temp = computed(() => (portal.value?.temperature ? `${portal.value.temperature.replace("-", "–")}°` : servingTemp(props.card, props.slug)));
const descOpen = ref(false);
const sheet = ref(false);
const confirmRemove = ref(false);
const toast = ref("");

const mine = computed(() => community.value?.mine);
const shareLine = computed(() => {
  if (!mine.value) return { icon: "ph-lock-simple", text: "Видите только вы" };
  return { icon: "ph-globe-hemisphere-east", text: mine.value.anonymous ? "Опубликован анонимно" : `Опубликован в отзывах от имени ${user.value?.name || "вас"}` };
});

function save(via: "button" | "rate") {
  if (!add(props.slug, props.card, props.analogues || [])) return;
  goal("shelf_add", { slug: props.slug, via, source: source.value });
  metrika.ecommerce("add", item());
}
function rate() {
  goal("review_open", { slug: props.slug, rated: Boolean(entry.value?.reaction) });
  if (!entry.value) save("rate");
  sheet.value = true;
}
function setStatus(status: string) {
  if (entry.value?.status === status) return;
  update(props.slug, { status: status as "want" | "home" | "tried" });
  goal("shelf_status", { slug: props.slug, status });
}
async function share() {
  const url = `${location.origin}${winePath(props.slug)}`;
  try {
    if (navigator.share) {
      await navigator.share({ title: String(props.card.name || "Вино"), text: `${props.card.name} — ${props.card.winery}`, url });
      goal("wine_share", { slug: props.slug, method: "native" });
    } else {
      await navigator.clipboard.writeText(url);
      goal("wine_share", { slug: props.slug, method: "clipboard" });
      toast.value = "Ссылка скопирована";
      setTimeout(() => (toast.value = ""), 2500);
    }
  } catch { /* пользователь закрыл системное меню */ }
}
function toggleDescription() {
  descOpen.value = !descOpen.value;
  if (descOpen.value) goal("wine_description_open", { slug: props.slug });
}
function askRemove() {
  confirmRemove.value = true;
  goal("shelf_remove_ask", { slug: props.slug });
}
async function deleteWine() {
  if (!remove(props.slug)) return;
  goal("shelf_remove", { slug: props.slug });
  metrika.ecommerce("remove", item());
  await navigateTo("/wines");
}
function goBack() {
  if (props.back) return navigateTo(props.back);
  if (window.history.length > 1) router.back();
  else navigateTo("/wines");
}
onMounted(() => {
  if (route.hash === "#review" && entry.value) sheet.value = true;
});
// Просмотр карточки — цель и «просмотр товара» электронной коммерции.
watch(() => props.slug, () => {
  if (!import.meta.client) return;
  goal("wine_view", { slug: props.slug, source: source.value, on_shelf: Boolean(entry.value) });
  metrika.ecommerce("detail", item());
}, { immediate: true });
watch(() => props.slug, () => { confirmRemove.value = false; descOpen.value = false; });
// Аналоги открываются по тапу — страница вина найдёт их карточки здесь.
watch(() => [props.slug, props.analogues], () => {
  if (import.meta.client) remember([{ ...props.card, slug: props.slug }, ...(props.analogues || [])]);
}, { immediate: true });
</script>

<template>
  <div class="navbar">
    <button class="icon-btn" aria-label="Назад" @click="goBack"><i class="ph ph-arrow-left" /></button>
    <div class="grow" />
    <button class="icon-btn" aria-label="Поделиться" @click="share"><i class="ph ph-share-network" /></button>
    <NuxtLink class="icon-btn" to="/" aria-label="Сканировать ещё" @click="goal('wine_scan_more', { slug })"><i class="ph ph-scan" /></NuxtLink>
  </div>
  <p v-if="toast" class="pill online" role="status" style="align-self: center">{{ toast }}</p>

  <div class="hero">
    <div class="shadow" />
    <WineBottle :slug="slug" :card="card" :height="236" style="position: relative" />
    <span v-if="entry" class="on-shelf"><i class="ph-bold ph-check" />На полке</span>
  </div>

  <div>
    <h1 class="wine-title">{{ card.name }}</h1>
    <div class="wine-sub">{{ card.winery }}<template v-if="card.region"> · {{ card.region }}</template></div>
    <div v-if="tags.length" class="tags" style="margin-top: 12px"><span v-for="t in tags" :key="t" class="tag">{{ t }}</span></div>
  </div>

  <a v-if="portal?.rating" class="card rating" :href="portal.url" target="_blank" rel="noopener" @click="goal('portal_open', { slug })">
    <span class="num">{{ ratingText }}</span>
    <span class="grow">
      <span class="stars" :aria-label="`${portal.rating} из 5`"><i v-for="(cls, i) in stars" :key="i" :class="cls" /></span>
      <span class="cap">Народный рейтинг на «Своё Вино»</span>
    </span>
    <span class="go"><i class="ph ph-arrow-up-right" /></span>
  </a>

  <slot name="extra" />

  <section v-if="card.description || notes.length || temp" class="stack">
    <div class="section-title">О вине</div>
    <template v-if="card.description">
      <p class="desc" :class="{ clamp: !descOpen }">{{ card.description }}</p>
      <button class="link" style="align-self: flex-start" @click="toggleDescription">{{ descOpen ? "Свернуть" : "Читать полностью" }}</button>
    </template>
    <div v-if="notes.length || temp" class="chips">
      <span v-for="a in notes" :key="a.label" class="aroma"><i :class="['ph', a.icon]" :style="{ color: a.color }" />{{ a.label }}</span>
      <span v-if="temp" class="aroma"><i class="ph ph-thermometer-simple" style="color: var(--accent)" />Подавать {{ temp }}</span>
    </div>
  </section>

  <!-- На моей полке -->
  <section class="card stack" style="gap: 14px">
    <template v-if="!ready"><p class="muted">Загружаем полку…</p></template>
    <template v-else-if="entry">
      <div class="head-row"><span class="section-title">На моей полке</span><span class="small muted" style="font-size: 12px">с {{ shelfDate(entry.addedAt) }}</span></div>
      <div class="segment" role="group" aria-label="Статус на полке">
        <button v-for="(label, value) in STATUS" :key="value" :class="{ on: entry.status === value }" :aria-pressed="entry.status === value" @click="setStatus(value)">{{ label }}</button>
      </div>
      <div v-if="entry.reaction" class="my-review">
        <div class="row" style="gap: 8px">
          <span class="badge" :class="entry.reaction" style="font-size: 13px; padding: 5px 10px"><i :class="['ph-fill', entry.reaction === 'liked' ? 'ph-thumbs-up' : 'ph-thumbs-down']" />{{ REACTIONS[entry.reaction] }}</span>
          <span v-for="d in entry.dishes" :key="d" class="dish-dot" :title="d"><i :class="['ph', DISH_ICONS[d]]" /></span>
        </div>
        <p class="quote">{{ entry.note ? `«${entry.note}»` : "Без заметки" }}</p>
        <div v-if="enabled" class="meta-line"><i :class="['ph', shareLine.icon]" />{{ shareLine.text }}</div>
      </div>
      <button class="btn btn-accent-outline" @click="rate"><i class="ph ph-pencil-simple" />{{ entry.reaction ? "Изменить отзыв" : "Оценить вино" }}</button>
    </template>
    <template v-else>
      <div><div class="section-title">Запомните это вино</div><p class="small muted" style="margin-top: 4px">Сохраните на полку, чтобы вернуться к нему без камеры.</p></div>
      <button class="btn btn-primary" @click="save('button')"><i class="ph-bold ph-books" />Сохранить в мои вина</button>
      <button class="btn btn-accent-outline" @click="rate"><i class="ph ph-pencil-simple" />Оценить вино</button>
    </template>
  </section>

  <Community :slug="slug" />

  <Sommelier :key="slug" :slug="slug" :card="card" />

  <section v-if="analogues && analogues.length" class="stack">
    <div class="section-title">Похожие вина других виноделен</div>
    <div class="grid-3">
      <NuxtLink v-for="a in analogues.slice(0, 3)" :key="a.slug" class="analogue" :to="winePath(a.slug)" @click="goal('analogue_open', { slug: a.slug, from: slug })">
        <div style="height: 84px; display: flex; align-items: flex-end; justify-content: center"><WineBottle :slug="a.slug" :card="a" :height="84" /></div>
        <div class="name">{{ a.name }}</div>
        <div class="w">{{ a.winery }}</div>
      </NuxtLink>
    </div>
  </section>

  <slot name="bottom" />

  <div v-if="entry" style="display: flex; flex-direction: column; align-items: center; gap: 10px">
    <button v-if="!confirmRemove" class="quiet-link" @click="askRemove"><i class="ph ph-trash" />Удалить с полки</button>
    <div v-else class="card cream stack" style="width: 100%">
      <p>Удалить вино, оценку и заметку? Напоминание тоже будет отменено.</p>
      <div class="grid-2">
        <button class="btn btn-outline" @click="confirmRemove = false">Оставить</button>
        <button class="btn btn-primary" style="min-height: 50px" @click="deleteWine">Удалить</button>
      </div>
    </div>
  </div>

  <ReviewSheet v-if="entry" :open="sheet" :slug="slug" :card="card" @close="sheet = false" @changed="refreshCommunity" />
</template>
