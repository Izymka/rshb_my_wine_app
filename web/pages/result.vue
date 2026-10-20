<script setup lang="ts">
// Ответ сканера: карточка (1c), вопрос о годе, отказ «Не узнали». Решает поле answered ответа
// сервиса, а не вероятность — порог откалиброван вместе с моделью, и заводить свой здесь нельзя.
import { similarKnown, winePath } from "~/utils/shelf.mjs";
const { preview, result, reset } = useScan();
const { remember } = useSeenCards();
const router = useRouter();
const { goal } = useMetrika();

if (!result.value) {
  // Прямой заход без сканирования (обновили страницу) — назад к камере.
  await navigateTo("/", { replace: true });
}

const r = computed(() => result.value!);
const answered = computed(() => r.value.answered && r.value.card);
const card = computed(() => r.value.card || r.value.candidates[0]?.card || {});
const shownSlug = computed(() => r.value.item_id || r.value.candidates[0]?.item_id || null);
const similar = computed(() => r.value.candidates.filter((c) => c.item_id !== r.value.item_id).slice(0, 5));
const wrong = ref(false); // нажали «это не то вино»
const pickedYear = ref<string | null>(null);
const yearChoices = computed(() => {
  const now = new Date().getFullYear();
  const guess = r.value.vintage?.year;
  const base = guess ? [guess - 1, guess, guess + 1] : [now - 3, now - 2, now - 1];
  return base.filter((y) => y <= now);
});
const shelfCard = computed(() => ({
  ...card.value,
  vintage: pickedYear.value && pickedYear.value !== "нет" ? pickedYear.value : r.value.vintage?.year || card.value.vintage,
}));

function again() {
  goal("scan_again", { answered: Boolean(answered.value), wrong: wrong.value });
  reset();
  router.push("/");
}

function pick(c: { item_id: string; card: Record<string, unknown> }, position: number) {
  goal("scan_candidate_pick", { slug: c.item_id, position: position + 1, wrong: wrong.value });
  // Пользователь выбрал вино из похожих — показываем его карточку как ответ.
  const knownCards = [
    ...(r.value.analogues || []),
    ...r.value.candidates.map((candidate) => ({ ...candidate.card, slug: candidate.item_id })),
  ];
  result.value = {
    ...r.value, answered: true, item_id: c.item_id, card: c.card as never,
    guard: null, vintage: null, analogues: similarKnown(c.item_id, c.card, knownCards), twins: [],
  };
  wrong.value = false;
  pickedYear.value = null;
  window.scrollTo({ top: 0, behavior: "smooth" });
}
// Близнец по этикетке: показываем его карточку, а прежний ответ оставляем в «или это вино»,
// чтобы можно было вернуться одним тапом.
function pickTwin(c: { item_id: string; probability: number; card: Record<string, unknown> }, position: number) {
  goal("scan_twin_pick", { slug: c.item_id, position: position + 1 });
  const previous = { item_id: r.value.item_id!, probability: r.value.probability, card: r.value.card as never };
  const rest = (r.value.twins || []).filter((t) => t.item_id !== c.item_id);
  pick(c, position);
  result.value = { ...result.value!, twins: [previous, ...rest] };
}
function notThis() {
  goal("scan_wrong_wine", { slug: shownSlug.value });
  wrong.value = true;
}
function pickYear(year: string) {
  pickedYear.value = year;
  goal("vintage_pick", { year, guess: r.value.vintage?.year ?? undefined });
}
function openAnalogue(slug: string) {
  goal("analogue_open", { slug, from: "not_recognized" });
  remember(r.value.analogues || []);
}
useHead({ title: computed(() => (answered.value ? `${card.value.name} — Своё Вино` : "Не узнали вино — Своё Вино")) });
</script>

<template>
  <div v-if="result" class="page">
    <WineDetails v-if="answered && !wrong && shownSlug" :key="shownSlug" :slug="shownSlug" :card="shelfCard" :analogues="r.analogues || []" back="/">
      <template #extra>
        <div v-if="preview" class="row small muted">
          <img class="photo" style="width: 44px; height: 58px" :src="preview" alt="ваш кадр" />
          <span class="grow">Ваш кадр — смотрели именно на эту бутылку</span>
          <button class="link" @click="notThis">Не то вино?</button>
        </div>
        <!-- Соседки по линейке, которых по лицевой этикетке не отличить -->
        <div v-if="r.twins && r.twins.length" class="card cream stack" style="gap: 8px">
          <div style="font-weight: 600">Или это вино?</div>
          <div class="small muted">По лицевой этикетке их не отличить. Сверьтесь с контрэтикеткой: на ней обычно указан сорт.</div>
          <button v-for="(c, i) in r.twins" :key="c.item_id" class="candidate" @click="pickTwin(c, i)">
            <div class="thumb"><WineBottle :slug="c.item_id" :card="c.card" :height="76" /></div>
            <div class="grow">
              <div style="font-weight: 600">{{ c.card.name }}</div>
              <div class="small muted">{{ c.card.winery }}<span v-if="c.card.grapes"> · {{ c.card.grapes }}</span></div>
            </div>
            <i class="ph ph-caret-right" style="color: var(--text-3)" />
          </button>
        </div>
        <!-- Вопрос о годе: один тап -->
        <div v-if="r.vintage && (r.vintage.year || r.vintage.ask)" class="card cream">
          <div v-if="!r.vintage.ask" class="small muted">
            Год урожая: <b style="color: var(--text)">{{ r.vintage.year }}</b>
            <span v-if="r.vintage.source === 'text'"> — прочитан с этикетки</span>
          </div>
          <div v-else class="stack" style="gap: 8px">
            <div style="font-weight: 600">Какой год на бутылке?</div>
            <div class="chips">
              <button v-for="y in yearChoices" :key="y" class="chip" :class="{ on: pickedYear === String(y) }" @click="pickYear(String(y))">{{ y }}</button>
              <button class="chip" :class="{ on: pickedYear === 'нет' }" @click="pickYear('нет')">не знаю</button>
            </div>
          </div>
        </div>
      </template>
      <template #bottom>
        <div class="footer" style="margin-top: 0">
          ответ за {{ Math.round(r.timings_ms?.total || 0) }} мс
          <template v-if="r.confidence"> · уверенность top-1 {{ (r.confidence.top1 * 100).toFixed(0) }}%</template>
        </div>
      </template>
    </WineDetails>

    <!-- Отказ или «это не то вино» -->
    <template v-else>
      <div class="navbar">
        <button class="icon-btn" aria-label="Назад" @click="wrong ? (wrong = false) : again()"><i class="ph ph-arrow-left" /></button>
        <div class="grow" />
        <NuxtLink class="icon-btn" to="/wines" aria-label="Мои вина"><i class="ph ph-books" /></NuxtLink>
      </div>
      <section class="stack" style="gap: 14px">
        <div class="row" style="gap: 14px; align-items: flex-start">
          <Mascot :size="64" mood="meh" />
          <div class="grow">
            <h1 class="page-title" style="font-size: 26px">{{ wrong ? "Покажем похожие" : "Не узнали это вино" }}</h1>
            <p class="muted small" style="margin-top: 6px">
              <template v-if="r.guard && (r.guard.includes('twin') || r.guard.includes('contra'))">
                Этикетка почти как у знакомого вина, но слова на ней не сходятся с карточкой — скорее
                всего, это соседний вариант линейки, которого пока нет в каталоге.
              </template>
              <template v-else-if="!wrong">Возможно, одно из этих — или этого вина ещё нет на платформе.</template>
              <template v-else>Выберите ваше вино из списка — или снимите этикетку ещё раз.</template>
            </p>
          </div>
        </div>
        <div v-if="preview" class="row small muted" style="align-items: flex-start">
          <img class="photo" :src="preview" alt="ваш кадр" />
          <div v-if="r.recognized_text" class="grow">Прочитали на этикетке:<br /><span style="color: var(--text)">{{ r.recognized_text.slice(0, 160) }}</span></div>
        </div>
      </section>

      <section v-if="r.candidates.length" class="stack">
        <div class="section-title">Возможно, одно из этих</div>
        <button v-for="(c, i) in (wrong ? similar : r.candidates.slice(0, 5))" :key="c.item_id" class="candidate" @click="pick(c, i)">
          <div class="thumb"><WineBottle :slug="c.item_id" :card="c.card" :height="76" /></div>
          <div class="grow">
            <div style="font-weight: 600">{{ c.card.name }}</div>
            <div class="small muted">{{ c.card.winery }}<span v-if="c.card.category"> · {{ c.card.category }}</span></div>
          </div>
          <i class="ph ph-caret-right" style="color: var(--text-3)" />
        </button>
      </section>

      <section v-if="r.analogues && r.analogues.length" class="stack">
        <div class="section-title">Чем заменить</div>
        <p class="small muted" style="margin-top: -4px">Тот же цвет и стиль, близкие сорта и регион</p>
        <div class="grid-3">
          <NuxtLink v-for="a in r.analogues.slice(0, 3)" :key="a.slug" class="analogue" :to="winePath(a.slug)" @click="openAnalogue(a.slug)">
            <div style="height: 84px; display: flex; align-items: flex-end"><WineBottle :slug="a.slug" :card="a" :height="84" /></div>
            <div class="name">{{ a.name }}</div>
            <div class="w">{{ a.winery }}</div>
          </NuxtLink>
        </div>
      </section>

      <button class="btn btn-primary" @click="again"><i class="ph-bold ph-camera" />Снять ещё раз</button>
    </template>
    <GameScanReward v-if="answered && !wrong" :slug="shownSlug" />
  </div>
</template>
