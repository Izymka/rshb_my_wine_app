<script setup lang="ts">
// Экраны 3 (карточка), 4 (вопрос о годе), 5 (не узнали) из DESIGN.md. Один маршрут, три
// состояния: решает поле answered ответа сервиса, а не вероятность — порог откалиброван
// вместе с моделью, и заводить свой здесь нельзя.
const { preview, result, reset } = useScan();
const router = useRouter();

if (!result.value) {
  // Прямой заход без сканирования (обновили страницу) — назад к камере.
  await navigateTo("/", { replace: true });
}

const r = computed(() => result.value!);
const answered = computed(() => r.value.answered && r.value.card);
const card = computed(() => r.value.card || r.value.candidates[0]?.card || {});
const shownSlug = computed(() => r.value.item_id || r.value.candidates[0]?.item_id || null);
const similar = computed(() => {
  // «Возможно, одно из этих»: кандидаты кроме того, что уже показан как ответ.
  return r.value.candidates.filter((c) => c.item_id !== r.value.item_id).slice(0, 5);
});
const wrong = ref(false); // нажали «это не то вино»
const pickedYear = ref<string | null>(null);
const yearChoices = computed(() => {
  const now = new Date().getFullYear();
  const guess = r.value.vintage?.year;
  const base = guess ? [guess - 1, guess, guess + 1] : [now - 3, now - 2, now - 1];
  return base.filter((y) => y <= now);
});
const grapes = computed(() => {
  const g = card.value.grapes;
  return typeof g === "string" ? g.split(",").map((s) => s.trim()).filter(Boolean) : [];
});

function again() {
  reset();
  router.push("/");
}

function pick(c: { item_id: string; card: Record<string, unknown> }) {
  // Пользователь выбрал вино из похожих — показываем его карточку как ответ.
  result.value = { ...r.value, answered: true, item_id: c.item_id, card: c.card as never, guard: null };
  wrong.value = false;
  window.scrollTo({ top: 0, behavior: "smooth" });
}
</script>

<template>
  <div v-if="result" class="page">
    <header class="topbar">
      <img src="/icon.png" alt="" />
      <div class="grow">
        <div class="brand">Своё Вино</div>
        <div class="sub">{{ answered ? "нашли карточку" : "не узнали это вино" }}</div>
      </div>
      <button class="btn btn-ghost" style="width: auto; padding: 10px 16px; font-size: 14px" @click="again">
        Снять ещё
      </button>
    </header>

    <!-- Ответ найден -->
    <section v-if="answered && !wrong" class="card">
      <div class="row" style="align-items: flex-start">
        <img v-if="shownSlug" class="bottle" :src="imageUrl(shownSlug)" alt="эталон из каталога" />
        <div class="grow">
          <h1 class="title">{{ card.name }}</h1>
          <div class="winery">{{ card.winery }}<span v-if="card.region"> · {{ card.region }}</span></div>
        </div>
      </div>
      <div style="margin-top: 12px">
        <span v-if="card.category" class="tag accent">{{ card.category }}</span>
        <span v-for="g in grapes" :key="g" class="tag">{{ g }}</span>
      </div>

      <p v-if="card.description" class="desc" style="margin: 12px 0 0">{{ card.description }}</p>
      <p v-if="card.color" class="small muted" style="margin: 8px 0 0">Цвет: {{ card.color }}</p>

      <div v-if="preview" class="row small muted" style="margin-top: 12px">
        <img class="photo" style="width: 48px; height: 64px" :src="preview" alt="ваш кадр" />
        <span>Ваш кадр — мы смотрели именно на эту бутылку</span>
      </div>

      <!-- Вопрос о годе: блок внутри карточки, один тап -->
      <template v-if="r.vintage && (r.vintage.year || r.vintage.ask)">
      <div class="hr" style="margin: 14px 0" />
      <div v-if="!r.vintage.ask" class="small muted">
        Год урожая: <b style="color: var(--text)">{{ r.vintage.year }}</b>
        <span v-if="r.vintage.source === 'text'"> — прочитан с этикетки</span>
      </div>
      <div v-else>
        <div style="margin-bottom: 8px">Какой год на бутылке?</div>
        <div class="vintage">
          <button v-for="y in yearChoices" :key="y" class="chip" :class="{ on: pickedYear === String(y) }" @click="pickedYear = String(y)">{{ y }}</button>
          <button class="chip" :class="{ on: pickedYear === 'нет' }" @click="pickedYear = 'нет'">не знаю</button>
        </div>
      </div>
      </template>

      <div class="hr" style="margin: 16px 0" />
      <button class="btn btn-ghost" @click="wrong = true">Это не то вино</button>
    </section>

    <!-- Отказ или «это не то вино» -->
    <section v-else class="card cream">
      <h2 class="title" style="font-size: 24px">{{ wrong ? "Покажем похожие" : "Не узнали это вино" }}</h2>
      <p class="muted" style="margin: 8px 0 0">
        <template v-if="r.guard && r.guard.includes('twin')">
          Этикетка почти как у знакомого вина, но слова на ней не сходятся с карточкой — скорее
          всего, это соседний вариант линейки, которого пока нет в каталоге.
        </template>
        <template v-else-if="!wrong">Возможно, одно из этих — или этого вина ещё нет на платформе.</template>
        <template v-else>Выберите ваше вино из списка — или снимите этикетку ещё раз.</template>
      </p>
      <div class="row" style="margin-top: 12px; align-items: flex-start">
        <img v-if="preview" class="photo" :src="preview" alt="ваш кадр" />
        <div class="grow small muted" v-if="r.recognized_text">
          Прочитали на этикетке:<br /><span style="color: var(--text)">{{ r.recognized_text.slice(0, 160) }}</span>
        </div>
      </div>
    </section>

    <!-- Похожие: близнецы из окна ре-ранкинга -->
    <section v-if="(!answered || wrong) && r.candidates.length" class="card">
      <h3 style="font-size: 18px; margin-bottom: 10px">Возможно, одно из этих</h3>
      <div class="list">
        <button v-for="c in (wrong ? similar : r.candidates.slice(0, 5))" :key="c.item_id" class="item" @click="pick(c)">
          <img class="bottle sm" :src="imageUrl(c.item_id)" alt="" />
          <div class="grow">
            <div class="name">{{ c.card.name }}</div>
            <div class="small muted">{{ c.card.winery }}<span v-if="c.card.category"> · {{ c.card.category }}</span></div>
          </div>
          <span class="muted">›</span>
        </button>
      </div>
      <p class="small muted" style="margin: 12px 0 0">Моего вина здесь нет — <a href="#" @click.prevent="again">переснять</a></p>
    </section>

    <!-- Аналоги из других виноделен -->
    <section v-if="r.analogues && r.analogues.length" class="card">
      <h3 style="font-size: 18px">{{ answered && !wrong ? "Похожие вина других виноделен" : "Чем заменить" }}</h3>
      <p class="small muted" style="margin: 4px 0 12px">Тот же цвет и стиль, близкие сорта и регион</p>
      <div class="strip">
        <div v-for="a in r.analogues" :key="a.slug" class="tile">
          <img :src="imageUrl(a.slug)" alt="" loading="lazy" />
          <div class="name">{{ a.name }}</div>
          <div class="w">{{ a.winery }}</div>
        </div>
      </div>
    </section>

    <Sommelier v-if="answered && !wrong && shownSlug" :slug="shownSlug" :card="card" />

    <div class="footer">
      ответ за {{ Math.round(r.timings_ms?.total || 0) }} мс
      <template v-if="r.confidence"> · уверенность top-1 {{ (r.confidence.top1 * 100).toFixed(0) }}%, top-5 {{ (r.confidence.top5 * 100).toFixed(0) }}%</template>
    </div>
  </div>
</template>
