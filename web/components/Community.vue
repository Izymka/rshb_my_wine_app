<script setup lang="ts">
// «Что говорят другие»: полки, доля «понравилось», с чем пьют, отзывы. Свой отзыв — первым.
import { DISH_ICONS, initial, reviewDate, plural } from "~/utils/wine.mjs";
const props = defineProps<{ slug: string }>();
const { data, all, loading, refresh, showAll } = useCommunity(() => props.slug);
const { enabled } = useAuth();
const { goal } = useMetrika();
function more() {
  goal("community_show_all", { slug: props.slug, reviews: data.value?.reviewCount });
  void showAll();
}
watch([enabled, () => props.slug], () => { if (enabled.value) void refresh(); }, { immediate: true });
defineExpose({ refresh });
</script>

<template>
  <section v-if="enabled && data" class="stack" style="gap: 14px; padding-top: 12px">
    <h2 class="serif-title">Что говорят другие</h2>
    <div v-if="data.added || data.reviewCount" class="grid-2">
      <div class="tile"><div class="num">{{ data.added.toLocaleString("ru-RU") }}</div><div class="cap">{{ plural(data.added, ["добавил", "добавили", "добавили"]) }} на полку</div></div>
      <div class="tile"><div class="num">{{ data.reviewCount.toLocaleString("ru-RU") }}</div><div class="cap">{{ plural(data.reviewCount, ["отзыв", "отзыва", "отзывов"]) }} в приложении</div></div>
    </div>
    <template v-if="data.reviewCount">
      <div class="stack" style="gap: 8px">
        <div class="head-row small">
          <span class="badge liked" style="background: none; padding: 0; font-size: 13px"><i class="ph-fill ph-thumbs-up" />{{ data.likedPct }}% понравилось</span>
          <span class="muted row" style="gap: 6px"><i class="ph ph-thumbs-down" />{{ 100 - (data.likedPct || 0) }}%</span>
        </div>
        <div class="like-bar"><div :style="{ width: `${data.likedPct}%` }" /></div>
      </div>
      <div v-if="data.pairings.length" class="stack" style="margin-top: 4px">
        <div style="font-size: 15px; font-weight: 600">С чем пробуют</div>
        <div v-for="p in data.pairings" :key="p.dish" class="pairing">
          <span class="ico"><i :class="['ph', DISH_ICONS[p.dish]]" /></span>
          <span>{{ p.dish }}</span>
          <span class="bar"><span :style="{ width: `${p.pct}%` }" /></span>
          <span class="pct">{{ p.pct }}%</span>
        </div>
      </div>
      <div class="stack" style="margin-top: 8px">
        <div style="font-size: 15px; font-weight: 600">Отзывы</div>
        <article v-for="(r, i) in data.reviews" :key="i" class="review" :class="{ mine: r.mine }">
          <div class="row" style="gap: 10px">
            <img v-if="r.avatarUrl" class="avatar" :src="r.avatarUrl" alt="" width="34" height="34" />
            <span v-else class="avatar" :class="{ sand: !r.mine }" style="width: 34px; height: 34px; font-size: 14px">{{ r.anonymous ? "?" : initial(r.name) }}</span>
            <div class="grow"><div class="who">{{ r.name }}{{ r.mine ? " (вы)" : "" }}</div><div class="when">{{ reviewDate(r.date) }}</div></div>
            <span class="badge" :class="r.liked ? 'liked' : 'disliked'"><i :class="['ph-fill', r.liked ? 'ph-thumbs-up' : 'ph-thumbs-down']" />{{ r.liked ? "Понравилось" : "Не моё" }}</span>
          </div>
          <p v-if="r.note" class="text">{{ r.note }}</p>
          <div v-if="r.dishes.length" class="dishes"><span v-for="d in r.dishes" :key="d"><i :class="['ph', DISH_ICONS[d]]" />{{ d }}</span></div>
        </article>
        <button v-if="!all && data.reviewCount > data.reviews.length" class="btn btn-sand" :disabled="loading" @click="more">Все {{ data.reviewCount }} {{ plural(data.reviewCount, ["отзыв", "отзыва", "отзывов"]) }}</button>
      </div>
    </template>
    <p v-else class="muted small">Отзывов пока нет — ваш может стать первым.</p>
  </section>
</template>
