<script setup lang="ts">
// Фото бутылки из каталога, а без фото — силуэт цвета вина (белое, розовое, красное).
import { bottleColor } from "~/utils/wine.mjs";
const props = withDefaults(defineProps<{ slug?: string | null; card?: Record<string, unknown> | null; height?: number; label?: boolean }>(), { height: 90, label: true });
const failed = ref(false);
watch(() => props.slug, () => { failed.value = false; });
const src = computed(() => (props.slug && !failed.value ? imageUrl(props.slug) : undefined));
const k = computed(() => props.height / 90);
const color = computed(() => bottleColor(props.card));
</script>

<template>
  <img v-if="src" :src="src" alt="" loading="lazy" :style="{ height: `${height}px`, maxWidth: '100%', objectFit: 'contain' }" @error="failed = true" />
  <div v-else class="silhouette" aria-hidden="true">
    <div :style="{ width: `${8 * k}px`, height: `${22 * k}px`, background: color, borderRadius: `${3 * k}px ${3 * k}px 0 0` }" />
    <div class="body" :style="{ width: `${28 * k}px`, height: `${68 * k}px`, background: color, borderRadius: `${11 * k}px ${11 * k}px ${4 * k}px ${4 * k}px` }">
      <div v-if="label && height >= 50" :style="{ width: `${20 * k}px`, height: `${16 * k}px`, background: '#fdf9ed', borderRadius: '2px' }" />
    </div>
  </div>
</template>

<style scoped>
.silhouette { display: flex; flex-direction: column; align-items: center; flex: none; }
.body { display: flex; align-items: center; justify-content: center; }
</style>
