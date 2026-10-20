<script setup lang="ts">
// Кольцо прогресса: уровень в HUD, таймер здания.
const props = withDefaults(defineProps<{ value: number; size: number; stroke?: number; color?: string; track?: string }>(), { stroke: 4, color: "#6f9a4f", track: "#efe6d8" });
const r = computed(() => (props.size - props.stroke) / 2);
const c = computed(() => 2 * Math.PI * r.value);
</script>

<template>
  <svg :width="size" :height="size" :viewBox="`0 0 ${size} ${size}`" style="display: block; transform: rotate(-90deg)" aria-hidden="true">
    <circle :cx="size / 2" :cy="size / 2" :r="r" fill="none" :stroke="track" :stroke-width="stroke" />
    <circle :cx="size / 2" :cy="size / 2" :r="r" fill="none" :stroke="color" :stroke-width="stroke" stroke-linecap="round"
      :stroke-dasharray="c" :stroke-dashoffset="c * (1 - Math.min(1, Math.max(0, value)))" />
  </svg>
</template>
