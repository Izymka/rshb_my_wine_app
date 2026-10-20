<script setup lang="ts">
// Шапка экранов винодельни: назад на поле, заголовок, справа — слот (валюта, подпись).
defineProps<{ title: string }>();
</script>

<template>
  <div class="top">
    <NuxtLink to="/winery" class="icon-btn" aria-label="К винодельне"><i class="ph ph-arrow-left" /></NuxtLink>
    <h1>{{ title }}</h1>
    <slot />
  </div>
</template>
