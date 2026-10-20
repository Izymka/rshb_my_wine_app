<script setup lang="ts">
// Сообщения игры без привязки к месту на поле: ошибки, награды на экранах списков.
const { toasts } = useGame();
const list = computed(() => toasts.value.filter((t) => t.x === undefined));
</script>

<template>
  <Teleport to="body">
    <div v-for="t in list.slice(-1)" :key="t.id" class="g-toast" :class="{ error: t.tone === 'error' }" role="status">
      <i :class="['ph-bold', t.tone === 'error' ? 'ph-warning-circle' : 'ph-sparkle']" />{{ t.text }}
    </div>
  </Teleport>
</template>
