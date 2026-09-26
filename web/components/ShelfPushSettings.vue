<script setup lang="ts">
const { enabled, available, busy, message, enable, disable } = useShelfPush();
</script>
<template>
  <div class="push-settings">
    <p class="muted" role="status">{{ message }}</p>
    <button v-if="available && !enabled" class="chip" :disabled="busy" @click="enable">{{ busy ? 'Подключаем…' : 'Включить пуш-напоминания' }}</button>
    <button v-if="enabled" class="chip" :disabled="busy" @click="disable">Отключить пуши</button>
  </div>
</template>
