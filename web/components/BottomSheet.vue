<script setup lang="ts">
// Шторка снизу: затемнение, «ручка», прокручиваемое тело и закреплённый низ (#foot).
const props = defineProps<{ open: boolean; label: string }>();
const emit = defineEmits<{ close: [] }>();
function onKey(e: KeyboardEvent) { if (e.key === "Escape") emit("close"); }
watch(() => props.open, (open) => {
  if (!import.meta.client) return;
  document.documentElement.style.overflow = open ? "hidden" : "";
  if (open) window.addEventListener("keydown", onKey);
  else window.removeEventListener("keydown", onKey);
}, { immediate: true });
onBeforeUnmount(() => {
  if (!import.meta.client) return;
  document.documentElement.style.overflow = "";
  window.removeEventListener("keydown", onKey);
});
</script>

<template>
  <Teleport to="body">
    <div v-if="open" class="sheet-backdrop" @click.self="emit('close')">
      <div class="sheet" role="dialog" aria-modal="true" :aria-label="label">
        <div class="handle"><span /></div>
        <button class="close" aria-label="Закрыть" @click="emit('close')"><i class="ph ph-x" /></button>
        <div class="body"><slot /></div>
        <div v-if="$slots.foot" class="foot"><slot name="foot" /></div>
        <slot name="overlay" />
      </div>
    </div>
  </Teleport>
</template>
