<script setup lang="ts">
const colors = ["#8f3d42", "#d9a441", "#4f7a4a", "#e2a39b", "#c8b46a"];
const bits = Array.from({ length: 18 }, (_, i) => {
  const angle = (i / 18) * Math.PI * 2, r = 70 + (i % 3) * 24;
  return {
    width: "8px", height: i % 2 ? "8px" : "14px", borderRadius: i % 2 ? "50%" : "3px", background: colors[i % 5],
    "--dx": `${Math.cos(angle) * r}px`, "--dy": `${Math.sin(angle) * r}px`, animationDelay: `${(i % 4) * 0.1}s`,
  };
});
</script>

<template>
  <div class="confetti" aria-hidden="true"><span v-for="(b, i) in bits" :key="i" :style="b" /></div>
</template>

<style scoped>
.confetti { position: absolute; left: 50%; top: 50%; width: 0; height: 0; }
.confetti span { position: absolute; animation: svPop 1.8s ease-out infinite; }
</style>
