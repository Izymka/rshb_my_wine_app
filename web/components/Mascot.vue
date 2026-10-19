<script setup lang="ts">
// Персонаж «Виноградинка»: фиолетовая ягода с веточкой и листиком. Подсказывает на главной,
// «удивляется» при поиске, хмурится на «Не моё» и радуется публикации. Рисуется CSS, без картинок.
const props = withDefaults(defineProps<{ size?: number; mood?: "happy" | "wow" | "meh" }>(), { size: 64, mood: "happy" });
const s = computed(() => props.size);
const px = (v: number) => `${v}px`;
const lw = computed(() => Math.max(2, s.value * 0.035));
const eyeStyle = (x: number) => ({
  top: px(s.value * 0.52), left: px(x), width: px(s.value * 0.12),
  height: px(props.mood === "meh" ? s.value * 0.05 : s.value * 0.15),
});
const mouthStyle = computed(() => {
  const v = s.value;
  if (props.mood === "happy") return {
    top: px(v * 0.67), left: px(v * 0.41), width: px(v * 0.18), height: px(v * 0.09),
    border: `${lw.value}px solid #2c2a28`, borderTop: "none", borderRadius: `0 0 ${v * 0.1}px ${v * 0.1}px`,
  };
  if (props.mood === "meh") return { top: px(v * 0.74), left: px(v * 0.43), width: px(v * 0.14), height: px(lw.value), background: "#2c2a28", borderRadius: px(lw.value) };
  return { top: px(v * 0.68), left: px(v * 0.45), width: px(v * 0.1), height: px(v * 0.12), borderRadius: "50%", background: "#2c2a28" };
});
</script>

<template>
  <div class="mascot" :style="{ width: px(s), height: px(s * 1.1) }" aria-hidden="true">
    <div class="m-shadow" :style="{ left: px(s * 0.22), right: px(s * 0.22), bottom: px(-s * 0.04), height: px(s * 0.06) }" />
    <div class="m-stem" :style="{ top: px(s * 0.17), left: px(s * 0.48), width: px(s * 0.05), height: px(s * 0.17), borderRadius: px(s * 0.03) }" />
    <div class="m-leaf" :style="{ top: px(s * 0.15), left: px(s * 0.52), width: px(s * 0.22), height: px(s * 0.12) }" />
    <div class="m-berry" :style="{ left: px(s * 0.17), top: px(s * 0.3), width: px(s * 0.66), height: px(s * 0.66) }" />
    <div class="m-shine" :style="{ top: px(s * 0.38), left: px(s * 0.3), width: px(s * 0.1), height: px(s * 0.07) }" />
    <div v-for="x in [s * 0.34, s * 0.54]" :key="x" class="m-eye" :style="eyeStyle(x)"><i v-if="mood !== 'meh'" /></div>
    <div v-for="x in [s * 0.24, s * 0.66]" :key="`c${x}`" class="m-cheek" :style="{ top: px(s * 0.69), left: px(x), width: px(s * 0.1), height: px(s * 0.065) }" />
    <div class="m-mouth" :style="mouthStyle" />
  </div>
</template>

<style scoped>
.mascot { position: relative; flex: none; animation: svBob 3.2s ease-in-out infinite; }
.mascot > div { position: absolute; box-sizing: border-box; }
.m-shadow { border-radius: 50%; background: rgba(60, 40, 20, 0.12); }
.m-stem { background: #7a5a3a; transform: rotate(8deg); }
.m-leaf { border-radius: 0 100% 0 100%; background: #6f9a4f; transform: rotate(-12deg); }
.m-berry { border-radius: 50%; background: radial-gradient(circle at 35% 30%, #b98bb8 0%, #8a5a92 45%, #6b3f74 100%); }
.m-shine { border-radius: 50%; background: rgba(255, 255, 255, 0.55); transform: rotate(-30deg); }
.m-eye { border-radius: 50%; background: #2c2a28; animation: svBlink 4.5s infinite; }
.m-eye i { position: absolute; top: 16%; left: 20%; width: 38%; height: 32%; border-radius: 50%; background: #fff; }
.m-cheek { border-radius: 50%; background: #e7a09c; opacity: 0.85; }
</style>
