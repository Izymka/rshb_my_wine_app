<script setup lang="ts">
// Цифровой сомелье: короткий разговор о найденном вине. Блок появляется только если сервис
// умеет отвечать (/sommelier не 404); при ошибке провайдера — честная строка, карточка не страдает.
const props = defineProps<{ slug: string; card: Record<string, unknown> }>();
const { online } = useConnectivity();

const available = ref<boolean | null>(null);
const history = ref<{ role: "user" | "assistant"; content: string }[]>([]);
const question = ref("");
const busy = ref(false);
const failed = ref(false);
const suggestions = ["К чему подать?", "При какой температуре пить?", "Чем отличается от соседей по линейке?"];

async function ask(text: string) {
  const q = text.trim();
  if (!q || busy.value || !online.value) return;
  question.value = "";
  history.value.push({ role: "user", content: q });
  busy.value = true;
  failed.value = false;
  try {
    const response = await fetch("/api/sommelier", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ item_id: props.slug, question: q, history: history.value.slice(0, -1) }),
    });
    if (response.status === 404) {
      available.value = false;
      return;
    }
    if (!response.ok) throw new Error(String(response.status));
    const data = await response.json();
    history.value.push({ role: "assistant", content: data.answer });
    available.value = true;
  } catch {
    failed.value = true;
  } finally {
    busy.value = false;
  }
}

onMounted(async () => {
  try {
    const response = await fetch("/api/sommelier/status");
    available.value = response.ok && (await response.json()).available === true;
  } catch {
    available.value = false;
  }
});
</script>

<template>
  <section v-if="available" class="card">
    <h3 style="font-size: 18px">Цифровой сомелье</h3>
    <p class="small muted" style="margin: 4px 0 12px">Спросите о вине — ответим по его карточке</p>
    <div class="chat">
      <div v-for="(m, i) in history" :key="i" class="bubble" :class="m.role === 'user' ? 'q' : 'a'">{{ m.content }}</div>
      <div v-if="busy" class="bubble a pulse">думаю…</div>
      <div v-if="failed" class="error small">Подсказки сейчас недоступны — попробуйте позже.</div>
    </div>
    <div v-if="!history.length" style="margin: 4px 0 10px">
      <button v-for="s in suggestions" :key="s" class="chip" :disabled="!online || busy" style="margin: 0 6px 6px 0" @click="ask(s)">{{ s }}</button>
    </div>
    <form class="input" @submit.prevent="ask(question)">
      <input v-model="question" placeholder="Ваш вопрос о вине" :disabled="busy || !online" />
      <button type="submit" :disabled="busy || !online || !question.trim()" aria-label="Отправить вопрос">→</button>
    </form>
  </section>
</template>
