<script setup lang="ts">
// Главная (1a) и распознавание (1b). Камера — системная: <input capture> открывает её на телефоне
// без разрешений и без своего видоискателя, а на компьютере превращается в выбор файла.
// Шаги распознавания идут по таймеру: сервис отвечает одним ответом, а видимый ход работы
// делает ожидание короче. Последний шаг закрывается только ответом сервиса.
import { winePath } from "~/utils/shelf.mjs";
const { preview, error, busy, scan, retry, canRetry, cancel, reset } = useScan();
const router = useRouter();
const { online } = useConnectivity();
const { entries } = useShelf();
const { goal } = useMetrika();
const cameraInput = ref<HTMLInputElement>();
const galleryInput = ref<HTMLInputElement>();
const step = ref(0);
const found = useState("scan.found", () => false);
let timers: ReturnType<typeof setTimeout>[] = [];

const steps = ["Нашли бутылку в кадре", "Читаем этикетку", "Сравниваем с каталогом"];
const recent = computed(() => entries.value.slice(0, 3));
const retryable = computed(() => error.value !== null && error.value !== "unreadable" && canRetry());

function clearTimers() { timers.forEach(clearTimeout); timers = []; }
async function run(send: () => Promise<boolean>) {
  clearTimers();
  step.value = 0;
  found.value = false;
  timers.push(setTimeout(() => (step.value = 1), 700), setTimeout(() => (step.value = 2), 1700));
  const ok = await send();
  clearTimers();
  if (!ok) return;
  step.value = 3;
  found.value = true;
  await new Promise((resolve) => setTimeout(resolve, 450));
  router.push("/result");
}

async function onFile(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  input.value = "";
  if (!file) return;
  goal("scan_start", { source: input === cameraInput.value ? "camera" : "gallery" });
  await run(() => scan(file));
}
function again() {
  goal("scan_retry", { reason: error.value });
  return run(retry);
}

function stop() { goal("scan_cancel", { step: step.value }); clearTimers(); cancel(); }
onMounted(() => { found.value = false; reset(); });
onBeforeUnmount(clearTimers);
</script>

<template>
  <!-- 1b: распознавание -->
  <div v-if="busy || found" class="page">
    <div class="viewfinder busy">
      <img v-if="preview" class="frame-photo" :src="preview" alt="ваш кадр" />
      <div class="corners"><span /><span /><span /><span /></div>
      <div class="scan-line" />
      <div class="frame-label"><i class="ph ph-image" />Ваш кадр</div>
    </div>
    <div class="row" style="gap: 14px; padding: 4px 4px 0">
      <Mascot :size="64" mood="wow" />
      <div class="grow">
        <div class="serif" style="font-size: 22px; line-height: 1.2">{{ found ? "Нашли! Открываем карточку" : "Ищем в каталоге…" }}</div>
        <div class="small muted" style="margin-top: 4px">Обычно это 2–3 секунды</div>
      </div>
    </div>
    <div class="steps" style="padding: 0 4px" role="status" aria-live="polite">
      <div v-for="(label, i) in steps" :key="label" :class="{ pending: i > step }">
        <span class="mark">
          <span v-if="i === step" class="spinner" />
          <i v-else-if="i < step" class="ph-fill ph-check-circle" />
        </span>
        {{ label }}
      </div>
    </div>
    <div style="margin-top: auto">
      <button v-if="!found" class="btn btn-outline" @click="stop">Отменить</button>
    </div>
  </div>

  <!-- 1a: главная -->
  <div v-else class="page with-tabs">
    <header class="topbar">
      <img class="logo" src="/icon.png" alt="" />
      <div class="grow">
        <div class="brand">Своё Вино</div>
        <div class="sub">сканер этикеток российских вин</div>
      </div>
      <span class="pill" :class="online ? 'online' : 'offline'"><span class="dot" />{{ online ? "онлайн" : "без сети" }}</span>
    </header>

    <div class="viewfinder idle">
      <div class="corners"><span /><span /><span /><span /></div>
      <div class="scan-line" />
      <div class="bubble-say">{{ online ? "Наведите на этикетку — я найду вино в каталоге" : "Без сети не найду вино, но полка работает" }}</div>
      <Mascot :size="84" :mood="online ? 'happy' : 'meh'" />
    </div>

    <div v-if="error" class="error" role="alert">
      <template v-if="error === 'unreadable'">Файл не похож на фотографию. Попробуйте снять ещё раз.</template>
      <template v-else-if="error === 'network'">Нет связи с сервисом. Проверьте сеть и повторите.</template>
      <template v-else-if="error === 'starting'">Сервис ещё запускается — подождите полминуты.</template>
      <template v-else>Сервис временно недоступен. Попробуйте чуть позже.</template>
      <div class="row" style="margin-top: 10px">
        <button v-if="retryable" class="btn btn-primary error-retry" style="min-height: 44px; margin: 0" :disabled="!online" @click="again">Повторить запрос</button>
        <button class="btn btn-text" style="min-height: 44px; width: auto" @click="reset">Скрыть</button>
      </div>
    </div>

    <div class="stack" style="gap: 8px">
      <button class="btn btn-primary btn-big" :disabled="!online" @click="cameraInput?.click()"><i class="ph-bold ph-camera" />Сканировать этикетку</button>
      <button class="btn btn-text" :disabled="!online" @click="galleryInput?.click()"><i class="ph ph-images" style="font-size: 19px" />Выбрать из галереи</button>
      <input ref="cameraInput" type="file" accept="image/*" capture="environment" hidden @change="onFile" />
      <input ref="galleryInput" type="file" accept="image/*,.heic" hidden @change="onFile" />
    </div>

    <div class="tips">
      <div><i class="ph ph-frame-corners" />Этикетка целиком</div>
      <div><i class="ph ph-sun" />Блики не мешают</div>
      <div><i class="ph ph-hand" />Можно в руке</div>
    </div>

    <section v-if="recent.length" class="stack" style="padding-top: 4px">
      <div class="head-row"><span style="font-size: 15px; font-weight: 600">Недавно на полке</span><NuxtLink to="/wines" @click="goal('home_shelf_all')">Все {{ entries.length }}</NuxtLink></div>
      <div class="grid-3">
        <NuxtLink v-for="w in recent" :key="w.slug" class="mini-wine" :to="winePath(w.slug)" @click="goal('home_recent_open', { slug: w.slug })">
          <div style="height: 48px; display: flex; align-items: flex-end"><WineBottle :slug="w.slug" :card="w.card" :height="48" :label="false" /></div>
          <div class="name">{{ w.card.name }}</div>
        </NuxtLink>
      </div>
    </section>
  </div>
</template>
