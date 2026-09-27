<script setup lang="ts">
// Экран 1 (камера) и экран 2 (ожидание) из DESIGN.md. Камера — системная: <input capture>
// открывает её на телефоне без разрешений и без своего видоискателя, а на компьютере
// превращается в выбор файла. После ответа — переход на карточку или отказ.
const { preview, error, busy, elapsedMs, scan, retry, canRetry, cancel, reset } = useScan();
const router = useRouter();
const { online } = useConnectivity();
const cameraInput = ref<HTMLInputElement>();
const galleryInput = ref<HTMLInputElement>();
const slow = ref(false);
let slowTimer: ReturnType<typeof setTimeout> | null = null;

// Сбой сервиса или сети — тот же кадр можно отправить ещё раз; нечитаемый файл — нет.
const retryable = computed(
  () => error.value !== null && error.value !== "unreadable" && canRetry(),
);

async function run(send: () => Promise<boolean>) {
  slow.value = false;
  slowTimer = setTimeout(() => (slow.value = true), 3000);
  const ok = await send();
  if (slowTimer) clearTimeout(slowTimer);
  if (ok) router.push("/result");
}

async function onFile(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  input.value = "";
  if (!file) return;
  await run(() => scan(file));
}

function onRetry() {
  run(retry);
}

onMounted(reset);
</script>

<template>
  <div class="page">
    <header class="topbar">
      <img src="/icon.png" alt="" />
      <div>
        <div class="brand">Своё Вино</div>
        <div class="sub">сканер этикеток российских вин</div>
      </div>
    </header>

    <NuxtLink class="shelf-nav" to="/wines">Мои вина →</NuxtLink>

    <div class="viewfinder">
      <img v-if="preview" :src="preview" alt="снятый кадр" :class="{ pulse: busy }" />
      <div v-else class="placeholder">
        <div class="glyph">🍷</div>
        <div>Наведите камеру на этикетку</div>
        <div class="small muted" style="margin-top: 6px">Бутылка — главная в кадре, этикетка целиком</div>
      </div>
      <div class="frame"><i /></div>
      <div v-if="busy" class="hint row" style="justify-content: center">
        <span class="spinner" style="width: 18px; height: 18px; border-width: 2px" />
        <span>{{ slow ? "Ещё ищем — сравниваем с каталогом…" : "Ищем в каталоге…" }}</span>
      </div>
    </div>

    <div v-if="error" class="error">
      <template v-if="error === 'unreadable'">Файл не похож на фотографию. Попробуйте снять ещё раз.</template>
      <template v-else-if="error === 'network'">Нет связи с сервисом. Проверьте сеть и повторите.</template>
      <template v-else-if="error === 'starting'">Сервис ещё запускается — подождите полминуты.</template>
      <template v-else>Сервис временно недоступен. Попробуйте чуть позже.</template>
      <button v-if="retryable" class="btn btn-secondary error-retry" :disabled="!online" @click="onRetry">
        Повторить запрос
      </button>
    </div>

    <template v-if="!busy">
      <button class="btn btn-primary" :disabled="!online" @click="cameraInput?.click()">
        <span>📷</span> Сканировать этикетку
      </button>
      <button class="btn btn-secondary" :disabled="!online" @click="galleryInput?.click()">Выбрать из галереи</button>
      <input ref="cameraInput" type="file" accept="image/*" capture="environment" hidden @change="onFile" />
      <input ref="galleryInput" type="file" accept="image/*,.heic" hidden @change="onFile" />
    </template>
    <button v-else class="btn btn-ghost" @click="cancel">Отменить</button>

    <p v-if="error" class="small muted" style="text-align: center; margin: 0">
      <a href="#" @click.prevent="reset">Скрыть сообщение</a>
    </p>

    <div class="card cream small">
      <b>Как снимать.</b> Сфотографируйте этикетку на бутылке, можно взять бутылку в руки. Блики и наклон не мешают.
    </div>

    <div class="footer">каталог платформы «Своё Вино» · {{ elapsedMs ? `последний ответ ${elapsedMs} мс` : "" }}</div>
  </div>
</template>
