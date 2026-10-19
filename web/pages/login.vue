<script setup lang="ts">
// Вход по телефону (2b–2e): номер → код → имя и 18+ (только при первом входе) → «Добро
// пожаловать». СМС-провайдер пока не подключён: сервер возвращает код, и мы показываем его здесь.
// После входа полка устройства сливается с аккаунтом, а отложенный отзыв публикуется.
import { formatPhone } from "~/utils/wine.mjs";
import { plural } from "~/utils/wine.mjs";
const route = useRoute();
const auth = useAuth();
const shelf = useShelf();
const reviews = useReviews();
const { goal } = useMetrika();

type Step = "phone" | "code" | "profile" | "welcome";
const step = ref<Step>("phone");
const phone = ref("");
const code = ref("");
const demoCode = ref<string | null>(null);
const name = ref("");
const adult = ref(false);
const busy = ref(false);
const error = ref("");
const help = ref(false);
const resendAt = ref(0);
const now = ref(Date.now());
const merged = ref(0);
const publishedWine = ref<{ slug: string; name: string } | null>(null);
const codeInput = ref<HTMLInputElement>();
let clock: ReturnType<typeof setInterval> | undefined;

const redirect = computed(() => {
  const value = typeof route.query.redirect === "string" ? route.query.redirect : "";
  return value.startsWith("/") && !value.startsWith("//") ? value : "/profile";
});
const phoneOk = computed(() => /^9\d{9}$/.test(phone.value));
const nameOk = computed(() => name.value.trim().length > 0);
const wait = computed(() => Math.max(0, Math.ceil((resendAt.value - now.value) / 1000)));
const waitLabel = computed(() => `${Math.floor(wait.value / 60)}:${String(wait.value % 60).padStart(2, "0")}`);
const progress = computed(() => ({ phone: 1, code: 2, profile: 3, welcome: 3 })[step.value]);

function onPhone(e: Event) {
  let d = (e.target as HTMLInputElement).value.replace(/\D/g, "");
  if (d.length > 10 && (d[0] === "7" || d[0] === "8")) d = d.slice(1);
  phone.value = d.slice(0, 10);
  (e.target as HTMLInputElement).value = formatPhone(phone.value);
}
function onCode(e: Event) {
  code.value = (e.target as HTMLInputElement).value.replace(/\D/g, "").slice(0, 4);
  (e.target as HTMLInputElement).value = code.value;
  if (code.value.length === 4) void verify("typed");
}
function fillDemo() {
  code.value = demoCode.value!;
  void verify("demo");
}
function toggleHelp() {
  help.value = !help.value;
  if (help.value) goal("auth_code_help");
}

async function sendCode() {
  if (!phoneOk.value || busy.value) return;
  const resend = step.value === "code";
  busy.value = true;
  error.value = "";
  try {
    const r = await auth.requestCode(phone.value);
    goal(resend ? "auth_code_resend" : "auth_code_request", { demo: Boolean(r.demoCode) });
    demoCode.value = r.demoCode;
    resendAt.value = Date.now() + r.retryIn * 1000;
    code.value = "";
    step.value = "code";
    await nextTick();
    codeInput.value?.focus();
  } catch (e) {
    const retryIn = (e as { data?: { data?: { retryIn?: number } } }).data?.data?.retryIn;
    if (retryIn && demoCode.value) { step.value = "code"; resendAt.value = Date.now() + retryIn * 1000; }
    else {
      error.value = apiError(e, "Не удалось отправить код. Проверьте сеть.");
      goal("auth_error", { step: "phone" });
    }
  } finally {
    busy.value = false;
  }
}

async function verify(via: "typed" | "demo") {
  if (busy.value) return;
  busy.value = true;
  error.value = "";
  try {
    const r = await auth.verify(phone.value, code.value);
    goal("auth_login", { new_user: r.isNew, via });
    if (!auth.complete.value) {
      name.value = auth.user.value?.name || "";
      adult.value = auth.user.value?.adult || false;
      step.value = "profile";
    } else await finish();
  } catch (e) {
    error.value = apiError(e, "Не удалось проверить код. Проверьте сеть.");
    goal("auth_error", { step: "code" });
    code.value = "";
    if (codeInput.value) codeInput.value.value = "";
  } finally {
    busy.value = false;
  }
}

async function saveProfile() {
  if (!nameOk.value || !adult.value || busy.value) return;
  busy.value = true;
  error.value = "";
  try {
    await auth.updateProfile({ name: name.value, adult: true });
    goal("auth_profile_complete");
    await finish();
  } catch (e) {
    error.value = apiError(e);
    goal("auth_error", { step: "profile" });
  } finally {
    busy.value = false;
  }
}

async function finish() {
  merged.value = shelf.entries.value.length;
  await shelf.syncNow();
  const pending = reviews.takePending();
  if (pending && shelf.find(pending.slug)?.reaction) {
    try {
      await reviews.publish(pending.slug, pending.anonymous);
      publishedWine.value = { slug: pending.slug, name: String(shelf.find(pending.slug)?.card.name || "") };
    } catch { /* отзыв остался на полке — опубликовать можно из карточки */ }
  }
  step.value = "welcome";
  goal("auth_welcome", { merged: merged.value, review_published: Boolean(publishedWine.value), redirect: redirect.value });
}

function back() {
  if (step.value === "code") { step.value = "phone"; error.value = ""; }
  else navigateTo(redirect.value === "/profile" ? "/" : redirect.value);
}

onMounted(() => {
  clock = setInterval(() => (now.value = Date.now()), 1000);
  watch(() => auth.ready.value, (ready) => {
    if (!ready || !auth.user.value || step.value !== "phone") return;
    if (auth.complete.value) navigateTo(redirect.value, { replace: true });
    else { name.value = auth.user.value.name; step.value = "profile"; }
  }, { immediate: true });
});
onBeforeUnmount(() => clearInterval(clock));
useHead({ title: "Вход — Своё Вино" });
</script>

<template>
  <main class="page" style="min-height: 100dvh">
    <div v-if="step !== 'welcome'" class="navbar">
      <button v-if="step !== 'profile'" class="icon-btn" aria-label="Назад" @click="back"><i class="ph ph-arrow-left" /></button>
      <div v-else style="width: 44px; height: 44px" />
      <div class="progress" aria-hidden="true"><span v-for="i in 3" :key="i" :class="{ on: i <= progress }" /></div>
      <div style="width: 44px" />
    </div>

    <p v-if="!auth.ready.value" class="muted">Загружаем…</p>
    <section v-else-if="!auth.enabled.value" class="card cream stack" style="align-items: center; text-align: center">
      <Mascot :size="72" mood="meh" />
      <h1 class="serif-title">Вход пока недоступен</h1>
      <p class="muted">На этом сервере аккаунты не настроены. Сканер и полка работают без входа.</p>
      <NuxtLink class="btn btn-primary" to="/">К сканеру</NuxtLink>
    </section>

    <!-- 2b: номер -->
    <template v-else-if="step === 'phone'">
      <div class="stack" style="gap: 8px; padding-top: 12px">
        <h1 class="page-title">Ваш номер телефона</h1>
        <p class="lead">Пришлём код для входа. Номер не увидят другие пользователи</p>
      </div>
      <form class="stack" style="flex: 1" @submit.prevent="sendCode">
        <label class="field" :class="{ ok: phoneOk }" style="min-height: 60px">
          <span style="font-size: 20px; font-weight: 600">+7</span>
          <input type="tel" inputmode="numeric" autocomplete="tel-national" placeholder="(000) 000-00-00" aria-label="Номер телефона" style="font-size: 20px; letter-spacing: 0.02em" :value="formatPhone(phone)" autofocus @input="onPhone" />
          <i v-if="phoneOk" class="ph-fill ph-check-circle" style="font-size: 22px; color: var(--ok)" />
        </label>
        <p v-if="error" class="error" role="alert">{{ error }}</p>
        <div style="flex: 1" />
        <p class="hint" style="text-align: center; color: var(--text-2); line-height: 1.45">Продолжая, вы принимаете условия сервиса и политику обработки данных</p>
        <button class="btn btn-primary" type="submit" :disabled="!phoneOk || busy"><span v-if="busy" class="spinner light" />Получить код</button>
      </form>
    </template>

    <!-- 2c: код -->
    <template v-else-if="step === 'code'">
      <div class="stack" style="gap: 8px; padding-top: 12px">
        <h1 class="page-title">Введите код</h1>
        <p class="lead">Для +7 {{ formatPhone(phone) }} · <button class="link" style="font-size: 15px" @click="step = 'phone'">изменить</button></p>
      </div>
      <div v-if="demoCode" class="demo-code" role="status">
        <i class="ph ph-chat-centered-dots" style="font-size: 26px; color: var(--sand-ink)" />
        <div class="grow"><div class="small" style="color: var(--sand-ink)">Демо-режим: СМС не отправляется, ваш код</div><b>{{ demoCode }}</b></div>
        <button class="chip" @click="fillDemo">Ввести</button>
      </div>
      <label class="code-boxes">
        <span v-for="i in 4" :key="i" :class="{ active: i - 1 === code.length }">{{ code[i - 1] || "" }}</span>
        <input ref="codeInput" type="text" inputmode="numeric" autocomplete="one-time-code" maxlength="4" aria-label="Код из 4 цифр" :value="code" @input="onCode" />
      </label>
      <div style="display: flex; justify-content: center; min-height: 24px">
        <span v-if="busy" class="row" style="gap: 8px; font-size: 14px; color: var(--ok); font-weight: 600"><span class="spinner ok" />Проверяем код</span>
        <span v-else-if="wait" class="small muted" style="font-size: 14px">Отправить снова через {{ waitLabel }}</span>
        <button v-else class="link" @click="sendCode">Отправить код ещё раз</button>
      </div>
      <p v-if="error" class="error" role="alert">{{ error }}</p>
      <div style="flex: 1" />
      <div style="text-align: center">
        <button class="link" style="padding: 10px" @click="toggleHelp">Не приходит код?</button>
        <p v-if="help" class="small muted" style="margin-top: 4px">СМС-провайдер пока не подключён — код показан на экране выше.</p>
      </div>
    </template>

    <!-- 2d: имя и возраст -->
    <template v-else-if="step === 'profile'">
      <div class="stack" style="gap: 8px; padding-top: 12px">
        <h1 class="page-title">Как вас подписать?</h1>
        <p class="lead">Имя увидят рядом с вашими отзывами</p>
      </div>
      <form class="stack" style="gap: 16px; flex: 1" @submit.prevent="saveProfile">
        <label class="field"><input v-model="name" maxlength="40" placeholder="Имя" aria-label="Имя" autocomplete="given-name" /></label>
        <div class="stack" style="gap: 6px">
          <div class="label">Так будет выглядеть отзыв</div>
          <div class="row" style="gap: 10px; border: 1px solid var(--line); border-radius: 20px; padding: 14px; background: var(--bg-cream)">
            <span class="avatar" style="width: 34px; height: 34px; font-size: 14px">{{ nameOk ? name.trim()[0].toUpperCase() : "?" }}</span>
            <div class="grow"><div style="font-size: 14px; font-weight: 600">{{ nameOk ? name.trim() : "Имя" }}</div><div class="small muted" style="font-size: 12px">Ваш отзыв · сегодня</div></div>
            <span class="badge liked"><i class="ph-fill ph-thumbs-up" />Понравилось</span>
          </div>
        </div>
        <button type="button" class="check-row" role="checkbox" :aria-checked="adult" @click="adult = !adult">
          <span class="box" :class="{ on: adult }"><i class="ph-bold ph-check" /></span>
          <span style="font-size: 14px; line-height: 1.4"><b>Мне исполнилось 18 лет</b><span class="muted" style="display: block; font-size: 13px; margin-top: 2px">Сервис рассказывает об алкогольной продукции</span></span>
        </button>
        <p v-if="error" class="error" role="alert">{{ error }}</p>
        <div style="flex: 1" />
        <button class="btn btn-primary" type="submit" :disabled="!nameOk || !adult || busy"><span v-if="busy" class="spinner light" />Готово</button>
      </form>
    </template>

    <!-- 2e: добро пожаловать -->
    <section v-else class="stack" style="flex: 1; align-items: center; justify-content: center; text-align: center; gap: 16px">
      <div style="position: relative; display: flex; justify-content: center"><Confetti /><Mascot :size="110" mood="happy" /></div>
      <h1 class="serif" style="font-size: 28px; line-height: 1.2">Добро пожаловать{{ auth.user.value?.name ? `, ${auth.user.value.name}` : "" }}!</h1>
      <div class="stack" style="width: 100%; gap: 8px; margin-top: 8px">
        <div class="done-row"><i class="ph-fill ph-check-circle" /><div class="grow"><b>Полка {{ merged ? "перенесена в аккаунт" : "подключена к аккаунту" }}</b><div class="small muted">{{ merged ? `${merged} ${plural(merged, ["вино", "вина", "вин"])} с этого устройства` : "Новые вина сохранятся и на других устройствах" }}</div></div></div>
        <div v-if="publishedWine" class="done-row"><i class="ph-fill ph-check-circle" /><div class="grow"><b>Отзыв опубликован</b><div class="small muted">{{ publishedWine.name }}</div></div></div>
      </div>
      <NuxtLink class="btn btn-primary" style="margin-top: 12px" :to="redirect" replace>{{ publishedWine ? "Вернуться к вину" : "Продолжить" }}</NuxtLink>
    </section>
  </main>
</template>
