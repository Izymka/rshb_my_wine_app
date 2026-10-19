<script setup lang="ts">
// Профиль (2f) — третья вкладка. Поля аккаунта: телефон, аватар, имя, email. Здесь же
// «публиковать отзывы по умолчанию» для шторки отзыва, напоминания и состояние синхронизации.
import { winePath } from "~/utils/shelf.mjs";
import { initial, maskedPhone, plural, reviewDate } from "~/utils/wine.mjs";
const auth = useAuth();
const { user } = auth;
const shelf = useShelf();
const { online } = useConnectivity();
const { goal } = useMetrika();

interface MyReview { slug: string; liked: boolean; name: string; winery: string; updatedAt: string; anonymous: boolean }
const myReviews = ref<MyReview[] | null>(null);
const open = ref<"" | "reviews" | "reminders">("");
const editing = ref(false);
const form = reactive({ name: "", email: "" });
const busy = ref(false);
const error = ref("");
const confirmDelete = ref(false);
const avatarInput = ref<HTMLInputElement>();
const syncBusy = ref(false);

const liked = computed(() => shelf.entries.value.filter((e) => e.reaction === "liked").length);
const syncedLabel = computed(() => {
  if (!shelf.synced.value) return online.value ? "ещё не синхронизирована" : "без сети — синхронизируем позже";
  const minutes = Math.round((Date.now() - shelf.synced.value) / 60000);
  return minutes < 1 ? "обновлено только что" : `обновлено ${minutes} мин назад`;
});

async function loadReviews() {
  try { myReviews.value = (await $fetch<{ reviews: MyReview[] }>("/api/me/reviews")).reviews; }
  catch { myReviews.value = myReviews.value || null; }
}
watch(user, (u) => { if (u) void loadReviews(); else myReviews.value = null; }, { immediate: true });

function startEdit() {
  goal("profile_edit_open");
  form.name = user.value?.name || "";
  form.email = user.value?.email || "";
  error.value = "";
  editing.value = true;
}
async function saveEdit() {
  busy.value = true;
  error.value = "";
  try {
    await auth.updateProfile({ name: form.name, email: form.email });
    goal("profile_save", { email: Boolean(form.email.trim()) });
    editing.value = false;
  } catch (e) { error.value = apiError(e); }
  finally { busy.value = false; }
}
async function onAvatar(e: Event) {
  const input = e.target as HTMLInputElement;
  const file = input.files?.[0];
  input.value = "";
  if (!file) return;
  busy.value = true;
  error.value = "";
  try { await auth.uploadAvatar(await avatarBlob(file)); goal("profile_avatar_upload"); }
  catch (err) { error.value = apiError(err, "Не удалось загрузить фото. Попробуйте другое."); goal("profile_error", { action: "avatar_upload" }); }
  finally { busy.value = false; }
}
async function dropAvatar() {
  busy.value = true;
  try { await auth.removeAvatar(); goal("profile_avatar_remove"); } catch (e) { error.value = apiError(e); } finally { busy.value = false; }
}
async function toggleDefaultShare() {
  if (!user.value) return;
  try {
    const next = await auth.updateProfile({ defaultShare: !user.value.defaultShare });
    goal("profile_default_share", { on: next.defaultShare });
  } catch (e) { error.value = apiError(e); }
}
async function syncNow() {
  syncBusy.value = true;
  goal("shelf_sync_manual", { ok: await shelf.syncNow() });
  syncBusy.value = false;
}
function toggle(section: "reviews" | "reminders") {
  open.value = open.value === section ? "" : section;
  if (open.value) goal("profile_section_open", { section });
}
function startDelete() {
  confirmDelete.value = true;
  goal("account_delete_ask");
}
async function logout() {
  goal("auth_logout");
  await auth.logout();
  open.value = "";
}
async function deleteAccount() {
  busy.value = true;
  try { await auth.deleteAccount(); goal("account_delete"); confirmDelete.value = false; }
  catch (e) { error.value = apiError(e); }
  finally { busy.value = false; }
}
useHead({ title: "Профиль — Своё Вино" });
</script>

<template>
  <main class="page with-tabs">
    <p v-if="!auth.ready.value" class="muted" style="padding-top: 12px">Загружаем профиль…</p>

    <!-- Без входа -->
    <template v-else-if="!user">
      <h1 class="page-title" style="padding-top: 6px">Профиль</h1>
      <section class="card cream stack" style="align-items: center; text-align: center; gap: 12px">
        <Mascot :size="84" mood="happy" />
        <h2 class="serif-title">Войдите, чтобы делиться отзывами</h2>
        <p class="muted small" style="max-width: 300px">Сканер и полка работают без входа. Аккаунт нужен, чтобы публиковать отзывы и не потерять полку.</p>
        <NuxtLink v-if="auth.enabled.value" class="btn btn-primary" :to="{ path: '/login', query: { redirect: '/profile' } }" @click="goal('profile_login_click')"><i class="ph-bold ph-device-mobile" />Войти по номеру телефона</NuxtLink>
        <template v-if="auth.enabled.value">
          <div class="grid-2" style="width: 100%">
            <button class="btn btn-outline" disabled>Яндекс ID</button>
            <button class="btn btn-outline" disabled>VK ID</button>
          </div>
          <p class="hint" style="text-align: center">Вход через Яндекс ID и VK ID появится позже</p>
        </template>
        <p v-else class="small muted">На этом сервере аккаунты не настроены.</p>
      </section>
      <div class="menu">
        <NuxtLink to="/wines"><i class="ph ph-books" /><span class="grow">Мои вина</span><span class="muted small">{{ shelf.entries.value.length }}</span><i class="ph ph-caret-right caret" /></NuxtLink>
        <button :aria-expanded="open === 'reminders'" @click="toggle('reminders')"><i class="ph ph-bell" /><span class="grow">Напоминания оценить вино</span><i :class="['ph', open === 'reminders' ? 'ph-caret-down' : 'ph-caret-right', 'caret']" /></button>
        <div v-if="open === 'reminders'" class="panel"><ShelfPushSettings /></div>
      </div>
    </template>

    <!-- 2f -->
    <template v-else>
      <header class="row" style="gap: 14px; padding-top: 6px">
        <button class="avatar-edit" aria-label="Сменить фото профиля" :disabled="busy" @click="avatarInput?.click()">
          <img v-if="user.avatarUrl" class="avatar" :src="user.avatarUrl" alt="" width="64" height="64" style="width: 64px; height: 64px" />
          <span v-else class="avatar serif" style="width: 64px; height: 64px; font-size: 28px; font-weight: 500">{{ initial(user.name) }}</span>
          <span class="cam"><i class="ph ph-camera" /></span>
        </button>
        <input ref="avatarInput" type="file" accept="image/*" hidden @change="onAvatar" />
        <div class="grow">
          <div class="serif" style="font-size: 26px; line-height: 1.15; overflow-wrap: anywhere">{{ user.name || "Без имени" }}</div>
          <div class="small muted" style="margin-top: 2px">{{ maskedPhone(user.phone) }}</div>
          <div v-if="user.email" class="small muted">{{ user.email }}</div>
        </div>
        <button class="icon-btn" aria-label="Редактировать профиль" @click="editing ? (editing = false) : startEdit()"><i :class="['ph', editing ? 'ph-x' : 'ph-pencil-simple']" /></button>
      </header>

      <form v-if="editing" class="card stack" style="gap: 12px" @submit.prevent="saveEdit">
        <div class="stack" style="gap: 6px"><span class="label">Имя</span><label class="field" style="min-height: 52px"><input v-model="form.name" maxlength="40" autocomplete="given-name" style="font-size: 16px" /></label></div>
        <div class="stack" style="gap: 6px"><span class="label">Email — необязательно</span><label class="field" style="min-height: 52px"><input v-model="form.email" type="email" maxlength="120" autocomplete="email" placeholder="name@example.ru" style="font-size: 16px" /></label></div>
        <div class="stack" style="gap: 6px"><span class="label">Телефон</span><div class="muted small">{{ maskedPhone(user.phone) }} · меняется новым входом</div></div>
        <div class="row">
          <button type="button" class="btn btn-outline" style="min-height: 46px" @click="avatarInput?.click()"><i class="ph ph-camera-plus" />Фото</button>
          <button v-if="user.avatarUrl" type="button" class="btn btn-outline" style="min-height: 46px" @click="dropAvatar"><i class="ph ph-trash" />Убрать</button>
        </div>
        <button class="btn btn-primary" type="submit" :disabled="busy || !form.name.trim()">Сохранить</button>
      </form>
      <p v-if="error" class="error" role="alert">{{ error }}</p>

      <div class="grid-3">
        <NuxtLink to="/wines" class="tile" style="color: var(--text); padding: 12px"><div class="num" style="font-size: 24px">{{ shelf.entries.value.length }}</div><div class="cap">на полке</div></NuxtLink>
        <div class="tile" style="padding: 12px"><div class="num" style="font-size: 24px">{{ myReviews?.length ?? "—" }}</div><div class="cap">{{ plural(myReviews?.length || 0, ["отзыв", "отзыва", "отзывов"]) }}</div></div>
        <div class="tile" style="padding: 12px"><div class="num" style="font-size: 24px">{{ liked }}</div><div class="cap">{{ plural(liked, ["понравилось", "понравились", "понравились"]) }}</div></div>
      </div>

      <div class="menu">
        <button :aria-expanded="open === 'reviews'" @click="toggle('reviews')"><i class="ph ph-chats-circle" /><span class="grow">Мои отзывы</span><i :class="['ph', open === 'reviews' ? 'ph-caret-down' : 'ph-caret-right', 'caret']" /></button>
        <div v-if="open === 'reviews'" class="panel stack" style="gap: 8px">
          <p v-if="!myReviews?.length" class="small muted">Опубликованных отзывов пока нет. Оцените вино на полке и включите «Поделиться отзывом».</p>
          <NuxtLink v-for="r in myReviews" :key="r.slug" :to="winePath(r.slug)" class="row" style="gap: 10px; color: var(--text)">
            <span class="badge" :class="r.liked ? 'liked' : 'disliked'"><i :class="['ph-fill', r.liked ? 'ph-thumbs-up' : 'ph-thumbs-down']" /></span>
            <span class="grow"><b style="font-size: 14px">{{ r.name }}</b><span class="small muted" style="display: block">{{ r.winery }} · {{ reviewDate(r.updatedAt) }}{{ r.anonymous ? " · анонимно" : "" }}</span></span>
            <i class="ph ph-caret-right caret" />
          </NuxtLink>
        </div>
        <button role="switch" :aria-checked="user.defaultShare" @click="toggleDefaultShare">
          <i class="ph ph-users-three" /><span class="grow">Публиковать отзывы<span class="sub" style="display: block">по умолчанию при сохранении</span></span><span class="switch" :class="{ on: user.defaultShare }" />
        </button>
        <button :aria-expanded="open === 'reminders'" @click="toggle('reminders')"><i class="ph ph-bell" /><span class="grow">Напоминания оценить вино</span><i :class="['ph', open === 'reminders' ? 'ph-caret-down' : 'ph-caret-right', 'caret']" /></button>
        <div v-if="open === 'reminders'" class="panel"><ShelfPushSettings /></div>
        <button :disabled="syncBusy" @click="syncNow">
          <i class="ph ph-cloud-check" style="color: var(--ok)" /><span class="grow">Синхронизация<span class="sub" style="display: block">{{ syncBusy ? "синхронизируем…" : syncedLabel }}</span></span><i class="ph ph-arrows-clockwise caret" />
        </button>
      </div>

      <div class="stack" style="align-items: center; gap: 2px">
        <button class="btn btn-text" style="width: auto" @click="logout">Выйти</button>
        <button v-if="!confirmDelete" class="quiet-link" style="color: var(--text-3); font-size: 13px; padding: 6px" @click="startDelete">Удалить аккаунт</button>
        <div v-else class="card cream stack" style="width: 100%">
          <p>Удалить аккаунт, облачную полку и опубликованные отзывы? Полка на этом устройстве останется.</p>
          <div class="grid-2">
            <button class="btn btn-outline" @click="confirmDelete = false">Оставить</button>
            <button class="btn btn-primary" style="min-height: 50px" :disabled="busy" @click="deleteAccount">Удалить</button>
          </div>
        </div>
      </div>
    </template>
  </main>
</template>
