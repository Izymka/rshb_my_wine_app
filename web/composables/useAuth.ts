// Аккаунт по телефону. Без аккаунта сканер и полка работают как раньше; вход нужен, чтобы
// публиковать отзывы и держать полку в облаке. Последний известный профиль кэшируется в
// localStorage, чтобы вкладка «Профиль» открывалась и без сети.
export interface User {
  id: string;
  phone: string;
  name: string;
  email: string | null;
  avatarUrl: string | null;
  adult: boolean;
  defaultShare: boolean;
  createdAt: string;
}

const CACHE_KEY = "wine-auth:user";

export function apiError(e: unknown, fallback = "Не получилось. Попробуйте ещё раз.") {
  const data = (e as { data?: { statusMessage?: string; message?: string } })?.data;
  const message = data?.statusMessage || data?.message;
  return message && /[а-яё]/i.test(message) ? message : fallback;
}

export function useAuth() {
  const user = useState<User | null>("auth.user", () => null);
  const enabled = useState("auth.enabled", () => false);
  const ready = useState("auth.ready", () => false);
  const complete = computed(() => Boolean(user.value?.name && user.value.adult));

  function remember(next: User | null) {
    user.value = next;
    try {
      if (next) localStorage.setItem(CACHE_KEY, JSON.stringify(next));
      else localStorage.removeItem(CACHE_KEY);
    } catch {}
  }

  async function refresh() {
    try {
      const r = await $fetch<{ enabled: boolean; user: User | null }>("/api/auth/me", { timeout: 10000 });
      enabled.value = r.enabled;
      remember(r.user);
    } catch {
      // Нет сети — показываем последний известный профиль; сервер проверит сессию позже.
      try {
        const cached = localStorage.getItem(CACHE_KEY);
        if (cached) { user.value = JSON.parse(cached); enabled.value = true; }
      } catch {}
    } finally {
      ready.value = true;
    }
  }

  async function requestCode(phone: string) {
    return await $fetch<{ phone: string; retryIn: number; demoCode: string | null }>("/api/auth/request-code", {
      method: "POST", body: { phone },
    });
  }

  async function verify(phone: string, code: string) {
    const r = await $fetch<{ user: User; isNew: boolean }>("/api/auth/verify", { method: "POST", body: { phone, code } });
    enabled.value = true;
    remember(r.user);
    return r;
  }

  async function updateProfile(patch: Partial<{ name: string; email: string; adult: boolean; defaultShare: boolean }>) {
    const r = await $fetch<{ user: User }>("/api/me/profile", { method: "PATCH", body: patch });
    remember(r.user);
    return r.user;
  }

  async function uploadAvatar(blob: Blob) {
    const form = new FormData();
    form.append("avatar", blob, "avatar.jpg");
    const r = await $fetch<{ user: User }>("/api/me/avatar", { method: "POST", body: form });
    remember(r.user);
  }

  async function removeAvatar() {
    const r = await $fetch<{ user: User }>("/api/me/avatar", { method: "DELETE" });
    remember(r.user);
  }

  async function logout() {
    await $fetch("/api/auth/logout", { method: "POST" }).catch(() => {});
    remember(null);
  }

  async function deleteAccount() {
    await $fetch("/api/me", { method: "DELETE" });
    remember(null);
  }

  return { user, enabled, ready, complete, refresh, requestCode, verify, updateProfile, uploadAvatar, removeAvatar, logout, deleteAccount };
}

// Квадратная JPEG-миниатюра 256×256 из выбранного фото: сервер принимает до 512 КБ.
export async function avatarBlob(file: File, size = 256): Promise<Blob> {
  const bitmap = await createImageBitmap(file);
  const side = Math.min(bitmap.width, bitmap.height);
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  canvas.getContext("2d")!.drawImage(bitmap, (bitmap.width - side) / 2, (bitmap.height - side) / 2, side, side, 0, 0, size, size);
  bitmap.close();
  return await new Promise((resolve, reject) => canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("encode"))), "image/jpeg", 0.85));
}
