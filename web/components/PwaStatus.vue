<script setup lang="ts">
interface InstallEvent extends Event {
  prompt(): Promise<void>;
  userChoice: Promise<{ outcome: "accepted" | "dismissed" }>;
}
const { online } = useConnectivity();
const restored = ref(false);
const visible = ref(false);
const ios = ref(false);
const help = ref(false);
const installing = ref(false);
const installError = ref(false);
let deferred: InstallEvent | null = null;
let timer: ReturnType<typeof setTimeout> | undefined;
let displayMode: MediaQueryList | undefined;
const dismissKey = "wine-install-dismissed";
const standalone = () => displayMode?.matches || (navigator as Navigator & { standalone?: boolean }).standalone;

function dismissed() {
  try { return Date.now() - Number(localStorage.getItem(dismissKey) || 0) < 14 * 86400000; }
  catch { return false; }
}
function updateConnection() {
  const wasOffline = !online.value;
  online.value = navigator.onLine;
  restored.value = wasOffline && online.value;
  clearTimeout(timer);
  if (restored.value) timer = setTimeout(() => (restored.value = false), 4000);
}
function offerInstall(event: Event) {
  if (standalone() || dismissed()) return;
  event.preventDefault();
  deferred = event as InstallEvent;
  visible.value = true;
}
function installed() {
  visible.value = false;
  deferred = null;
}
function modeChanged() { if (standalone()) installed(); }
function dismiss() {
  visible.value = false;
  help.value = false;
  try { localStorage.setItem(dismissKey, String(Date.now())); } catch {}
}
async function install() {
  if (ios.value) { help.value = !help.value; return; }
  if (!deferred || installing.value) return;
  installing.value = true;
  installError.value = false;
  try {
    await deferred.prompt();
    await deferred.userChoice;
    visible.value = false;
  } catch {
    installError.value = true;
  } finally {
    deferred = null;
    installing.value = false;
  }
}
onMounted(() => {
  online.value = navigator.onLine;
  displayMode = window.matchMedia("(display-mode: standalone)");
  ios.value = /iphone|ipad|ipod/i.test(navigator.userAgent)
    || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
  visible.value = ios.value && !standalone() && !dismissed();
  window.addEventListener("online", updateConnection);
  window.addEventListener("offline", updateConnection);
  window.addEventListener("beforeinstallprompt", offerInstall);
  window.addEventListener("appinstalled", installed);
  displayMode.addEventListener("change", modeChanged);
});
onBeforeUnmount(() => {
  clearTimeout(timer);
  window.removeEventListener("online", updateConnection);
  window.removeEventListener("offline", updateConnection);
  window.removeEventListener("beforeinstallprompt", offerInstall);
  window.removeEventListener("appinstalled", installed);
  displayMode?.removeEventListener("change", modeChanged);
});
</script>

<template>
  <div class="pwa-notices">
    <div aria-live="polite" aria-atomic="true" role="status">
      <div v-if="!online || restored" class="connection-notice" :class="{ restored }">
        <span class="connection-dot" aria-hidden="true" />
        <div>
          <strong>{{ online ? "Подключение восстановлено" : "Нет подключения к интернету" }}</strong>
          <p v-if="!online">Для распознавания этикеток и ответов сомелье нужна сеть. Продолжите, когда связь появится.</p>
        </div>
      </div>
    </div>
    <section v-if="visible && online" class="install-notice" aria-label="Установка приложения">
      <img src="/icon.png" width="40" height="40" alt="" />
      <div class="grow">
        <strong>Своё Вино всегда под рукой</strong>
        <p v-if="help">В Safari нажмите «Поделиться», затем «На экран “Домой”» и «Добавить».</p>
        <p v-else-if="installError" role="alert">Откройте меню браузера и выберите установку приложения.</p>
        <p v-else>Добавьте сканер на главный экран устройства.</p>
        <button v-if="!installError" class="install-action" :disabled="installing" @click="install">
          {{ ios ? (help ? "Понятно" : "Как установить") : "Установить приложение" }}
        </button>
      </div>
      <button class="notice-close" aria-label="Скрыть предложение установки" @click="dismiss">×</button>
    </section>
  </div>
</template>
