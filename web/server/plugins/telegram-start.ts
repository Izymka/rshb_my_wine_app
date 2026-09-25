import { telegramNotifier } from "../utils/telegram-notify";

export default defineNitroPlugin(() => {
  void telegramNotifier.htmlMessage("🟢 <b>Веб-сервер запущен</b>");
});
