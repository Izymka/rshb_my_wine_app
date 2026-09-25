import { telegramNotifier } from "../utils/telegram-notify";

export default defineNitroPlugin(() => {
  void telegramNotifier.message("Веб-сервер запущен");
});
