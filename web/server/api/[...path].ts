// Прокси в ML-сервис: /api/scan -> {scannerUrl}/scan, /api/catalog/image/x -> .../catalog/image/x.
// Multipart уходит потоком как есть; тело ответа возвращается без изменений.
import { proxyRequest } from "h3";

export default defineEventHandler(async (event) => {
  const path = event.context.params?.path || "";
  const target = `${useRuntimeConfig().scannerUrl}/${path}`;
  return proxyRequest(event, target);
});
