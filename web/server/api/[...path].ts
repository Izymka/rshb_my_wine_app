// Прокси в ML-сервис: /api/scan -> {scannerUrl}/scan, /api/catalog/image/x -> .../catalog/image/x.
// Multipart уходит потоком как есть; тело ответа возвращается без изменений.
import { proxyRequest } from "h3";

export default defineEventHandler(async (event) => {
  // event.path сохраняет query string и кодирование пути.
  const path = event.path.slice("/api/".length);
  const base = useRuntimeConfig(event).scannerUrl.replace(/\/+$/, "");
  const target = `${base}/${path}`;
  return proxyRequest(event, target);
});
