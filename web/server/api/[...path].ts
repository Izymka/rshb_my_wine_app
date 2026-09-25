// Прокси в ML-сервис: /api/scan -> {scannerUrl}/scan, /api/catalog/image/x -> .../catalog/image/x.
// Multipart и ответ проксируются без изменений; уведомления создаются только для web-запросов.
import { randomUUID } from "node:crypto";
import { getRequestHeader, proxyRequest, readMultipartFormData, readRawBody } from "h3";
import { telegramNotifier } from "../utils/telegram-notify";

export default defineEventHandler(async (event) => {
  // event.path сохраняет query string и кодирование пути.
  const path = event.path.slice("/api/".length);
  const base = useRuntimeConfig(event).scannerUrl.replace(/\/+$/, "");
  const target = `${base}/${path}`;
  const route = path.split("?", 1)[0];
  const isScan = event.method === "POST" && route === "scan";
  const isSommelier = event.method === "POST" && route === "sommelier";
  if (!telegramNotifier.enabled || (!isScan && !isSommelier)) {
    return proxyRequest(event, target);
  }

  const requestId = getRequestHeader(event, "x-request-id") || randomUUID().slice(0, 12);
  const started = performance.now();
  if (isScan) {
    const parts = await readMultipartFormData(event);
    for (const part of parts || []) {
      if (part.name === "files" && part.filename) {
        void telegramNotifier.photo(requestId, part.filename, part.data, part.type);
      }
    }
  } else {
    const body = await readRawBody(event);
    void telegramNotifier.message(`Запрос сомелье · задача ${requestId}\n${body || ""}`);
  }

  try {
    return await proxyRequest(event, target, {
      headers: { "x-request-id": requestId },
      onResponse: (_event, response) => {
        const elapsedMs = Math.round(performance.now() - started);
        const content = response.clone().text();
        if (isScan) {
          const caption = `Результат распознавания · задача ${requestId} · ${elapsedMs} мс · HTTP ${response.status}`;
          void telegramNotifier.document(caption, `scan-${requestId}.txt`, content);
        } else {
          void telegramNotifier.message(content.then(
            (text) => `Ответ сомелье · задача ${requestId} · HTTP ${response.status}\n${text}`,
          ));
        }
      },
    });
  } catch (error) {
    void telegramNotifier.message(`Ошибка web-прокси · задача ${requestId}: ${String(error)}`);
    throw error;
  }
});
