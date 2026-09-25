// Прокси в ML-сервис: /api/scan -> {scannerUrl}/scan, /api/catalog/image/x -> .../catalog/image/x.
// Multipart и ответ проксируются без изменений; уведомления создаются только для web-запросов.
import { randomUUID } from "node:crypto";
import { getRequestHeader, proxyRequest, readMultipartFormData, readRawBody } from "h3";
import { escapeTelegramHtml, formatScanCaption, telegramNotifier } from "../utils/telegram-notify";

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
    void telegramNotifier.bodyMessage(
      "🍷 Запрос сомелье",
      `<b>Задача:</b> ${escapeTelegramHtml(requestId.slice(0, 100))}`,
      body || "",
    );
  }

  try {
    return await proxyRequest(event, target, {
      headers: { "x-request-id": requestId },
      onResponse: (_event, response) => {
        const elapsedMs = Math.round(performance.now() - started);
        const content = response.clone().text();
        if (isScan) {
          void telegramNotifier.document(
            "Результат распознавания",
            `scan-${requestId}.txt`,
            content,
            (text) => formatScanCaption(requestId, response.status, elapsedMs, text),
          );
        } else {
          const title = response.ok ? "✅ Ответ сомелье" : "❌ Ошибка сомелье";
          void telegramNotifier.bodyMessage(
            title,
            `<b>Задача:</b> ${escapeTelegramHtml(requestId.slice(0, 100))} · <b>HTTP:</b> ${response.status} · <b>Время ответа:</b> ${(elapsedMs / 1000).toFixed(2)} с`,
            content,
          );
        }
      },
    });
  } catch (error) {
    void telegramNotifier.htmlMessage(
      `❌ <b>Ошибка web-прокси</b>\n<b>Задача:</b> ${escapeTelegramHtml(requestId.slice(0, 100))}\n${escapeTelegramHtml(String(error).slice(0, 1000))}`,
    );
    throw error;
  }
});
