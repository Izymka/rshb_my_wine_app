import assert from "node:assert/strict";
import { test } from "node:test";

import { formatScanCaption, TelegramNotifier } from "../server/utils/telegram-notify.ts";

test("пустая настройка не отправляет запросы", async () => {
  let calls = 0;
  const notifier = new TelegramNotifier("", "123", async () => {
    calls++;
    return Response.json({ ok: true });
  });

  await notifier.message("старт");
  await notifier.photo("id", "label.png", Buffer.from("image"), "image/png");
  assert.equal(calls, 0);
});

test("сообщения и вложения отправляются в порядке задач", async () => {
  const sent = [];
  const notifier = new TelegramNotifier("test-token", "123", async (url, options) => {
    sent.push({ method: String(url).split("/").at(-1), body: options.body });
    return Response.json({ ok: true });
  });

  const first = notifier.message("Веб-сервер запущен");
  const second = notifier.photo("task-1", "label.png", Buffer.from("image"), "image/png");
  const third = notifier.document("Результат · task-1 · 123 мс", "scan-task-1.txt", '{"ok":true}');
  await Promise.all([first, second, third]);

  assert.deepEqual(sent.map((item) => item.method), ["sendMessage", "sendPhoto", "sendDocument"]);
  assert.ok(sent.every((item) => item.body.get("chat_id") === "123"));
  assert.equal(sent[1].body.get("photo").name, "label.png");
  assert.equal(await sent[1].body.get("photo").text(), "image");
  assert.equal(sent[2].body.get("document").name, "scan-task-1.txt");
  assert.equal(await sent[2].body.get("document").text(), '{"ok":true}');
});

test("ошибка Telegram не останавливает следующие задачи", async () => {
  let calls = 0;
  const originalError = console.error;
  console.error = () => {};
  try {
    const notifier = new TelegramNotifier("test-token", "123", async () => {
      calls++;
      return Response.json({ ok: calls > 1 }, { status: calls > 1 ? 200 : 503 });
    });
    await notifier.message("первое");
    await notifier.message("второе");
    assert.equal(calls, 2);
  } finally {
    console.error = originalError;
  }
});

test("сводка различает распознанное вино и неподтверждённого кандидата", () => {
  const unanswered = formatScanCaption("web-1", 200, 2574, JSON.stringify({
    request_id: "scanner-1", answered: false, probability: 0.007840576,
    threshold: 0.48, candidates: [{ probability: 0.007840576, card: { name: "Алиготе <Баррель>" } }],
    recognized_text: "fou.nS", frames: 1,
  }));
  assert.match(unanswered, /⚠️ Вино не распознано/);
  assert.match(unanswered, /Уверенность:<\/b> 0\.78%/);
  assert.match(unanswered, /Порог принятия:<\/b> 48\.00%/);
  assert.match(unanswered, /Ближайший кандидат:<\/b> Алиготе &lt;Баррель&gt;/);
  assert.match(unanswered, /Время ответа:<\/b> 2\.57 с/);
  assert.match(unanswered, /ID scanner:<\/b> scanner-1/);

  const answered = formatScanCaption("web-2", 200, 150, JSON.stringify({
    answered: true, probability: 0.91, card: { name: "Вино A&B", winery: "Винодельня" },
  }));
  assert.match(answered, /✅ Вино распознано/);
  assert.match(answered, /Вино:<\/b> Вино A&amp;B/);
  assert.doesNotMatch(answered, /Ближайший кандидат/);
  assert.match(formatScanCaption("web-3", 503, 1000, "unavailable"), /❌ Ошибка распознавания/);
});

test("форматированные уведомления передают Telegram HTML и экранируют текст", async () => {
  const sent = [];
  const notifier = new TelegramNotifier("test-token", "123", async (url, options) => {
    sent.push({ method: String(url).split("/").at(-1), body: options.body });
    return Response.json({ ok: true });
  });
  await notifier.htmlMessage("🟢 <b>Веб-сервер запущен</b>");
  await notifier.bodyMessage("🍷 Запрос сомелье", "<b>Задача:</b> 1", "Есть <вино> & сыр?");
  await notifier.document("unused", "scan-1.txt", '{"answered":false}',
    (text) => formatScanCaption("1", 200, 250, text));
  assert.deepEqual(sent.map((item) => item.method), ["sendMessage", "sendMessage", "sendDocument"]);
  assert.ok(sent.every((item) => item.body.get("parse_mode") === "HTML"));
  assert.match(sent[1].body.get("text"), /Есть &lt;вино&gt; &amp; сыр\?/);
  assert.match(sent[2].body.get("caption"), /⚠️ Вино не распознано/);
  assert.equal(await sent[2].body.get("document").text(), '{"answered":false}');
});
