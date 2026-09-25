import assert from "node:assert/strict";
import { test } from "node:test";

import { TelegramNotifier } from "../server/utils/telegram-notify.ts";

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
