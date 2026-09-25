// Тестовый перехват Telegram Bot API для запуска Nitro без реального токена.
const realFetch = globalThis.fetch;

globalThis.fetch = async (input, init) => {
  const url = String(input);
  if (!url.startsWith("https://api.telegram.org/bot")) return realFetch(input, init);

  const method = url.split("/").at(-1);
  const form = init?.body;
  const attachment = form?.get("photo") || form?.get("document");
  const record = {
    event: "telegram_mock",
    method,
    chatId: form?.get("chat_id"),
    text: form?.get("text"),
    caption: form?.get("caption"),
    filename: attachment?.name,
    content: attachment ? await attachment.text() : undefined,
  };
  console.info(JSON.stringify(record));
  return Response.json({ ok: true });
};
