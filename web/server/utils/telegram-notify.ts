// Уведомления отправляет только сервер Nuxt. Ошибки Telegram не влияют на ответы посетителям.
type Transport = typeof fetch;

export function escapeTelegramHtml(value: unknown): string {
  return String(value).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function displayText(value: unknown, limit = 100): string {
  return typeof value === "string" ? escapeTelegramHtml(value.slice(0, limit)) : "";
}

function percent(value: unknown): string | null {
  return typeof value === "number" && Number.isFinite(value)
    ? `${(value * 100).toFixed(2)}%`
    : null;
}

type ScanSummary = {
  request_id?: string;
  answered?: boolean;
  probability?: number;
  threshold?: number;
  card?: { name?: string; winery?: string } | null;
  candidates?: { probability?: number; card?: { name?: string; winery?: string } | null }[];
  recognized_text?: string;
  frames?: number;
  guard?: string | null;
};

export function formatScanCaption(requestId: string, status: number, elapsedMs: number, text: string): string {
  let result: ScanSummary;
  try {
    result = JSON.parse(text) as ScanSummary;
    if (!result || typeof result !== "object" || Array.isArray(result)) throw new Error("invalid JSON");
  } catch {
    result = {};
  }

  const ok = status >= 200 && status < 300;
  const title = !ok ? "❌ Ошибка распознавания" : result.answered === true
    ? "✅ Вино распознано" : result.answered === false
      ? "⚠️ Вино не распознано" : "⚠️ Нет результата распознавания";
  const lines = [`<b>${title}</b>`, `<b>Задача:</b> ${displayText(requestId)}`];
  if (typeof result.request_id === "string" && result.request_id !== requestId) {
    lines.push(`<b>ID scanner:</b> ${displayText(result.request_id)}`);
  }
  lines.push(`<b>Время ответа:</b> ${(elapsedMs / 1000).toFixed(2)} с`, `<b>HTTP:</b> ${status}`);

  if (ok) {
    const confidence = percent(result.probability);
    if (result.answered === true && typeof result.card?.name === "string") {
      lines.push(`<b>Вино:</b> ${displayText(result.card.name)}`);
      if (result.card.winery) lines.push(`<b>Винодельня:</b> ${displayText(result.card.winery)}`);
    }
    if (confidence) lines.push(`<b>Уверенность:</b> ${confidence}`);
    const threshold = percent(result.threshold);
    if (threshold) lines.push(`<b>Порог принятия:</b> ${threshold}`);
    if (result.answered === false && typeof result.candidates?.[0]?.card?.name === "string") {
      const best = result.candidates[0];
      lines.push(`<b>Ближайший кандидат:</b> ${displayText(best.card!.name)}`);
      const bestScore = percent(best.probability);
      if (bestScore && bestScore !== confidence) lines.push(`<b>Оценка кандидата:</b> ${bestScore}`);
    }
    if (result.recognized_text) {
      lines.push(`<b>Текст с этикетки:</b> ${displayText(result.recognized_text)}`);
    }
    if (typeof result.frames === "number") lines.push(`<b>Кадров:</b> ${result.frames}`);
    if (result.guard) lines.push(`<b>Причина отказа:</b> ${displayText(result.guard)}`);
  }
  const caption: string[] = [];
  for (const line of lines) {
    if (caption.join("\n").length + line.length + 1 <= 1000) caption.push(line);
  }
  return caption.join("\n");
}

export class TelegramNotifier {
  private pending: Promise<void> = Promise.resolve();
  private readonly token: string;
  private readonly target: string;
  private readonly transport: Transport;

  constructor(token: string, target: string, transport: Transport = fetch) {
    this.token = token;
    this.target = target;
    this.transport = transport;
  }

  get enabled(): boolean {
    return Boolean(this.token && this.target);
  }

  private enqueue(send: () => Promise<void>): Promise<void> {
    if (!this.enabled) return Promise.resolve();
    this.pending = this.pending.then(send).catch((error) => {
      console.error("Не удалось отправить уведомление Telegram", error);
    });
    return this.pending;
  }

  private async post(method: string, body: FormData): Promise<void> {
    body.set("chat_id", this.target);
    const response = await this.transport(`https://api.telegram.org/bot${this.token}/${method}`, {
      method: "POST",
      body,
      signal: AbortSignal.timeout(15000),
    });
    if (!response.ok) throw new Error(`Telegram ${method}: HTTP ${response.status}`);
    const result = await response.json() as { ok?: boolean };
    if (!result.ok) throw new Error(`Telegram ${method}: ответ без ok`);
  }

  message(content: string | Promise<string>): Promise<void> {
    return this.enqueue(async () => {
      const text = await content;
      for (let offset = 0; offset < text.length; offset += 4000) {
        const body = new FormData();
        body.set("text", text.slice(offset, offset + 4000));
        await this.post("sendMessage", body);
      }
    });
  }

  htmlMessage(content: string): Promise<void> {
    return this.enqueue(async () => {
      const body = new FormData();
      body.set("text", content);
      body.set("parse_mode", "HTML");
      await this.post("sendMessage", body);
    });
  }

  bodyMessage(title: string, details: string, content: string | Promise<string>): Promise<void> {
    return this.enqueue(async () => {
      const bodyText = await content;
      const chars = Array.from(bodyText);
      const chunks = chars.length ? Array.from(
        { length: Math.ceil(chars.length / 700) },
        (_, index) => chars.slice(index * 700, (index + 1) * 700).join(""),
      ) : [""];
      for (const chunk of chunks) {
        const body = new FormData();
        body.set("text", `<b>${escapeTelegramHtml(title)}</b>\n${details}\n${escapeTelegramHtml(chunk)}`);
        body.set("parse_mode", "HTML");
        await this.post("sendMessage", body);
      }
    });
  }

  photo(requestId: string, filename: string, data: Buffer, type?: string): Promise<void> {
    return this.enqueue(async () => {
      const body = new FormData();
      const caption = `📷 <b>Новое фото для распознавания</b>\n<b>Задача:</b> ${escapeTelegramHtml(requestId.slice(0, 60))}\n<b>Файл:</b> ${escapeTelegramHtml(filename.slice(0, 100))}`;
      body.set("caption", caption);
      body.set("parse_mode", "HTML");
      const isPhoto = ["image/jpeg", "image/png"].includes(type || "") && data.length <= 10_000_000;
      const method = isPhoto ? "sendPhoto" : "sendDocument";
      body.set(isPhoto ? "photo" : "document", new Blob([data], { type }), filename);
      await this.post(method, body);
    });
  }

  document(
    caption: string,
    filename: string,
    content: string | Promise<string>,
    formatCaption?: (text: string) => string,
  ): Promise<void> {
    return this.enqueue(async () => {
      const body = new FormData();
      const text = await content;
      body.set("caption", formatCaption ? formatCaption(text) : caption.slice(0, 1024));
      if (formatCaption) body.set("parse_mode", "HTML");
      body.set("document", new Blob([text], { type: "text/plain; charset=utf-8" }), filename);
      await this.post("sendDocument", body);
    });
  }
}

export const telegramNotifier = new TelegramNotifier(
  process.env.TG_NOTIFY_TOKEN?.trim() || "",
  process.env.TG_NOTIFY_TARGET?.trim() || "",
);
