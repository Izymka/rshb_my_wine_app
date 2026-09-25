// Уведомления отправляет только сервер Nuxt. Ошибки Telegram не влияют на ответы посетителям.
type Transport = typeof fetch;

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

  photo(requestId: string, filename: string, data: Buffer, type?: string): Promise<void> {
    return this.enqueue(async () => {
      const body = new FormData();
      const caption = `Новое фото для распознавания · задача ${requestId}: ${filename}`;
      body.set("caption", caption.slice(0, 1024));
      const isPhoto = ["image/jpeg", "image/png"].includes(type || "") && data.length <= 10_000_000;
      const method = isPhoto ? "sendPhoto" : "sendDocument";
      body.set(isPhoto ? "photo" : "document", new Blob([data], { type }), filename);
      await this.post(method, body);
    });
  }

  document(caption: string, filename: string, content: string | Promise<string>): Promise<void> {
    return this.enqueue(async () => {
      const body = new FormData();
      body.set("caption", caption.slice(0, 1024));
      body.set("document", new Blob([await content], { type: "text/plain; charset=utf-8" }), filename);
      await this.post("sendDocument", body);
    });
  }
}

export const telegramNotifier = new TelegramNotifier(
  process.env.TG_NOTIFY_TOKEN?.trim() || "",
  process.env.TG_NOTIFY_TARGET?.trim() || "",
);
