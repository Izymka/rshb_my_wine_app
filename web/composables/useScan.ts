// Состояние одного сканирования, общее для экранов: снятый кадр, ответ сервиса, ошибка.
// Живёт в useState, чтобы переход камера -> карточка не терял результат.

export interface Card {
  name?: string;
  winery?: string;
  category?: string;
  color?: string;
  region?: string;
  grapes?: string;
  description?: string;
  slug?: string;
  vintage?: number | string | null;
  [key: string]: unknown;
}

export interface Candidate {
  item_id: string;
  probability: number;
  card: Card;
}

export interface Analogue {
  slug: string;
  name: string;
  winery: string;
  category?: string;
  region?: string;
  grapes?: string;
  score: number;
}

export interface ScanResponse {
  request_id: string;
  answered: boolean;
  probability: number;
  item_id: string | null;
  card: Card | null;
  candidates: Candidate[];
  recognized_text: string;
  vintage: { year: number | null; source: string; ask: boolean } | null;
  confidence: { top1: number; top5: number; margin: number } | null;
  guard: string | null;
  judge: Record<string, unknown> | null;
  analogues: Analogue[] | null;
  timings_ms: Record<string, number>;
}

export type ScanError = "unreadable" | "network" | "unavailable" | "starting" | null;

// Последний отправленный кадр — для кнопки «Повторить» после сбоя сервиса или сети.
// Держим вне useState: File не сериализуется, а сканирование идёт только в браузере.
let lastFile: File | null = null;

export function useScan() {
  const preview = useState<string | null>("scan.preview", () => null);
  const result = useState<ScanResponse | null>("scan.result", () => null);
  const error = useState<ScanError>("scan.error", () => null);
  const busy = useState<boolean>("scan.busy", () => false);
  const elapsedMs = useState<number>("scan.elapsed", () => 0);
  let controller: AbortController | null = null;

  async function scan(file: File): Promise<boolean> {
    reset();
    lastFile = file;
    if (!navigator.onLine) {
      error.value = "network";
      return false;
    }
    preview.value = URL.createObjectURL(file);
    busy.value = true;
    controller = new AbortController();
    const form = new FormData();
    form.append("files", file, file.name || "frame.jpg");
    const started = performance.now();
    try {
      const response = await fetch("/api/scan", { method: "POST", body: form, signal: controller.signal });
      elapsedMs.value = Math.round(performance.now() - started);
      if (response.status === 400) {
        error.value = "unreadable";
        return false;
      }
      if (response.status === 503) {
        error.value = "starting";
        return false;
      }
      if (!response.ok) {
        error.value = "unavailable";
        return false;
      }
      result.value = (await response.json()) as ScanResponse;
      lastFile = null;
      return true;
    } catch (e) {
      if ((e as Error).name === "AbortError") return false;
      error.value = "network";
      return false;
    } finally {
      busy.value = false;
      controller = null;
    }
  }

  function cancel() {
    controller?.abort();
    busy.value = false;
  }

  // Повторная отправка того же кадра; false, если повторять нечего.
  function retry(): Promise<boolean> {
    return lastFile ? scan(lastFile) : Promise.resolve(false);
  }

  function canRetry(): boolean {
    return lastFile !== null;
  }

  function reset() {
    result.value = null;
    error.value = null;
    elapsedMs.value = 0;
  }

  return { preview, result, error, busy, elapsedMs, scan, retry, canRetry, cancel, reset };
}

export function imageUrl(slug: string | undefined | null): string | undefined {
  return slug ? `/api/catalog/image/${encodeURIComponent(slug)}` : undefined;
}
