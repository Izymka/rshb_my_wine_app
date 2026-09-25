import { randomUUID } from "node:crypto";

type ApiRequest = {
  id: string;
  method: string;
  path: string;
  startedAt: number;
};

const requests = new WeakMap<object, ApiRequest>();

export default defineNitroPlugin((nitroApp) => {
  nitroApp.hooks.hook("request", (event) => {
    const path = event.path.split("?", 1)[0];
    if (path !== "/api" && !path.startsWith("/api/")) return;

    const request = {
      id: randomUUID(),
      method: event.method,
      path,
      startedAt: Date.now(),
    };
    requests.set(event, request);
    console.info(JSON.stringify({
      event: "api_request",
      id: request.id,
      method: request.method,
      path: request.path,
    }));
  });

  nitroApp.hooks.hook("afterResponse", (event) => {
    const request = requests.get(event);
    if (!request) return;

    requests.delete(event);
    console.info(JSON.stringify({
      event: "api_response",
      id: request.id,
      method: request.method,
      path: request.path,
      status: event.node.res.statusCode,
      durationMs: Date.now() - request.startedAt,
    }));
  });
});
