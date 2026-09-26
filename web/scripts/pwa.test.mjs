import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { runInNewContext } from "node:vm";

const source = await readFile(new URL("./service-worker.js", import.meta.url), "utf8");
function worker({ offline = false, installFails = false } = {}) {
  const handlers = {};
  const removed = [];
  const shell = { body: "offline shell" };
  let claimed = false;
  runInNewContext(source.replace("/* __PRECACHE__ */ []", '["/", "/offline.html", "/_payload.json", "/_nuxt/app.js"]'), {
    URL,
    self: {
      location: { origin: "https://wine.example" },
      addEventListener: (name, handler) => { handlers[name] = handler; },
      clients: { claim: async () => { claimed = true; } },
    },
    caches: {
      open: async () => ({ addAll: async () => { if (installFails) throw Error("cache failed"); } }),
      keys: async () => ["wine-pwa-old", "wine-pwa-__CACHE_VERSION__", "other-app"],
      delete: async (name) => { removed.push(name); },
      match: async (path) => path === "/offline.html" ? shell : { body: "cached asset" },
    },
    fetch: async () => { if (offline) throw Error("offline"); return { body: "network" }; },
  });
  function fetchRequest(path, mode = "cors", method = "GET") {
    let response;
    handlers.fetch({ request: { url: `https://wine.example${path}`, mode, method }, respondWith: (value) => { response = value; } });
    return response;
  }
  return { handlers, removed, shell, fetchRequest, claimed: () => claimed };
}

test("offline navigation falls back to the application shell; payload and chunks stay cached", async () => {
  const w = worker({ offline: true });
  assert.equal(await w.fetchRequest("/result", "navigate"), w.shell);
  assert.equal((await w.fetchRequest("/_payload.json?_b=build")).body, "cached asset");
  assert.equal((await w.fetchRequest("/_nuxt/app.js")).body, "cached asset");
});

test("online navigation uses the server and API requests bypass the cache", async () => {
  const w = worker();
  assert.equal((await w.fetchRequest("/", "navigate")).body, "network");
  assert.equal(w.fetchRequest("/api/scan", "cors", "POST"), undefined);
  assert.equal(w.fetchRequest("/api/catalog/image/wine"), undefined);
  assert.equal(w.fetchRequest("/unknown.png"), undefined);
});

test("activation only removes older wine caches", async () => {
  const w = worker();
  let done;
  w.handlers.activate({ waitUntil: (value) => { done = value; } });
  await done;
  assert.deepEqual(w.removed, ["wine-pwa-old"]);
  assert.equal(w.claimed(), true);
});

test("failed precaching rejects installation, preserving the existing worker", async () => {
  const w = worker({ installFails: true });
  let done;
  w.handlers.install({ waitUntil: (value) => { done = value; } });
  await assert.rejects(done, /cache failed/);
});

test("push displays an actionable notification and a click opens the saved card", async () => {
  const handlers = {};
  const notifications = [];
  const opened = [];
  let closed = false;
  runInNewContext(source, { URL, self: {
    location: { origin: 'https://wine.example' },
    addEventListener: (name, handler) => { handlers[name] = handler; },
    registration: { showNotification: async (title, options) => { notifications.push({ title, options }); } },
    clients: { matchAll: async () => [], openWindow: async url => { opened.push(url); } },
  } });
  let done;
  handlers.push({ data: { json: () => ({ title: 'Как вам вино?', url: '/wines/test#review', tag: 'wine-test' }) }, waitUntil: p => { done = p; } });
  await done;
  assert.equal(notifications[0].options.data.url, '/wines/test#review');
  handlers.notificationclick({ notification: { data: notifications[0].options.data, close: () => { closed = true; } }, waitUntil: p => { done = p; } });
  await done;
  assert.equal(closed, true);
  assert.deepEqual(opened, ['https://wine.example/wines/test#review']);
  handlers.push({ data: { json: () => ({ url: 'https://evil.test' }) }, waitUntil: p => { done = p; } });
  await done;
  assert.equal(notifications[1].options.data.url, '/wines');
});
