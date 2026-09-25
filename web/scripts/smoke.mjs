// CI integration check: real Nitro server -> mock scanner, without models or PostgreSQL.
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { setTimeout } from "node:timers/promises";

if (process.argv[2] === "serve") {
  createServer(async (request, response) => {
    const chunks = [];
    for await (const chunk of request) chunks.push(chunk);
    const body = Buffer.concat(chunks).toString();
    if (request.url === "/prefix/catalog/image/wine%20one") {
      response.writeHead(200, { "content-type": "image/png" });
      response.end(Buffer.from([137, 80, 78, 71]));
      return;
    }
    const status = request.url === "/prefix/unavailable" ? 503 : 200;
    response.writeHead(status, { "content-type": "application/json", "x-request-id": "smoke" });
    response.end(JSON.stringify({
      path: request.url, method: request.method, body,
      contentType: request.headers["content-type"],
    }));
  }).listen(8080, "0.0.0.0");
} else {
  const base = process.argv[2] || "http://127.0.0.1:3000";
  let ready = false;
  for (let attempt = 0; attempt < 30; attempt++) {
    try {
      const response = await fetch(`${base}/api/health`, { signal: AbortSignal.timeout(2000) });
      if (response.ok && (await response.json()).path === "/prefix/health") {
        ready = true;
        break;
      }
    } catch {}
    await setTimeout(1000);
  }
  assert.ok(ready, "Nuxt and mock scanner must be ready");
  const page = await fetch(base);
  assert.equal(page.status, 200);
  const html = await page.text();
  assert.match(html, /Сфотографируйте/);
  assert.match(html, /<button/); // Actual SSR markup, not an empty SPA shell.
  assert.ok(!html.includes("scanner-mock"), "Private upstream must not reach the browser");
  assert.equal((await fetch(`${base}/result`)).status, 200);
  assert.equal((await fetch(`${base}/icon.png`)).status, 200);

  const form = new FormData();
  form.append("files", new Blob(["test-photo"], { type: "image/jpeg" }), "frame.jpg");
  const scan = await fetch(`${base}/api/scan?mode=a%20b`, { method: "POST", body: form });
  assert.equal(scan.status, 200);
  assert.equal(scan.headers.get("x-request-id"), "smoke");
  const uploaded = await scan.json();
  assert.equal(uploaded.path, "/prefix/scan?mode=a%20b");
  assert.equal(uploaded.method, "POST");
  assert.match(uploaded.contentType, /^multipart\/form-data; boundary=/);
  assert.match(uploaded.body, /test-photo/);
  assert.match(uploaded.body, /filename="frame.jpg"/);

  const question = JSON.stringify({ item_id: "wine-one", question: "К чему подать?" });
  const sommelier = await fetch(`${base}/api/sommelier`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: question,
  });
  assert.equal((await sommelier.json()).body, question);
  const status = await fetch(`${base}/api/sommelier/status`);
  assert.equal((await status.json()).path, "/prefix/sommelier/status");
  const image = await fetch(`${base}/api/catalog/image/wine%20one`);
  assert.equal(image.headers.get("content-type"), "image/png");
  assert.deepEqual([...new Uint8Array(await image.arrayBuffer())], [137, 80, 78, 71]);
  assert.equal((await fetch(`${base}/api/unavailable`)).status, 503);
  console.log("SSR, runtime scanner URL, multipart, JSON, images, query and error forwarding passed");
}
