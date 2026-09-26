// Freeze the complete application shell and cache version together on every build.
import { readdir, readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { join } from "node:path";

async function files(directory, prefix = "") {
  const entries = await readdir(directory, { withFileTypes: true });
  const nested = await Promise.all(entries.map((entry) => {
    const name = `${prefix}/${entry.name}`;
    return entry.isDirectory() ? files(join(directory, entry.name), name) : [name];
  }));
  return nested.flat();
}
export async function buildPwa(root) {
  const assets = (await files(root)).filter((path) =>
    path.startsWith("/_nuxt/") && !/\.(map|gz|br)$/.test(path)
  ).sort();
  assets.push("/offline.html", "/", "/_payload.json", "/icon.png", "/icon-512.png", "/logo.png", "/manifest.webmanifest");
  const hash = createHash("sha256");
  for (const asset of assets) hash.update(await readFile(join(root, asset === "/" ? "index.html" : asset)));
  const source = await readFile(new URL("./service-worker.js", import.meta.url), "utf8");
  hash.update(source);
  await writeFile(join(root, "sw.js"), source
    .replace("__CACHE_VERSION__", hash.digest("hex").slice(0, 16))
    .replace("/* __PRECACHE__ */ []", JSON.stringify(assets)));
  console.log(`PWA: cached ${assets.length} application assets`);
}
