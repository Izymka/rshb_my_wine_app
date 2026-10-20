// Генерирует PNG-иконки PWA и заставки iOS из public/icon.svg. Запуск (macOS): node scripts/build-icons.mjs
// Результат коммитится в public/, при сборке приложения скрипт не нужен.
import { execFileSync } from "node:child_process";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { SPLASH_SCREENS } from "./splash-screens.mjs";

const BACKGROUND = "#fefdfa"; // = background_color манифеста и --bg приложения
const publicDir = fileURLToPath(new URL("../public/", import.meta.url));
const mark = (await readFile(join(publicDir, "icon.svg"), "utf8"))
  .replace(/^[\s\S]*?<svg[^>]*>/, "").replace(/<\/svg>\s*$/, "").replace(/<!--[\s\S]*?-->/g, "");

// Знак размером size (в пикселях) с центром в (cx, cy) на холсте width×height.
function compose(width, height, size, cx, cy, background) {
  const scale = size / 512;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`
    + (background ? `<rect width="${width}" height="${height}" fill="${background}"/>` : "")
    + `<g transform="translate(${cx - size / 2} ${cy - size / 2}) scale(${scale})">${mark}</g></svg>`;
}

const jobs = [
  ["icon.png", 192, 192, compose(192, 192, 192, 96, 96)],
  ["icon-512.png", 512, 512, compose(512, 512, 512, 256, 256)],
  // Безопасная зона maskable — круг диаметром 80%: знак уменьшен до 62% и залит фоном до краёв.
  ["icon-maskable-512.png", 512, 512, compose(512, 512, 318, 256, 256, BACKGROUND)],
  ["apple-touch-icon.png", 180, 180, compose(180, 180, 144, 90, 90, BACKGROUND)],
  ...SPLASH_SCREENS.map(([w, h, ratio]) => {
    const width = w * ratio, height = h * ratio;
    return [`splash/${width}x${height}.png`, width, height,
      compose(width, height, Math.round(width * 0.42), width / 2, height * 0.46, BACKGROUND)];
  }),
];

const work = await mkdtemp(join(tmpdir(), "wine-icons-"));
const args = [];
for (const [name, width, height, svg] of jobs) {
  const source = join(work, `${args.length}.svg`);
  await writeFile(source, svg);
  args.push(source, join(publicDir, name), String(width), String(height));
}
execFileSync("swift", [fileURLToPath(new URL("./rasterize-svg.swift", import.meta.url)), ...args], { stdio: "inherit" });
console.log(`Иконки: ${jobs.map(([name]) => name).join(", ")}`);
