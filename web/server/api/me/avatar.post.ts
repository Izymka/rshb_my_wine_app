// Аватар: интерфейс уже ужал картинку до 256×256, здесь — проверка типа по сигнатуре и размера.
import { USER_COLUMNS, assertSameOrigin, requireUser, toUser } from "../../utils/auth";
import { useDb } from "../../utils/db";

const MAX_BYTES = 512 * 1024;

function imageType(data: Buffer) {
  if (data[0] === 0xff && data[1] === 0xd8 && data[2] === 0xff) return "image/jpeg";
  if (data.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]))) return "image/png";
  if (data.subarray(0, 4).toString() === "RIFF" && data.subarray(8, 12).toString() === "WEBP") return "image/webp";
  return null;
}

export default defineEventHandler(async (event) => {
  assertSameOrigin(event);
  const user = await requireUser(event);
  if (Number(getRequestHeader(event, "content-length") || 0) > MAX_BYTES + 4096) throw createError({ statusCode: 413 });
  const parts = await readMultipartFormData(event);
  const file = parts?.find((p) => p.name === "avatar");
  if (!file || file.data.length > MAX_BYTES) throw createError({ statusCode: 413, statusMessage: "Картинка больше 512 КБ" });
  const type = imageType(file.data);
  if (!type) throw createError({ statusCode: 400, statusMessage: "Нужна картинка JPEG, PNG или WebP" });
  const sql = await useDb();
  const [row] = await sql`
    update users set avatar = ${file.data}, avatar_type = ${type}, avatar_version = avatar_version + 1, updated_at = now()
    where id = ${user.id} returning ${sql.unsafe(USER_COLUMNS)}`;
  return { user: toUser(row as never) };
});
