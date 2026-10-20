// Винордл дня. Без входа можно играть: попытки хранит браузер и присылает их в POST, награды
// и серия — только у вошедших. Ответ раскрывается лишь после конца игры.
import { WORDLE_WINES, wordleNumber, wordleView } from "~/utils/wordle.mjs";
import { dayKey } from "~/utils/game.mjs";
import { currentUser } from "../../utils/auth";
import { dbEnabled, useDb } from "../../utils/db";
import { wordleStats } from "../../utils/game";

export default defineEventHandler(async (event) => {
  setResponseHeader(event, "Cache-Control", "no-store");
  const now = Date.now();
  const number = wordleNumber(now);
  const options = WORDLE_WINES.map((w) => ({ id: w.id, name: w.name, winery: w.winery }));
  const user = dbEnabled() ? await currentUser(event).catch(() => null) : null;
  if (!user?.adult) return { guest: true, options, view: wordleView(number, []), stats: null };
  const sql = await useDb();
  const day = dayKey(now);
  const [row] = await sql<{ guesses: number[] }[]>`select guesses from wordle_attempts where user_id = ${user.id} and day = ${day}`;
  const [game] = await sql<{ state: { wordle?: { streak: number } } }[]>`select state from game_profiles where user_id = ${user.id}`;
  return { guest: false, options, view: wordleView(number, row?.guesses || []), stats: await wordleStats(day), streak: game?.state?.wordle?.streak || 0 };
});

