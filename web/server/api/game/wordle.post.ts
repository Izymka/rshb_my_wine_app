// Попытка в «Вине дня». Для вошедших попытки копятся в wordle_attempts (одна строка на день), а
// конец игры начисляет Грозди и продлевает серию — в одной транзакции с состоянием винодельни.
import { WORDLE_TRIES, findWine, wordleNumber, wordleView } from "~/utils/wordle.mjs";
import { applyWordle, dayKey, wordleReward } from "~/utils/game.mjs";
import { assertSameOrigin, currentUser, rateLimit } from "../../utils/auth";
import { dbEnabled } from "../../utils/db";
import { clientState, withGame, wordleStats } from "../../utils/game";

export default defineEventHandler(async (event) => {
  setResponseHeader(event, "Cache-Control", "no-store");
  assertSameOrigin(event);
  const body = await readBody(event).catch(() => null);
  const guess = findWine(body?.guess);
  if (!guess) throw createError({ statusCode: 400, statusMessage: "Выберите вино из списка" });
  const now = Date.now();
  const number = wordleNumber(now);
  const user = dbEnabled() ? await currentUser(event).catch(() => null) : null;

  if (!user?.adult) {
    rateLimit(`wordle:${getRequestIP(event, { xForwardedFor: true }) || "guest"}`, 60, 60000);
    const prev = Array.isArray(body?.guesses) ? body.guesses.map(Number).filter((id: number) => findWine(id)).slice(0, WORDLE_TRIES) : [];
    const before = wordleView(number, prev);
    if (before.solved || before.lost) return { guest: true, view: before };
    if (prev.includes(guess.id)) throw createError({ statusCode: 409, statusMessage: "Это вино уже было" });
    return { guest: true, view: wordleView(number, [...prev, guess.id]) };
  }

  rateLimit(`wordle:${user.id}`, 60, 60000);
  const day = dayKey(now);
  const { result, state } = await withGame(user.id, async (state, { now: t, speed, tx }) => {
    const [row] = await tx<{ guesses: number[]; finished: boolean }[]>`
      select guesses, finished from wordle_attempts where user_id = ${user.id} and day = ${day} for update`;
    const prev = row?.guesses || [];
    if (row?.finished) return { view: wordleView(number, prev), reward: null };
    if (prev.includes(guess.id)) throw createError({ statusCode: 409, statusMessage: "Это вино уже было" });
    const guesses = [...prev, guess.id];
    const view = wordleView(number, guesses);
    const finished = view.solved || view.lost;
    await tx`
      insert into wordle_attempts (user_id, day, guesses, solved, finished) values (${user.id}, ${day}, ${guesses}, ${view.solved}, ${finished})
      on conflict (user_id, day) do update set guesses = excluded.guesses, solved = excluded.solved, finished = excluded.finished, updated_at = now()`;
    let reward = null;
    if (finished) {
      reward = applyWordle(state, day, view.attempt || 0, t, speed);
      await tx`insert into game_service_rewards (user_id, kind, ref, day, reward, seen)
        values (${user.id}, 'wordle', ${day}, ${day}, ${tx.json(view.attempt ? wordleReward(view.attempt) : {})}, true) on conflict do nothing`;
    }
    return { view, reward };
  }, { action: "wordle" });
  return { guest: false, ...result, streak: state.wordle?.streak || 0, stats: await wordleStats(day), state: clientState(state) };
});
