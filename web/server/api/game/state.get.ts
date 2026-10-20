// Состояние винодельни: таймеры досчитываются на сервере, заказы и задания обновляются при чтении.
import { clientState, readGame, requireGamer, unseenRewards } from "../../utils/game";

export default defineEventHandler(async (event) => {
  setResponseHeader(event, "Cache-Control", "no-store");
  const user = await requireGamer(event);
  const { state, now, speed } = await readGame(user.id);
  return { state: clientState(state), now, speed, rewards: await unseenRewards(user.id) };
});
