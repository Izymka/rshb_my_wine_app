import { shelfPushConfig, shelfPushEnabled } from '../../utils/shelf-push';
export default defineEventHandler((event) => {
  setResponseHeader(event, 'Cache-Control', 'no-store');
  return { publicKey: shelfPushEnabled() ? shelfPushConfig().publicKey : null };
});
