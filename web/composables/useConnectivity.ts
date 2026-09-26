export function useConnectivity() {
  return { online: useState<boolean>("connection.online", () => true) };
}
