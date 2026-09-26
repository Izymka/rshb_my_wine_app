#!/usr/bin/env bash
# Окружение для start загружает Node из web/.env в Makefile.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE="$ROOT/web/.ngrok"

running() {
  [[ -f "$STATE/pid" ]] || return 1
  read -r pid < "$STATE/pid"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  # Не останавливаем посторонний процесс при повторном использовании PID.
  ps -p "$pid" -o comm= | grep -Eq '(^|/)ngrok$'
}

case "${1:-start}" in
  stop)
    if running; then
      kill "$pid"
      echo "Туннель остановлен."
    else
      echo "Туннель этого проекта не запущен."
    fi
    exit 0
    ;;
  start) ;;
  *) echo "Использование: $0 [start|stop]" >&2; exit 1 ;;
esac

command -v ngrok >/dev/null || { echo "Установите ngrok: https://ngrok.com/download" >&2; exit 1; }
if [[ -z "${NGROK_AUTHTOKEN:-}" ]]; then
  echo "Заполните NGROK_AUTHTOKEN в web/.env." >&2
  exit 1
fi
if ! mkdir "$STATE" 2>/dev/null; then
  echo "Туннель уже запущен или остался $STATE после аварийного завершения."
  echo "Выполните make tunnel-stop; если процесс не запущен, удалите $STATE."
  exit 1
fi

cleanup() {
  if [[ -n "${pid:-}" ]]; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
  rm -f "$STATE/pid"
  rmdir "$STATE"
}
trap cleanup EXIT
trap 'exit 0' INT TERM

# Vite проверяет Host: передаём локальный адрес вместо публичного домена ngrok.
args=(http "http://127.0.0.1:${NITRO_PORT:-3000}" --host-header=rewrite)
if [[ -n "${NGROK_URL:-}" ]]; then
  args+=(--url "$NGROK_URL")
fi
echo "Публичный HTTPS-адрес появится в выводе ngrok. Инспектор: http://127.0.0.1:4040"
ngrok "${args[@]}" &
pid=$!
printf '%s\n' "$pid" > "$STATE/pid"
wait "$pid"
