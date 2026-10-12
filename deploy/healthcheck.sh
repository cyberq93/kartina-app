#!/usr/bin/env bash
# Проверка живости сервиса с оповещением в Telegram.
# Запускается по таймеру раз в 5 минут (см. SETUP.md, шаг 9).
# Оповещает при смене состояния (упал / восстановился), а не каждые 5 минут.
#
# ВАЖНО: эта проверка работает на самом ПК и не заметит, что ПК выключен
# или пропал интернет. Для этого нужна ВНЕШНЯЯ проверка публичного адреса
# (UptimeRobot или аналог), см. SETUP.md.
set -u

ENV_FILE="${PBN_ENV_FILE:-/etc/pbn/pbn.env}"
STATE_FILE="/var/lib/pbn-health/state"
# shellcheck disable=SC1090
[ -r "$ENV_FILE" ] && . "$ENV_FILE"

URL="${PBN_HEALTH_URL:-http://127.0.0.1:8000/health}"
mkdir -p "$(dirname "$STATE_FILE")"
prev="$(cat "$STATE_FILE" 2>/dev/null || echo ok)"

if curl -fsS --max-time 10 "$URL" >/dev/null 2>&1; then
    now="ok"
else
    now="down"
fi

notify() {
    [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ] || return 0
    curl -fsS --max-time 10 \
        "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
        --data-urlencode "text=$1" >/dev/null 2>&1
}

if [ "$now" != "$prev" ]; then
    if [ "$now" = "down" ]; then
        notify "pbn_engine НЕ отвечает на $(hostname)"
    else
        notify "pbn_engine снова работает на $(hostname)"
    fi
fi
echo "$now" > "$STATE_FILE"
