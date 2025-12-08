#!/bin/bash
set -e

APP_UID=${APP_UID:-1000}
APP_GID=${APP_GID:-1000}

# Ensure group exists
if ! getent group "${APP_GID}" >/dev/null; then
  groupadd -g "${APP_GID}" appgroup
fi

# Ensure user exists
if ! id -u "${APP_UID}" >/dev/null 2>&1; then
  useradd -u "${APP_UID}" -g "${APP_GID}" -m appuser
fi

mkdir -p /app/logs/requests /app/results /tmp/mplcache /models /config /tmp
chown -R "${APP_UID}:${APP_GID}" /app /models /config /tmp

# If mounted volumes override ownership, fix only log/result dirs
chown -R "${APP_UID}:${APP_GID}" /app/logs /app/results /tmp/mplcache || true

export LOG_DIR=${LOG_DIR:-/app/logs}
export RESULTS_DIR=${RESULTS_DIR:-/app/results}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mplcache}

exec gosu "${APP_UID}:${APP_GID}" "$@"

