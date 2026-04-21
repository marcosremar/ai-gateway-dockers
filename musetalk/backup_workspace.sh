#!/usr/bin/env bash
# Sincroniza /workspace pro B2 (Backblaze) ou qualquer S3-compatible.
# Chamado pelo cron interno do container ou ad-hoc via SSH.
#
# Configuração via env vars (passadas via deployEnv ao subir o pod):
#   B2_ACCOUNT_ID, B2_APPLICATION_KEY, B2_BUCKET     — credenciais Backblaze
#   B2_ENDPOINT     (default: https://s3.us-west-004.backblazeb2.com)
#   B2_REGION       (default: us-west-004)
#   B2_PREFIX       (default: workspaces/<HOSTNAME>)
#   BACKUP_PATH     (default: /workspace)
#   BACKUP_LOG      (default: /var/log/workspace_backup.log)
#
# Sem B2_ACCOUNT_ID setado → no-op silencioso (não falha o container).

set -e

LOG="${BACKUP_LOG:-/var/log/workspace_backup.log}"
mkdir -p "$(dirname "$LOG")"

log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*" | tee -a "$LOG"; }

if [ -z "${B2_ACCOUNT_ID:-}" ] || [ -z "${B2_APPLICATION_KEY:-}" ] || [ -z "${B2_BUCKET:-}" ]; then
    log "B2 credentials not set (B2_ACCOUNT_ID/B2_APPLICATION_KEY/B2_BUCKET) — backup skipped"
    exit 0
fi

ENDPOINT="${B2_ENDPOINT:-https://s3.us-west-004.backblazeb2.com}"
REGION="${B2_REGION:-us-west-004}"
PREFIX="${B2_PREFIX:-workspaces/$(hostname)}"
SRC="${BACKUP_PATH:-/workspace}"

if [ ! -d "$SRC" ]; then
    log "BACKUP_PATH '$SRC' does not exist — nothing to back up"
    exit 0
fi

log "═══ Workspace backup → s3://${B2_BUCKET}/${PREFIX}/ ═══"
log "  source: $SRC ($(du -sh "$SRC" 2>/dev/null | cut -f1))"

# rclone usa ENV vars pra config — sem precisar arquivo /root/.config/rclone
export RCLONE_CONFIG_B2_TYPE=s3
export RCLONE_CONFIG_B2_PROVIDER=Other
export RCLONE_CONFIG_B2_ENDPOINT="$ENDPOINT"
export RCLONE_CONFIG_B2_ACCESS_KEY_ID="$B2_ACCOUNT_ID"
export RCLONE_CONFIG_B2_SECRET_ACCESS_KEY="$B2_APPLICATION_KEY"
export RCLONE_CONFIG_B2_REGION="$REGION"
export RCLONE_CONFIG_B2_NO_CHECK_BUCKET=true

START=$(date +%s)
# --transfers 8: paralelismo
# --checksum: usa md5 pra detectar mudanças (mais rápido que size+mtime nessa escala)
# --exclude: pula caches volumosos não-essenciais
if rclone sync "$SRC" "b2:${B2_BUCKET}/${PREFIX}" \
    --transfers 8 \
    --checksum \
    --exclude "**/_refined_frames/**" \
    --exclude "**/_prewarped/**" \
    --exclude "**/_blend/**" \
    --exclude "**/__pycache__/**" \
    --exclude "**/.cache/**" \
    --stats=30s --stats-one-line --log-file="$LOG" --log-level INFO; then
    DURATION=$(( $(date +%s) - START ))
    log "✓ Backup completed in ${DURATION}s"
    exit 0
else
    log "✗ Backup FAILED with rc=$?"
    exit 1
fi
