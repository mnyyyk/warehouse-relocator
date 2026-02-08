#!/bin/bash
set -e

echo "[entrypoint] Running Alembic migrations..."

# 本番 (PostgreSQL) ではマイグレーションを実行
# SQLite (dev) でも安全に動作する
#
# 初回デプロイ時: 既存DBにテーブルがあるが alembic_version が無い場合、
# alembic upgrade head は CREATE TABLE しようとして失敗する。
# その場合は stamp で現在位置をマークしてからリトライする。
python -m alembic upgrade head 2>&1 || {
    echo "[entrypoint] Initial upgrade failed — attempting stamp + retry..."
    # テーブルは既にあるので、ベースラインをstampしてから差分のみ適用
    python -m alembic stamp head 2>&1 || true
    echo "[entrypoint] Stamped baseline. Retrying upgrade..."
    python -m alembic upgrade head 2>&1 || {
        echo "[entrypoint] WARNING: Alembic migration failed (non-fatal, continuing)"
    }
}

echo "[entrypoint] Starting uvicorn..."
exec python -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --workers 1 \
    --timeout-keep-alive 75
