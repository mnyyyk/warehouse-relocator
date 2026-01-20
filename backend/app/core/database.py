from sqlmodel import SQLModel, create_engine
from sqlalchemy import event
from sqlalchemy.pool import QueuePool, NullPool
import os

# Default to SQLite for local/dev to avoid accidental Postgres connections before .env loads
DATABASE_URL = os.getenv("DATABASE_URL") or "sqlite:///./dev.db"

# エンジン作成時のオプション
engine_kwargs = {
    "echo": False,
}

# PostgreSQL用の接続プール設定
if DATABASE_URL.startswith("postgresql"):
    engine_kwargs.update({
        "poolclass": QueuePool,
        "pool_size": 10,           # 通常の接続数
        "max_overflow": 20,        # 追加で作成可能な接続数
        "pool_timeout": 30,        # 接続取得のタイムアウト（秒）
        "pool_recycle": 1800,      # 接続の再利用時間（30分）
        "pool_pre_ping": False,    # 起動時の接続テストを無効化（ヘルスチェック高速化）
        "connect_args": {
            "connect_timeout": 3,   # DB接続タイムアウト（秒）- 起動高速化のため短く
        },
    })

engine = create_engine(
    DATABASE_URL.replace("+asyncpg", ""),  # Alembic は sync エンジンで OK
    **engine_kwargs,
)

# SQLite 環境でのデフォルトPRAGMA（接続時に一度だけ適用）
if DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _sqlite_on_connect(dbapi_connection, connection_record):  # type: ignore
        try:
            cursor = dbapi_connection.cursor()
            # WAL で同時実行性とスループット改善
            cursor.execute("PRAGMA journal_mode=WAL")
            # 安定性と速度のバランス（bulk時は個別に OFF をかける実装あり）
            cursor.execute("PRAGMA synchronous=NORMAL")
            # 長めの busy_timeout（ミリ秒）
            cursor.execute("PRAGMA busy_timeout=30000")
            # temp をメモリに
            cursor.execute("PRAGMA temp_store=MEMORY")
            cursor.close()
        except Exception:
            # PRAGMA 失敗は無視（他DBやドライバ差異に備える）
            pass