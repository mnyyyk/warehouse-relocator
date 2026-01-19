from sqlmodel import SQLModel, create_engine
from sqlalchemy import event
from sqlalchemy.pool import QueuePool
import os

# Default to SQLite for local/dev to avoid accidental Postgres connections before .env loads
DATABASE_URL = os.getenv("DATABASE_URL") or "sqlite:///./dev.db"

# PostgreSQL用の接続プール設定
pool_kwargs = {}
if DATABASE_URL.startswith("postgresql"):
    pool_kwargs = {
        "poolclass": QueuePool,
        "pool_size": 10,           # 通常の接続数
        "max_overflow": 20,        # 追加で作成可能な接続数
        "pool_timeout": 60,        # 接続取得のタイムアウト（秒）
        "pool_recycle": 1800,      # 接続の再利用時間（30分）
        "pool_pre_ping": True,     # 接続確認を有効化
    }

engine = create_engine(
    DATABASE_URL.replace("+asyncpg", ""),  # Alembic は sync エンジンで OK
    echo=False,
    **pool_kwargs,
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