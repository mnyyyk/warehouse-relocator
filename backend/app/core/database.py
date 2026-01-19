from sqlmodel import SQLModel, create_engine
from sqlalchemy import event
import os

# Default to SQLite for local/dev to avoid accidental Postgres connections before .env loads
DATABASE_URL = os.getenv("DATABASE_URL") or "sqlite:///./dev.db"

engine = create_engine(
    DATABASE_URL.replace("+asyncpg", ""),  # Alembic は sync エンジンで OK
    echo=False,
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