from sqlmodel import SQLModel, create_engine
import os

DATABASE_URL = os.getenv("DATABASE_URL") or "postgresql+asyncpg://whuser:whpass@localhost:5432/warehouse"

engine = create_engine(
    DATABASE_URL.replace("+asyncpg", ""),  # Alembic は sync エンジンで OK
    echo=False,
)