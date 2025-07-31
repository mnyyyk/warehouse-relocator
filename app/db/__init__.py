"""
Shared database objects and helpers for the warehouse‑relocator app.

Usage
-----
1.  Call :pyfunc:`init_db` once at application start‑up to ensure all tables
    defined in *app/db/models.py* have been created.
2.  Import :pyfunc:`get_session` as a FastAPI dependency whenever you need a
    database session, e.g.

        @router.get("/items")
        def read_items(session: Session = Depends(get_session)):
            return session.exec(select(Item)).all()

The actual table / model definitions live in *app/db/models.py* (created in the
next step of the guide).  They are imported lazily inside :pyfunc:`init_db`
to avoid circular‑import problems. These models inherit from Base (SQLAlchemy declarative_base).
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel, Session

# We use SQLModel as the declarative base so that all table‑classes that
# inherit from `SQLModel` share a single metadata collection.
Base = SQLModel

###############################################################################
# Configuration
###############################################################################

# Default location for the development database (SQLite).
_DEFAULT_SQLITE_PATH = Path(__file__).resolve().parents[2] / "data" / "warehouse_relocator.db"

# Ensure the database directory exists so that SQLite can create the file
_DEFAULT_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)

# Connection URL can be overridden with an env var so we can point to Postgres
# (or any other RDBMS) in staging / production.
DATABASE_URL: str = os.getenv("WARELOC_DB_URL", f"sqlite:///{_DEFAULT_SQLITE_PATH}")

# Extra kwargs are required when using SQLite so that the same connection can
# be shared across threads in the dev server.
_engine_kwargs: dict[str, object] = {"echo": False}
if DATABASE_URL.startswith("sqlite"):
    _engine_kwargs["connect_args"] = {"check_same_thread": False}


engine = create_engine(DATABASE_URL, **_engine_kwargs)

# ---------------------------------------------------------------------------
# Session factory (for dependency‑injected DB sessions)
# ---------------------------------------------------------------------------
# Return `sqlmodel.Session` instances so that `.exec()` is available
SessionLocal = sessionmaker(class_=Session, autocommit=False, autoflush=False, bind=engine)


###############################################################################
# Helper API
###############################################################################

def init_db() -> None:
    """
    Create all tables that are declared in *app/db/models.py*.

    Importing models here ensures their metadata is registered before we call
    :pyfunc:`Base.metadata.create_all`.
    """
    # Lazy imports to avoid circular dependencies if models import this module.
    # Import ALL model modules that declare SQLModel tables so their metadata
    # gets registered before we call `SQLModel.metadata.create_all`.
    from . import models  # noqa: F401  pylint: disable=unused-import
    from . import models_analysis  # noqa: F401  pylint: disable=unused-import

    SQLModel.metadata.create_all(bind=engine)


@contextmanager
def get_session() -> Session:
    """
    Context manager / FastAPI dependency that yields a SQLAlchemy
    :class:`Session` produced by :data:`SessionLocal`.

    Examples
    --------
    >>> with get_session() as sess:
    ...     sess.exec(select(MyModel)).all()
    """
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -----------------------------------------------------------------------
# Backwards compatibility alias
# -----------------------------------------------------------------------
# Older modules import `get_db`; keep an alias to avoid churn.
get_db = get_session


__all__ = ["engine", "SessionLocal", "Base", "init_db", "get_session", "get_db"]