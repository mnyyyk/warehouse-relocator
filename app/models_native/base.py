"""
Project‑wide Declarative Base for native SQLAlchemy 2 models.

Any table model in `app.models_native.*` should inherit from `Base`.
"""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Root declarative base class (SQLAlchemy 2.x)."""
    pass
