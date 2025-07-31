

"""
Location master table – native SQLAlchemy 2 model.
Slot (ロケーション) の物理座標を保持するマスター。
"""

from __future__ import annotations

from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class Location(Base):
    """
    倉庫ロケーション（棚番）マスタ。

    * slot_code … 「A‑01‑01‑1」など人可読の棚番
    * aisle     … 通路番号（例: 1, 2, 3…）
    * bay       … 列（奥行）番号
    * level     … 段(レベル)
    """

    __tablename__ = "locations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    slot_code: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    aisle: Mapped[int] = mapped_column(Integer)
    bay: Mapped[int] = mapped_column(Integer)
    level: Mapped[int] = mapped_column(Integer)

    def __repr__(self) -> str:
        return (
            f"<Location id={self.id} slot={self.slot_code} "
            f"aisle={self.aisle} bay={self.bay} level={self.level}>"
        )