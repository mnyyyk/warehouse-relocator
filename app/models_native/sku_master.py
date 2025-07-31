

"""
SKU master table – native SQLAlchemy 2 model.
Excel 「SKU.xlsx」 の各列をデータベースに正規化する。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Float, Integer, String, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class SKUMaster(Base):
    """
    製品マスタ

    * sku_code      … 商品ID
    * name          … 商品名称
    * length_mm     … 商品予備項目003（長さ mm）
    * width_mm      … 商品予備項目004（幅 mm）
    * height_mm     … 商品予備項目005（高さ mm）
    * volume_m3     … 商品予備項目006 × 入数（m³）
    * pack_qty      … 入数
    """

    __tablename__ = "sku_master"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    sku_code: Mapped[str] = mapped_column(String(32), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=True)

    length_mm: Mapped[float] = mapped_column(Float, nullable=True)
    width_mm: Mapped[float] = mapped_column(Float, nullable=True)
    height_mm: Mapped[float] = mapped_column(Float, nullable=True)

    pack_qty: Mapped[int] = mapped_column(Integer, nullable=True)
    volume_m3: Mapped[float] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<SKUMaster {self.sku_code} ({self.name})>"