

"""
Inbound (入荷履歴) native SQLAlchemy 2 model.

Excel 「入荷実績_*.xlsx」 の recordtype="発注" 行を取り込む想定。
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Date, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class Inbound(Base):
    """
    入荷レコード

    * sku_code    … SKU（item_internalid）
    * qty         … 入荷数
    * inbound_date… 入庫日 (trandate)
    * lot         … ロット文字列
    * lot_date    … ロットに含まれる日付 (JP 20241030 → 2024‑10‑30)
    """

    __tablename__ = "inbounds"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    sku_code: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    qty: Mapped[int] = mapped_column(Integer, nullable=False)

    inbound_date: Mapped[date] = mapped_column(Date, nullable=False)

    lot: Mapped[str] = mapped_column(String(40), nullable=True)
    lot_date: Mapped[date] = mapped_column(Date, nullable=True, index=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Inbound id={self.id} sku={self.sku_code} "
            f"qty={self.qty} date={self.inbound_date}>"
        )