"""
Outbound (出荷実績) native SQLAlchemy 2 model.

Excel 「出荷実績.xlsx」 から取り込む行を格納する。
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Date, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class Outbound(Base):
    """
    出荷レコード

    * sku_code    … SKU（item_internalid）
    * qty         … 出荷数 (item_shipquantity)
    * ship_date   … 出荷日 (trandate)
    """

    __tablename__ = "outbounds"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    sku_code: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    qty: Mapped[int] = mapped_column(Integer, nullable=False)

    ship_date: Mapped[date] = mapped_column(Date, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Outbound id={self.id} sku={self.sku_code} qty={self.qty} date={self.ship_date}>"
