

"""
InventorySnapshot – native SQLAlchemy 2 model.

Excel 「在庫データ.xlsx」 を取り込み、現在庫のスナップショットを保持する。
ロケーションコードは 6 桁の数値文字列 (level + column + depth) を前提に
level/column/depth を分解保存し、クエリ効率を高める。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Date, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, composite

from .base import Base


class InventorySnapshot(Base):
    """
    在庫スナップショット

    * sku_code   … SKU
    * loc_code   … 6 桁ロケーションコード (例: 101419)
    * qty        … 箱数
    * level      … 段 (1 = 地面に近い)
    * column     … 列 (入口側が若い)
    * depth      … 奥行き (入口側が若い)
    * lot        … ロット文字列
    * lot_date   … ロットから抽出した日付
    """

    __tablename__ = "inventory_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    sku_code: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    loc_code: Mapped[str] = mapped_column(String(6), index=True, nullable=False)
    qty: Mapped[int] = mapped_column(Integer, nullable=False)

    level: Mapped[int] = mapped_column(Integer, nullable=False)
    column: Mapped[int] = mapped_column("col", Integer, nullable=False)
    depth: Mapped[int] = mapped_column(Integer, nullable=False)

    lot: Mapped[str] = mapped_column(String(40), nullable=True)
    lot_date: Mapped[Date] = mapped_column(Date, nullable=True, index=True)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<InvSnap sku={self.sku_code} loc={self.loc_code} "
            f"qty={self.qty}>"
        )