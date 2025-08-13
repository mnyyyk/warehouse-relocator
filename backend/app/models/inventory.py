from __future__ import annotations
from typing import Optional

from sqlmodel import SQLModel, Field
from datetime import date
from sqlalchemy import Column, Date


class Inventory(SQLModel, table=True):
    __tablename__ = "inventory"

    # 在庫キー（複合主キー）
    location_id: str = Field(primary_key=True, index=True, description="ロケーションID")
    sku_id: str = Field(primary_key=True, index=True, description="商品ID")
    lot: str = Field(primary_key=True, index=True, description="ロット")
    # 入数スナップショット（取り込み時にSKUから引いた値）
    pack_qty: int = Field(primary_key=True, description="入数(スナップショット)")

    # ロットから正規化した日付（未判定はNULL）
    lot_date: Optional[date] = Field(
        default=None,
        sa_column=Column(Date, nullable=True),
        description="ロット正規化日付"
    )

    # 属性
    block_code: Optional[str] = Field(default=None, description="ブロック略称")

    # 数量（単位: 個）
    qty: int = Field(default=0, description="在庫数(引当数を含む)")

    # 位置（アップロード時にロケーションを分解して保存）
    level: Optional[int] = Field(default=None, index=True, description="段(1=最下段)")
    column: Optional[int] = Field(default=None, index=True, description="列(入口側が若番)")
    depth: Optional[int] = Field(default=None, index=True, description="奥行(入口側が若番)")

    # ケース数（qty/入数 の切り上げ等。取り込み時に算出して保存）
    cases: Optional[float] = Field(default=None, description="ケース数")

    # 品質区分名（良品/修理部品/出荷止め/一時仮置き/外装不良/販促物/什器/資材/廃棄予定 等）
    quality_name: Optional[str] = Field(default=None, description="品質区分名")