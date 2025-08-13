from sqlmodel import SQLModel, Field
from typing import Optional
from sqlalchemy import Column, Float, Numeric, BigInteger

class Sku(SQLModel, table=True):
    __tablename__ = "sku"

    sku_id: str = Field(primary_key=True)
    # Inner‑pack quantity (was part of composite PK) — keep as regular column
    pack_qty: int = Field(
        default=1,
        sa_column=Column(BigInteger)
    )
    length_mm: Optional[float] = Field(default=None, sa_column=Column(Float))
    width_mm: Optional[float] = Field(default=None, sa_column=Column(Float))
    height_mm: Optional[float] = Field(default=None, sa_column=Column(Float))
    volume_m3: Optional[float] = Field(
        default=None,
        sa_column=Column(Numeric(12, 8, asdecimal=False)),  # e.g., 0.04479000
    )  # 商品予備項目006

    # --- analysis metrics (populated by Celery tasks) -----------------------
    # 平均在庫日数（古さ）: 小数点付きで保持しておく
    oldness_days: Optional[float] = Field(
        default=None,
        sa_column=Column(Float),
        description="平均在庫日数 (日)",
    )
    # 直近 30 日間の回転率 = 出荷数 / 平均在庫数
    turnover_30d: Optional[float] = Field(
        default=None,
        sa_column=Column(Float),
        description="30日間の在庫回転率",
    )