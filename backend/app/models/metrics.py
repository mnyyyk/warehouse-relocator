"""
metrics.py

SKU 毎の各種指標（古さ = Oldness, 回転日数 = Turn‑over など）を
計算し、DB テーブルへ保存するためのユーティリティ。

Step 11 以降で Celery タスクから呼び出して利用する。
"""
from __future__ import annotations

from datetime import datetime, date, timedelta
from typing import List, Dict

from sqlmodel import SQLModel, Field, Session, select, func
from sqlalchemy import literal

from app.models import Inventory, ShipTx


class SkuMetric(SQLModel, table=True):
    """
    SKU 単位で計算されたメトリクスを保持するテーブル。

    * ``oldness_days``  – 在庫最古ロット日から今日までの日数  
    * ``turnover_days`` – 現在庫量 ÷ 直近 *window_days* 出庫平均 (日)
    """
    sku_id: str = Field(primary_key=True, description="SKU ID")
    oldness_days: int | None = Field(default=None, description="最古ロット経過日数")
    turnover_days: float | None = Field(default=None, description="想定在庫回転日数")
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, nullable=False, description="算出日時"
    )


# --------------------------------------------------------------------------- #
# 計算ロジック                                                                 #
# --------------------------------------------------------------------------- #
def calculate_metrics(
    session: Session,
    *,
    window_days: int = 90,
) -> Dict[str, Dict[str, float | int | None]]:
    """
    DB 上の在庫・出庫実績を用いて SKU ごとの指標を計算して返す。

    戻り値は ``{sku_id: {"oldness_days": …, "turnover_days": …}}``。
    * window_days > 0 … 直近 window_days の出庫平均を使用
    * window_days <= 0 … 全期間の出庫を対象に、(今日 - 初回出庫日) で平均化
    * Inventory.lot_date が存在しない場合は oldness を None とする
    """
    today = date.today()
    window_start = (
        today - timedelta(days=window_days) if window_days and window_days > 0 else None
    )

    # ------ 1) oldness: 在庫最古ロット (lot_date が無い環境でも動くように保護) ------
    if hasattr(Inventory, "lot_date"):
        oldest_q = (
            select(
                Inventory.sku_id,
                func.min(Inventory.lot_date).label("oldest"),
            )
            .where(Inventory.lot_date.is_not(None))
            .group_by(Inventory.sku_id)
        )
    else:
        # lot_date が無い場合は None を返す（解析を落とさないためのフォールバック）
        oldest_q = select(
            Inventory.sku_id, literal(None).label("oldest")
        ).group_by(Inventory.sku_id)

    oldness_map: Dict[str, int | None] = {}
    for sku_id, oldest in session.exec(oldest_q):
        oldness_map[sku_id] = (today - oldest).days if oldest else None

    # ------ 2) turnover: 在庫 ÷ 平均出庫 ----------------------------------------------
    # 2‑a.  現在庫
    inv_q = select(
        Inventory.sku_id, func.sum(Inventory.qty).label("stock")
    ).group_by(Inventory.sku_id)
    stock_map = dict(session.exec(inv_q).all())

    turnover_map: Dict[str, float | None] = {}

    if window_start is not None:
        # 直近 window_days の出庫で平均を出す
        ship_q = (
            select(
                ShipTx.sku_id,
                func.sum(ShipTx.qty).label("shipped"),
            )
            .where(ShipTx.trandate >= window_start)
            .group_by(ShipTx.sku_id)
        )
        for sku_id, shipped in session.exec(ship_q):
            if shipped and shipped > 0 and window_days > 0:
                daily_avg = shipped / window_days
                stock = stock_map.get(sku_id, 0)
                turnover_map[sku_id] = round(stock / daily_avg, 2) if daily_avg else None
            else:
                turnover_map[sku_id] = None
    else:
        # 全期間: 初回出庫日から今日までで平均を出す
        ship_q = (
            select(
                ShipTx.sku_id,
                func.sum(ShipTx.qty).label("shipped"),
                func.min(ShipTx.trandate).label("first_date"),
            )
            .group_by(ShipTx.sku_id)
        )
        for sku_id, shipped, first_date in session.exec(ship_q):
            if shipped and shipped > 0 and first_date:
                days = max((today - first_date).days, 1)
                daily_avg = shipped / days
                stock = stock_map.get(sku_id, 0)
                turnover_map[sku_id] = round(stock / daily_avg, 2) if daily_avg else None
            else:
                turnover_map[sku_id] = None

    # ------ 3) マージして返す -----------------------------------------------------------
    all_skus = set(stock_map) | set(oldness_map) | set(turnover_map)
    return {
        sku: {
            "oldness_days": oldness_map.get(sku),
            "turnover_days": turnover_map.get(sku),
        }
        for sku in all_skus
    }


# --------------------------------------------------------------------------- #
# バルク Upsert                                                                #
# --------------------------------------------------------------------------- #
def upsert_metrics(session: Session, metrics: Dict[str, Dict[str, float | int | None]]) -> int:
    """
    ``calculate_metrics`` の戻り値を ``sku_metric`` テーブルへ upsert する。

    戻り値: upsert した行数
    """
    if not metrics:
        return 0

    rows = [
        SkuMetric(
            sku_id=sku,
            oldness_days=data.get("oldness_days"),
            turnover_days=data.get("turnover_days"),
        ).model_dump()
        for sku, data in metrics.items()
    ]

    # primary key は sku_id
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    stmt = pg_insert(SkuMetric).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["sku_id"],
        set_={
            "oldness_days": stmt.excluded.oldness_days,
            "turnover_days": stmt.excluded.turnover_days,
            "updated_at": datetime.utcnow(),
        },
    )
    session.execute(stmt)
    session.commit()
    return len(rows)