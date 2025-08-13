from __future__ import annotations
from typing import Optional, List
from datetime import date
import os

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlmodel import Session, select

from app.core.database import engine
from app.models import Sku, Inventory, ShipTx, RecvTx

# 読み取り専用のデバッグ用API（副作用なし）。
router = APIRouter(prefix="/v1/debug", tags=["debug"])


def get_session():
    """短命セッション（読み取り専用エンドポイント用）。"""
    with Session(engine) as s:
        yield s


# ----------------------------- SKU -------------------------------------
@router.get("/sku")
def list_sku(
    q: Optional[str] = Query(None, description="部分一致検索: sku_id"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session),
):
    conds = []
    if q:
        conds.append(Sku.sku_id.contains(q))
    total = session.exec(select(func.count()).select_from(Sku).where(*conds)).one()
    rows = session.exec(
        select(Sku).where(*conds).order_by(Sku.sku_id).offset(offset).limit(limit)
    ).all()
    return {"rows": [r.model_dump(mode="json") for r in rows], "total": int(total)}


# --------------------------- INVENTORY ----------------------------------
@router.get("/inventory")
def list_inventory(
    block: Optional[List[str]] = Query(None, description="ブロック略称（複数可）"),
    sku_id: Optional[str] = Query(None),
    location_like: Optional[str] = Query(None),
    lot_like: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session),
):
    conds = []
    if block:
        conds.append(Inventory.block_code.in_(block))
    if sku_id:
        conds.append(Inventory.sku_id == sku_id)

    # location_like は location_id または location のどちらかが存在すれば適用
    loc_attr = getattr(Inventory, "location_id", None) or getattr(Inventory, "location", None)
    if location_like and loc_attr is not None:
        conds.append(loc_attr.contains(location_like))

    # lot_like は lot があれば部分一致
    lot_attr = getattr(Inventory, "lot", None)
    if lot_like and lot_attr is not None:
        conds.append(lot_attr.contains(lot_like))

    total = session.exec(select(func.count()).select_from(Inventory).where(*conds)).one()

    # 並び順：location_id/location → level → column → depth （存在するものだけ）
    order_cols = []
    if getattr(Inventory, "location_id", None) is not None:
        order_cols.append(Inventory.location_id)
    elif getattr(Inventory, "location", None) is not None:
        order_cols.append(Inventory.location)
    if getattr(Inventory, "level", None) is not None:
        order_cols.append(Inventory.level)
    if getattr(Inventory, "column", None) is not None:
        order_cols.append(Inventory.column)
    if getattr(Inventory, "depth", None) is not None:
        order_cols.append(Inventory.depth)

    stmt = select(Inventory).where(*conds)
    if order_cols:
        stmt = stmt.order_by(*order_cols)
    stmt = stmt.offset(offset).limit(limit)

    rows = session.exec(stmt).all()
    return {"rows": [r.model_dump(mode="json") for r in rows], "total": int(total)}


# ---------------------------- 入荷実績 ----------------------------------
@router.get("/recv_tx")
def list_recv_tx(
    sku_id: Optional[str] = Query(None),
    start: Optional[date] = Query(None, description="開始日"),
    end: Optional[date] = Query(None, description="終了日(含む)"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session),
):
    conds = []
    if sku_id:
        conds.append(RecvTx.sku_id == sku_id)
    if start:
        conds.append(RecvTx.trandate >= start)
    if end:
        conds.append(RecvTx.trandate <= end)

    total = session.exec(select(func.count()).select_from(RecvTx).where(*conds)).one()
    rows = session.exec(
        select(RecvTx)
        .where(*conds)
        .order_by(RecvTx.trandate.desc())
        .offset(offset)
        .limit(limit)
    ).all()
    return {"rows": [r.model_dump(mode="json") for r in rows], "total": int(total)}


# ---------------------------- 出荷実績 ----------------------------------
@router.get("/ship_tx")
def list_ship_tx(
    sku_id: Optional[str] = Query(None),
    start: Optional[date] = Query(None),
    end: Optional[date] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session),
):
    conds = []
    if sku_id:
        conds.append(ShipTx.sku_id == sku_id)
    if start:
        conds.append(ShipTx.trandate >= start)
    if end:
        conds.append(ShipTx.trandate <= end)

    total = session.exec(select(func.count()).select_from(ShipTx).where(*conds)).one()
    rows = session.exec(
        select(ShipTx)
        .where(*conds)
        .order_by(ShipTx.trandate.desc())
        .offset(offset)
        .limit(limit)
    ).all()
    return {"rows": [r.model_dump(mode="json") for r in rows], "total": int(total)}


# ---------------------------- AI DEBUG ----------------------------------
@router.get("/ai")
def debug_ai(probe: bool = False):
    """Return AI-related config and optionally probe OpenAI connectivity.

    Query params:
      - probe: when true, attempt a lightweight call (`client.models.list()`).
    """
    data = {
        "has_api_key": bool(os.getenv("OPENAI_API_KEY")),
        "model": os.getenv("OPENAI_MODEL", "gpt-5-2025-08-07"),
        "temperature": float(os.getenv("AI_PLANNER_TEMPERATURE", "0.2")),
        "timeout_s": int(os.getenv("AI_PLANNER_TIMEOUT_S", "60")),
        "frontend_origins": os.getenv("FRONTEND_ORIGINS") or os.getenv("FRONTEND_ORIGIN"),
    }

    if probe:
        try:
            # import inside to avoid hard dependency when not probing
            from openai import OpenAI
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            _ = client.models.list()
            data["probe_ok"] = True
        except Exception as e:
            data["probe_ok"] = False
            data["probe_error"] = str(e)

    return data