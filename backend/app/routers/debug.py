from __future__ import annotations
from typing import Optional, List, Dict, Any
from datetime import date, datetime
import os

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy import func, text
from sqlmodel import Session, select

from app.core.database import engine
from app.models import Sku, Inventory, ShipTx, RecvTx
from app.services import optimizer as optimizer_service

# 読み取り専用のデバッグ用API（副作用なし）。
router = APIRouter(prefix="/v1/debug", tags=["debug"])


def get_session():
    """短命セッション（読み取り専用エンドポイント用）。"""
    with Session(engine) as s:
        yield s


@router.get("/version")
def debug_version():
    return {
        "version": "2025-11-18-inv-volume-debug",
        "timestamp": datetime.now().isoformat(),
        "msg": "Debug: Added inventory-volume matching count to relocation response",
    }


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


@router.get("/locations")
def debug_locations(
    block: Optional[List[str]] = Query(None),     # 例: ?block=B&block=C
    quality: Optional[List[str]] = Query(None),   # 例: ?quality=良品&quality=検品待ち
    limit: int = Query(20, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session),
):
    """
    block_code × quality_name ごとのロケ状態サマリーを返す。
    - total_slots / can_receive / cannot_receive / highness は location_master から集計
    - used_slots は inventory から (location_master に JOIN) 集計
    - usage_rate = used_slots / can_receive （0除算回避）
    """
    # --- 動的 WHERE 生成（IN 句をバインドする）
    conds: List[str] = []
    params: dict = {}

    def _in_clause(field: str, values: List[str], prefix: str):
        ph = []
        pv = {}
        for i, v in enumerate(values):
            key = f"{prefix}{i}"
            ph.append(f":{key}")
            pv[key] = v
        return f"{field} IN (" + ",".join(ph) + ")", pv

    if block:
        blocks = [str(b).strip() for b in block if str(b).strip()]
        if blocks:
            c, p = _in_clause("lm.block_code", blocks, "b")
            conds.append(c); params.update(p)

    if quality:
        quals = [str(q).strip() for q in quality if str(q).strip()]
        if quals:
            c, p = _in_clause("lm.quality_name", quals, "q")
            conds.append(c); params.update(p)

    where_sql = ("WHERE " + " AND ".join(conds)) if conds else ""

    # --- 総グループ数（ページネーション用）
    total_sql = f"""
        SELECT COUNT(*) AS cnt
        FROM (
            SELECT lm.block_code, lm.quality_name
            FROM location_master lm
            {where_sql}
            GROUP BY lm.block_code, lm.quality_name
        ) g
    """
    total = session.exec(text(total_sql).bindparams(**params)).one()[0]

    # --- サマリー本体（ページングあり）
    sum_sql = f"""
        SELECT
            lm.block_code,
            lm.quality_name,
            COUNT(*) AS total_slots,
            SUM(CASE WHEN lm.can_receive THEN 1 ELSE 0 END) AS can_receive,
            COUNT(*) - SUM(CASE WHEN lm.can_receive THEN 1 ELSE 0 END) AS cannot_receive,
            SUM(CASE WHEN lm.highness THEN 1 ELSE 0 END) AS highness
        FROM location_master lm
        {where_sql}
        GROUP BY lm.block_code, lm.quality_name
        ORDER BY lm.block_code, lm.quality_name
        LIMIT :limit OFFSET :offset
    """
    pg_params = dict(params); pg_params["limit"] = limit; pg_params["offset"] = offset
    sum_rows = session.exec(text(sum_sql).bindparams(**pg_params)).all()

    # --- 使用中スロット（在庫JOIN）
    used_sql = f"""
        SELECT
            lm.block_code,
            lm.quality_name,
            COUNT(DISTINCT inv.location_id) AS used_slots
        FROM inventory inv
        JOIN location_master lm
          ON lm.numeric_id = inv.location_id
        {where_sql}
        GROUP BY lm.block_code, lm.quality_name
    """
    used_rows = session.exec(text(used_sql).bindparams(**params)).all()
    used_map = {}
    for r in used_rows:
        m = r._mapping if hasattr(r, "_mapping") else None
        if m:
            used_map[(str(m["block_code"]), str(m["quality_name"]))] = int(m["used_slots"] or 0)

    # --- 整形
    out = []
    for r in sum_rows:
        m = r._mapping if hasattr(r, "_mapping") else None
        if not m:
            continue
        b = str(m["block_code"])
        q = str(m["quality_name"])
        can_recv = int(m["can_receive"] or 0)
        used = int(used_map.get((b, q), 0))
        usage_rate = float(used / can_recv) if can_recv > 0 else 0.0
        out.append({
            "block_code": b,
            "quality_name": q,
            "total_slots": int(m["total_slots"] or 0),
            "can_receive": can_recv,
            "cannot_receive": int(m["cannot_receive"] or 0),
            "highness": int(m["highness"] or 0),
            "used_slots": used,
            "usage_rate": round(usage_rate, 4),
        })

    return {"total": int(total or 0), "rows": out}

# ------------------------- SKU METRICS (velocity) -----------------------
@router.get("/sku_metrics")
def list_sku_metrics(
    window_days: int = Query(..., description="90, 180, 365, 99999"),
    sku_id: Optional[str] = Query(None, description="部分一致検索: sku_id"),
    sort: str = Query("cases_per_day", description="ソートキー"),
    order: str = Query("desc", description="asc/desc"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session),
):
    """sku_metrics テーブルをページング返却するデバッグ用API。

    - window_days は 90/180/365/99999 を想定
    - sort はホワイトリストでカラム固定（SQLインジェクション対策）
    - order は asc/desc
    """
    # --- ソート列をホワイトリスト化
    ALLOWED_SORTS = {
        "sku_id": "sku_id",
        "shipped_cases_all": "shipped_cases_all",
        "current_cases": "current_cases",
        "turnover_rate": "turnover_rate",
        "cases_per_day": "cases_per_day",
        "hits_per_day": "hits_per_day",
        "cube_per_day": "cube_per_day",
        # 受入の速度（テーブルに存在する前提。存在しない環境では値は常にNULLのため並び順に影響なし）
        "recv_cases_per_day": "recv_cases_per_day",
        "recv_hits_per_day": "recv_hits_per_day",
        "recv_cube_per_day": "recv_cube_per_day",
        "updated_at": "updated_at",
    }
    sort_col = ALLOWED_SORTS.get(sort, "cases_per_day")
    order_kw = "DESC" if str(order).lower() == "desc" else "ASC"

    # --- WHERE 句
    conds = ["window_days = :wd"]
    params = {"wd": window_days}
    if sku_id:
        params["sku_like"] = f"%{sku_id}%"
        conds.append("sku_id LIKE :sku_like")
    where_sql = "WHERE " + " AND ".join(conds)

    # --- 件数
    total_sql = f"SELECT COUNT(*) FROM sku_metrics {where_sql}"
    total = session.exec(text(total_sql).bindparams(**params)).one()[0]

    # --- 本体（SQLite 互換の NULLS LAST エミュレーション）
    bind = session.get_bind()
    dialect = getattr(bind.dialect, "name", "") if bind is not None else ""
    if dialect == "sqlite":
        # SQLite は NULLS LAST 未対応のため、CASE 式で擬似的に再現
        # 昇順:  nullを最後 => (col IS NULL) ASC, col ASC
        # 降順:  nullを最後 => (col IS NULL) ASC, col DESC
        nulls_last = "ASC"
        order_clause = f"( {sort_col} IS NULL ) {nulls_last}, {sort_col} {order_kw}"
    else:
        order_clause = f"{sort_col} {order_kw} NULLS LAST"
    select_cols = (
        "sku_id, "
        "window_days, "
        "shipped_cases_all, "
        "current_cases, "
        "turnover_rate, "
        "cases_per_day, "
        "hits_per_day, "
        "cube_per_day, "
        # 受入速度系（列が無い場合はマイグレーション前想定だが、COALESCEで数値化）
        "COALESCE(recv_cases_per_day, 0) AS recv_cases_per_day, "
        "COALESCE(recv_hits_per_day, 0) AS recv_hits_per_day, "
        "COALESCE(recv_cube_per_day, 0) AS recv_cube_per_day, "
        "updated_at"
    )

    data_sql = f"""
        SELECT {select_cols}
        FROM sku_metrics
        {where_sql}
        ORDER BY {order_clause}
        LIMIT :limit OFFSET :offset
    """
    q_params = dict(params)
    q_params.update({"limit": limit, "offset": offset})
    rows = session.exec(text(data_sql).bindparams(**q_params)).all()

    # --- 整形
    out = []
    for r in rows:
        m = r._mapping if hasattr(r, "_mapping") else None
        if not m:
            continue
        out.append({
            "sku_id": str(m["sku_id"]),
            "window_days": int(m["window_days"]) if m.get("window_days") is not None else None,
            "shipped_cases_all": float(m["shipped_cases_all"]) if m["shipped_cases_all"] is not None else 0,
            "current_cases": float(m["current_cases"]) if m["current_cases"] is not None else 0,
            "turnover_rate": float(m["turnover_rate"]) if m["turnover_rate"] is not None else 0,
            "cases_per_day": float(m["cases_per_day"]) if m["cases_per_day"] is not None else 0,
            "hits_per_day": float(m["hits_per_day"]) if m["hits_per_day"] is not None else 0,
            "cube_per_day": float(m["cube_per_day"]) if m["cube_per_day"] is not None else 0,
            "recv_cases_per_day": float(m["recv_cases_per_day"]) if m.get("recv_cases_per_day") is not None else 0,
            "recv_hits_per_day": float(m["recv_hits_per_day"]) if m.get("recv_hits_per_day") is not None else 0,
            "recv_cube_per_day": float(m["recv_cube_per_day"]) if m.get("recv_cube_per_day") is not None else 0,
            "updated_at": str(m["updated_at"]) if m["updated_at"] is not None else None,
        })

    return {"total": int(total or 0), "rows": out}


# ---------------------- RELOCATION DROP TRACE -------------------------
@router.get("/relocation/trace/{trace_id}/drops")
def get_relocation_drops(
    trace_id: str,
    agg: Optional[str] = Query(
        None, description="reason を指定すると理由別の集計を返す"
    ),
    limit: int = Query(20, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """フロントの「落選理由」ビュー用 API。

    - `/v1/debug/relocation/trace/{trace_id}/drops?agg=reason` … 理由別集計
    - `/v1/debug/relocation/trace/{trace_id}/drops?limit=..&offset=..` … 明細
    """
    # 集計モード
    if agg == "reason":
        try:
            rows = optimizer_service.get_drop_summary(trace_id)
        except AttributeError:
            raise HTTPException(
                status_code=501,
                detail="drop summary API 未実装（optimizer.get_drop_summary が見つかりません）",
            )
        total = sum(int(r.get("count", 0)) for r in rows)
        return {"total": int(total), "rows": rows}

    # 明細モード
    try:
        rows, total = optimizer_service.get_drop_details(
            trace_id, limit=limit, offset=offset
        )
    except AttributeError:
        raise HTTPException(
            status_code=501,
            detail="drop details API 未実装（optimizer.get_drop_details が見つかりません）",
        )

    # trace が存在しない場合は空配列で返す
    if rows is None:
        rows, total = [], 0
    return {"total": int(total or 0), "rows": rows}


# ---------------------- RELOCATION CANDIDATES (ranking) -------------------------
@router.get("/relocation/candidates")
def relocation_candidates(
    window_days: int = Query(..., description="分析対象の期間（例: 90, 180, 365, 99999）"),
    mode: str = Query("combo", description="ランキングのスコア種別: hits|cases|turnover|combo"),
    w_hits: float = Query(1.0, description="combo時の重み: 出荷ヒット/日"),
    w_cases: float = Query(0.4, description="combo時の重み: 出荷ケース/日"),
    w_recv_hits: float = Query(0.2, description="combo時の重み: 入荷ヒット/日"),
    min_current_cases: float = Query(0.0, ge=0.0, description="現在庫の下限（0で制限なし）"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session),
):
    """リロケーションの優先候補SKUを返す簡易ランキングAPI。

    - mode="hits":      score = hits_per_day
    - mode="cases":     score = cases_per_day
    - mode="turnover":  score = turnover_rate
    - mode="combo":     score = w_hits*hits_per_day + w_cases*cases_per_day + w_recv_hits*recv_hits_per_day
    """
    # パラメータのバリデーション（mode）
    _mode = str(mode or "").lower()
    if _mode not in {"hits", "cases", "turnover", "combo"}:
        raise HTTPException(status_code=400, detail="mode must be one of: hits|cases|turnover|combo")

    # DB方言
    bind = session.get_bind()
    dialect = getattr(bind.dialect, "name", "") if bind is not None else ""

    # WHERE 条件
    conds = ["window_days = :wd"]
    params: Dict[str, Any] = {"wd": int(window_days)}
    if float(min_current_cases or 0) > 0:
        conds.append("current_cases >= :min_cc")
        params["min_cc"] = float(min_current_cases)
    where_sql = "WHERE " + " AND ".join(conds)

    # スコア式（COALESCEでNULLを0に）
    if _mode == "hits":
        score_expr = "COALESCE(hits_per_day, 0.0)"
    elif _mode == "cases":
        score_expr = "COALESCE(cases_per_day, 0.0)"
    elif _mode == "turnover":
        score_expr = "COALESCE(turnover_rate, 0.0)"
    else:  # combo
        score_expr = (
            "(:w_hits*COALESCE(hits_per_day,0.0) + "
            ":w_cases*COALESCE(cases_per_day,0.0) + "
            ":w_recv*COALESCE(recv_hits_per_day,0.0))"
        )
        params.update({"w_hits": float(w_hits), "w_cases": float(w_cases), "w_recv": float(w_recv_hits)})

    # NULLS LAST エミュレーション（SQLite）
    if dialect == "sqlite":
        order_clause = f"( {score_expr} IS NULL ) ASC, {score_expr} DESC"
    else:
        order_clause = f"{score_expr} DESC NULLS LAST"

    select_cols = (
        "sku_id, "
        "shipped_cases_all, current_cases, turnover_rate, "
        "cases_per_day, hits_per_day, cube_per_day, "
        "COALESCE(recv_cases_per_day,0) AS recv_cases_per_day, "
        "COALESCE(recv_hits_per_day,0)  AS recv_hits_per_day, "
        "COALESCE(recv_cube_per_day,0)  AS recv_cube_per_day, "
        f"{score_expr} AS score"
    )

    sql = f"""
        SELECT {select_cols}
        FROM sku_metrics
        {where_sql}
        ORDER BY {order_clause}
        LIMIT :limit OFFSET :offset
    """
    params.update({"limit": int(limit), "offset": int(offset)})
    rows = session.exec(text(sql).bindparams(**params)).all()

    out: List[Dict[str, Any]] = []
    for r in rows:
        m = r._mapping if hasattr(r, "_mapping") else None
        if not m:
            continue
        out.append({
            "sku_id": str(m["sku_id"]),
            "score": float(m["score"]) if m["score"] is not None else 0.0,
            "hits_per_day": float(m["hits_per_day"]) if m["hits_per_day"] is not None else 0.0,
            "cases_per_day": float(m["cases_per_day"]) if m["cases_per_day"] is not None else 0.0,
            "recv_hits_per_day": float(m["recv_hits_per_day"]) if m.get("recv_hits_per_day") is not None else 0.0,
            "turnover_rate": float(m["turnover_rate"]) if m["turnover_rate"] is not None else 0.0,
            "current_cases": float(m["current_cases"]) if m["current_cases"] is not None else 0.0,
        })

    return {"rows": out, "count": len(out)}


@router.get("/location_master_raw")
def debug_location_master_raw(
    block: Optional[str] = Query(None),
    quality: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=100),
    session: Session = Depends(get_session),
):
    """LocationMasterテーブルの実データを返す"""
    try:
        from app.routers.upload import LocationMaster
        stmt = select(
            LocationMaster.block_code,
            LocationMaster.quality_name,
            LocationMaster.level,
            LocationMaster.column,
            LocationMaster.depth,
            LocationMaster.can_receive,
            LocationMaster.numeric_id,
            LocationMaster.display_code,
        )
        if block:
            stmt = stmt.where(LocationMaster.block_code == block)
        if quality:
            stmt = stmt.where(LocationMaster.quality_name == quality)
        stmt = stmt.limit(limit)
        
        rows = session.exec(stmt).all()
        return {
            "rows": [
                {
                    "block_code": r[0],
                    "quality_name": r[1],
                    "level": r[2],
                    "column": r[3],
                    "depth": r[4],
                    "can_receive": r[5],
                    "numeric_id": r[6],
                    "display_code": r[7],
                }
                for r in rows
            ],
            "count": len(rows),
        }
    except Exception as e:
        return {"error": str(e), "rows": [], "count": 0}


# Build: 20251110-233750
