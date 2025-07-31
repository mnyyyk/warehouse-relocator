

"""
api.py
~~~~~~
FastAPI endpoints for *warehouse‑relocator* PoC.

Current endpoints
-----------------
POST /api/optimize
    Multipart upload of 4 Excel files + parameters.
    Returns JSON with relocation plan and basic KPIs.
"""
from __future__ import annotations

import io
from typing import List, Tuple

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile, Form

from .services import data_io, features, optimizer, move_order

router = APIRouter(prefix="/api", tags=["optimizer"])


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _excel_to_df(file: UploadFile) -> pd.DataFrame:
    """Read an UploadFile (Excel) into a pandas DataFrame."""
    try:
        binary = io.BytesIO(file.file.read())
        return pd.read_excel(binary)
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=400, detail=f"Invalid Excel: {file.filename}") from exc


def _validate_files(files: List[UploadFile]) -> Tuple[UploadFile, UploadFile, UploadFile, UploadFile]:
    """Ensure exactly 4 files are provided in the expected order."""
    if len(files) != 4:
        raise HTTPException(
            status_code=400,
            detail="Exactly 4 Excel files required: inventory, SKU, inbound, outbound",
        )
    return tuple(files)  # type: ignore[return-value]


# ----------------------------------------------------------------------
# Endpoint
# ----------------------------------------------------------------------
@router.post("/optimize")
async def optimize(
    files: List[UploadFile] = File(..., description="inventory.xlsx, SKU.xlsx, inbound.xlsx, outbound.xlsx (in order)"),
    max_moves: int = Form(100),
    w1: float = Form(1.0, description="Weight for pick distance"),
    w2: float = Form(1.0, description="Weight for ageing (lot level)"),
    w3: float = Form(1.0, description="Weight for move cost"),
    time_limit: int = Form(60, description="OR‑Tools solver time limit (sec)"),
):
    """
    Run slotting optimisation and return relocation plan.

    **files** must be exactly four Excel sheets in the following order:

    1. 在庫データ.xlsx
    2. SKU.xlsx
    3. 入荷実績.xlsx
    4. 出荷実績.xlsx

    Response schema
    ---------------
    {
        "records": [ {move_order, from_loc, to_loc, will_move, ...}, ... ],
        "kpi": {
            "total_moves": int,
            "moved_items": int
        }
    }
    """
    inv_file, sku_file, inbound_file, outbound_file = _validate_files(files)

    # ---------------- Load data into DataFrames ----------------
    inv_df = _excel_to_df(inv_file)
    sku_df = _excel_to_df(sku_file)
    inbound_df = _excel_to_df(inbound_file)
    outbound_df = _excel_to_df(outbound_file)

    # ---------------- Pipeline ----------------
    # 1) 在庫と SKU マスタをマージし、箱体積を算出
    # -------------------------------------------------
    # SKU 列名を前処理（前後空白を除去）
    sku_df.columns = sku_df.columns.str.strip()

    # 入り数 / 体積 列を動的に検出
    alt_qty_cols = ["入り数", "入数", "入数(ケース)"]
    alt_vol_cols = ["商品予備項目006", "商品予備項目6", "商品予備項目００６", "体積"]

    qty_col = next((c for c in alt_qty_cols if c in sku_df.columns), None)
    vol_col = next((c for c in alt_vol_cols if c in sku_df.columns), None)

    if not qty_col or not vol_col:
        raise HTTPException(
            status_code=400,
            detail="SKUシートに『入り数』『商品予備項目006』相当の列が見つかりません",
        )

    # 列名を統一してマージ
    sku_norm = sku_df.rename(columns={qty_col: "入り数", vol_col: "商品予備項目006"})
    snapshot = inv_df.merge(
        sku_norm[["商品ID", "入り数", "商品予備項目006"]],
        on="商品ID",
        how="left",
    )

    # pandas merge may create suffixes (_x, _y) if duplicates existed; handle that
    if "入り数" not in snapshot.columns:
        qty_candidates = [c for c in snapshot.columns if c.startswith("入り数")]
        if qty_candidates:
            snapshot["入り数"] = snapshot[qty_candidates[0]]

    if "商品予備項目006" not in snapshot.columns:
        vol_candidates = [c for c in snapshot.columns if c.startswith("商品予備項目006")]
        if vol_candidates:
            snapshot["商品予備項目006"] = snapshot[vol_candidates[0]]

    snapshot["case_vol_m3"] = snapshot["入り数"] * snapshot["商品予備項目006"]

    # 2) ロケーションコードを level / column / depth に分解
    from app.services.data_io import _parse_location_code  # type: ignore

    loc_parts = (
        snapshot["ロケーション"]
        .astype(str)
        .apply(_parse_location_code)
        .apply(pd.Series)
    )
    loc_parts.columns = ["level", "column", "depth"]
    snapshot = pd.concat([snapshot, loc_parts], axis=1)

    # 3) 指標生成とクラスタリング
    today = pd.Timestamp("today").normalize()
    snapshot = features.add_metrics(snapshot, outbound_df, today)
    snapshot = features.cluster_sku(snapshot, n_clusters=10)

    # Optimisation
    w = (w1, w2, w3)
    result_df = optimizer.solve(snapshot, max_moves=max_moves, w=w, time_limit=time_limit)

    # Plan move order
    result_df = move_order.plan_move_order(result_df)

    # ---------------- Build response ----------------
    records = result_df.to_dict(orient="records")

    kpi = {
        "total_items": len(result_df),
        "moved_items": int(result_df["will_move"].sum()),
    }
    return {"records": records, "kpi": kpi}