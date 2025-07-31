

"""
Native ingestion pipeline for Excel source files.

Expected files:
    - SKU.xlsx
    - 入荷実績_*.xlsx  (Inbound history, recordtype="発注")
    - 出荷実績.xlsx   (Outbound history)
    - 在庫データ.xlsx (Current inventory snapshot)

Each load_*() function converts pandas DataFrame rows to native SQLAlchemy
models and bulk‑inserts them with a single commit.

Usage (example):
    from app.db.ingest_native import ingest_all
    ingest_all(
        sku_path="SKU.xlsx",
        inbound_path="入荷実績_20250722.xlsx",
        outbound_path="出荷実績.xlsx",
        inventory_path="在庫データ.xlsx",
    )
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Literal

import pandas as pd
from sqlalchemy.orm import Session

from app.db.session_native import SessionLocal
from app.models_native.sku_master import SKUMaster
from app.models_native.inbound import Inbound
from app.models_native.outbound import Outbound
from app.models_native.inventory_snapshot import InventorySnapshot
from app.utils.lot import parse_lot_date


# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #


def _read_excel(path: Path | str, sheet_name: str | None = None) -> pd.DataFrame:
    """Read an Excel file and return a DataFrame (empty if not found)."""
    if not path:
        return pd.DataFrame()
    try:
        return pd.read_excel(path, sheet_name=sheet_name)
    except FileNotFoundError:
        raise RuntimeError(f"Excel file not found: {path}")


# --------------------------------------------------------------------------- #
# Individual loaders
# --------------------------------------------------------------------------- #


def load_sku_master(df: pd.DataFrame, db: Session) -> None:
    """Insert SKU master rows."""
    if df.empty:
        return

    records: list[SKUMaster] = []
    for _, row in df.iterrows():
        try:
            volume_m3 = float(row["商品予備項目００６"]) * int(row["入数"])
        except Exception:
            volume_m3 = None

        records.append(
            SKUMaster(
                sku_code=str(row["商品ID"]).strip(),
                name=str(row.get("商品名", "")),
                length_mm=row.get("商品予備項目００３"),
                width_mm=row.get("商品予備項目００４"),
                height_mm=row.get("商品予備項目００５"),
                pack_qty=row.get("入数"),
                volume_m3=volume_m3,
            )
        )
    db.bulk_save_objects(records)


def load_inbounds(df: pd.DataFrame, db: Session) -> None:
    """Insert inbound history rows (recordtype='発注')."""
    if df.empty:
        return

    # Filter 発注
    df = df[df["recordtype"] == "発注"]

    recs: list[Inbound] = []
    for _, row in df.iterrows():
        lot = str(row.get("lot", ""))
        recs.append(
            Inbound(
                sku_code=str(row["item_internalid"]).strip(),
                qty=int(row["item_quantity"]),
                inbound_date=row["trandate"].date() if isinstance(row["trandate"], dt.datetime) else row["trandate"],
                lot=lot,
                lot_date=parse_lot_date(lot),
            )
        )
    db.bulk_save_objects(recs)


def load_outbounds(df: pd.DataFrame, db: Session) -> None:
    """Insert outbound history rows."""
    if df.empty:
        return

    recs: list[Outbound] = []
    for _, row in df.iterrows():
        recs.append(
            Outbound(
                sku_code=str(row["item_internalid"]).strip(),
                qty=int(row["item_shipquantity"]),
                ship_date=row["trandate"].date() if isinstance(row["trandate"], dt.datetime) else row["trandate"],
            )
        )
    db.bulk_save_objects(recs)


def load_inventory_snapshot(df: pd.DataFrame, db: Session) -> None:
    """Insert current inventory snapshot rows."""
    if df.empty:
        return

    recs: list[InventorySnapshot] = []
    for _, row in df.iterrows():
        loc_code = str(row["ロケーション"]).zfill(6)
        level = int(loc_code[0])
        column = int(loc_code[1:4])
        depth = int(loc_code[4:6])

        lot = str(row.get("ロット", ""))
        recs.append(
            InventorySnapshot(
                sku_code=str(row["SKU"]).strip(),
                loc_code=loc_code,
                qty=int(row["数量"]),
                level=level,
                column=column,
                depth=depth,
                lot=lot,
                lot_date=parse_lot_date(lot),
            )
        )
    # First, clear existing snapshot (keep history in separate table if needed)
    db.query(InventorySnapshot).delete()
    db.bulk_save_objects(recs)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #


def ingest_all(
    *,
    sku_path: Path | str,
    inbound_path: Path | str,
    outbound_path: Path | str,
    inventory_path: Path | str,
    commit: bool = True,
) -> None:
    """Run full ingestion pipeline into native tables."""
    with SessionLocal() as db:
        # 1. SKU
        load_sku_master(_read_excel(sku_path), db)

        # 2. Inbound
        load_inbounds(_read_excel(inbound_path), db)

        # 3. Outbound
        load_outbounds(_read_excel(outbound_path), db)

        # 4. Inventory snapshot
        load_inventory_snapshot(_read_excel(inventory_path), db)

        if commit:
            db.commit()