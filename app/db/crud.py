"""
CRUD helper functions for the warehouse‑relocator project.

These thin wrappers keep raw SQLAlchemy queries out of the service / API
layers, and centralise DB‑access patterns so they can be adjusted in one place
(e.g. when moving from a local SQLite database to Amazon RDS in production).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from sqlmodel import Session, select
from sqlalchemy import delete

from . import models

import logging

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Helper utilities                                                           #
# --------------------------------------------------------------------------- #

_SKU_CODE_KEYS = {"code", "sku_code", "SKU", "商品ID"}
_SKU_DESC_KEYS: set[str] = {"description", "name", "sku_name", "商品名"}
_SKU_VOL_KEYS: set[str] = {"volume_l", "volume", "容積L", "容積ｌ", "商品予備項目００６"}


def _normalise_sku_dict(raw: Dict) -> Dict:
    """
    Convert a heterogenous Excel‑derived row dict into the canonical form
    expected by :func:`upsert_skus`.

    The canonical keys are::

        sku_code   : str
        description : str | None
        volume_l   : float | None

    Extraneous keys are preserved so that upstream callers can still
    access them if required.
    """
    norm: Dict = {}

    # ---- code / SKU --------------------------------------------------------
    for key in _SKU_CODE_KEYS:
        if key in raw and raw[key]:
            norm["sku_code"] = str(raw[key]).strip()
            break
    else:
        raise ValueError("SKU row missing an identifier column (e.g. 'sku_code').")

    # ---- description --------------------------------------------------------------
    for key in _SKU_DESC_KEYS:
        if key in raw and raw[key]:
            norm["description"] = str(raw[key]).strip()
            break

    # ---- volume ------------------------------------------------------------
    for key in _SKU_VOL_KEYS:
        if key in raw and raw[key] not in (None, ""):
            try:
                norm["volume_l"] = float(raw[key])
            except Exception:
                pass
            break

    # Preserve any additional columns (they might be useful elsewhere)
    norm.update(raw)
    if "sku_code" not in norm:
        raise ValueError("SKU row missing 'sku_code' after normalization.")
    return norm

# --------------------------------------------------------------------------- #
#  SKU                                                                        #
# --------------------------------------------------------------------------- #


def get_sku(db: Session, sku_code: str) -> models.SKUMaster | None:
    stmt = select(models.SKUMaster).where(models.SKUMaster.sku_code == sku_code)
    return db.exec(stmt).first()


def upsert_sku(db: Session, sku_dict: Dict) -> None:
    """
    Insert or update a *single* SKU.

    This is a thin convenience wrapper around :func:`upsert_skus` so that
    callers can work with one‑item payloads without needing to wrap them in a
    list.  The *sku_dict* must contain the same mandatory keys as in
    ``upsert_skus`` (``code``, ``description``, ``volume_l``).

    Parameters
    ----------
    db : Session
        An active SQLAlchemy session.
    sku_dict : Dict
        Dictionary of SKU attributes (see :func:`upsert_skus`).
    """
    if not sku_dict:  # no‑op on empty / None input
        return
    # Delegate to the bulk helper to avoid code duplication.
    upsert_skus(db, [sku_dict])


def upsert_skus(db: Session, sku_dicts: Sequence[Dict]) -> None:
    """
    Insert new SKUs or update existing ones.

    Each dict may originate from various Excel exports so the following
    aliases are recognised (case‑sensitive):

        - ``code`` / ``sku_code`` / ``SKU`` / ``商品ID``
        - ``description`` / ``name`` / ``sku_name`` / ``商品名``  (optional)
        - ``volume_l`` / ``volume`` / ``容積L`` / ``商品予備項目００６`` (optional)

    Extra keys are ignored.

    This helper is intentionally *bulk*‑oriented and commits once at the end.
    """
    for raw in sku_dicts:
        data = _normalise_sku_dict(raw)
        vol = float(data.get("volume_l") or 0.0)
        sku = get_sku(db, data["sku_code"])
        if sku is None:
            sku = models.SKUMaster(
                sku_code=data["sku_code"],
                volume_l=vol,
            )
            if data.get("description"):
                sku.description = data["description"]
            db.add(sku)
        else:
            # Update mutable fields
            if data.get("description"):
                sku.description = data["description"]
            if "volume_l" in data and data["volume_l"] is not None:
                sku.volume_l = float(data["volume_l"])

    db.commit()


# --------------------------------------------------------------------------- #
#  Inbound / Outbound                                                         #
# --------------------------------------------------------------------------- #


def add_inbound_records(db: Session, rows: Sequence[Dict]) -> None:
    """
    Bulk‑insert inbound (入荷) records.

    Each *row* dict must include:
        - ``sku_code`` (str)
        - ``amount``   (int)
        - ``arrived_at`` (datetime.datetime)
    """
    objs: List[models.Receipt] = []
    for r in rows:
        sku = get_sku(db, str(r["sku_code"]))
        if sku is None:
            logger.warning("Skipping inbound row – unknown SKU: %s", r["sku_code"])
            continue
        objs.append(
            models.Receipt(
                sku_id=sku.id,
                qty=int(r["qty"]),
                receipt_date=r["receipt_date"],
            )
        )

    # If nothing to insert, return early
    if not objs:
        return

    db.bulk_save_objects(objs)
    db.commit()


def add_outbound_records(db: Session, rows: Sequence[Dict]) -> None:
    """
    Bulk‑insert outbound (出荷) records.

    Each *row* dict must include:
        - ``sku_code`` (str)
        - ``amount``   (int)
        - ``shipped_at`` (datetime.datetime)
    """
    objs: List[models.Shipment] = []
    for r in rows:
        sku = get_sku(db, str(r["sku_code"]))
        if sku is None:
            logger.warning("Skipping outbound row – unknown SKU: %s", r["sku_code"])
            continue
        objs.append(
            models.Shipment(
                sku_id=sku.id,
                qty=int(r["qty"]),
                ship_date=r["shipment_date"],
            )
        )

    if not objs:
        return

    db.bulk_save_objects(objs)
    db.commit()


# --------------------------------------------------------------------------- #
#  Compatibility wrappers expected by app.services.ingest                     #
# --------------------------------------------------------------------------- #

def insert_inbound(db: Session, row: Dict) -> None:
    """
    Single‑row wrapper around :func:`add_inbound_records` kept for backwards
    compatibility with the ingest code (which iterates one row at a time).
    """
    if not row:
        return
    add_inbound_records(
        db,
        [
            {
                "sku_code": str(row.get("SKU") or row.get("sku_code")),
                "qty": int(row.get("数量") or row.get("qty") or 0),
                "receipt_date": row.get("受入日") or row.get("receipt_date"),
            }
        ],
    )


def insert_outbound(db: Session, row: Dict) -> None:
    """
    Single‑row wrapper around :func:`add_outbound_records` kept for backwards
    compatibility with the ingest code (which iterates one row at a time).
    """
    if not row:
        return
    add_outbound_records(
        db,
        [
            {
                "sku_code": str(row.get("SKU") or row.get("sku_code")),
                "qty": int(row.get("数量") or row.get("qty") or 0),
                "shipment_date": row.get("出荷日") or row.get("shipment_date"),
            }
        ],
    )


# --------------------------------------------------------------------------- #
#  Inventory                                                                  #
# --------------------------------------------------------------------------- #

def upsert_inventory(db: Session, inv_dict: Dict) -> None:
    """
    Insert or update a single inventory snapshot line.

    Recognised keys (case‑sensitive/JP mixed):
        - ``ロケーション`` / ``location``  (str)
        - ``SKU`` / ``sku_code``           (str)
        - ``ケース数`` / ``cases``          (int)

    Extra keys are ignored.
    """
    loc = str(inv_dict.get("ロケーション") or inv_dict.get("location")).strip()
    sku = str(inv_dict.get("SKU") or inv_dict.get("sku_code")).strip()
    cases = int(inv_dict.get("ケース数") or inv_dict.get("cases") or 0)

    stmt = (
        select(models.Inventory)
        .where(models.Inventory.location == loc, models.Inventory.sku_code == sku)
    )
    rec = db.exec(stmt).first()

    if rec is None:
        rec = models.Inventory(location=loc, sku_code=sku, cases=cases)
        db.add(rec)
    else:
        rec.cases = cases

    db.commit()


# --------------------------------------------------------------------------- #
#  Analysis Results                                                           #
# --------------------------------------------------------------------------- #


def save_analysis_result(db: Session, result_name: str, payload: Dict) -> models.AnalysisResult:
    """
    Store a named analysis result as JSON (JSONB in PostgreSQL).
    If a record with the same *result_name* already exists, it is updated.
    """
    stmt = select(models.AnalysisResult).where(models.AnalysisResult.result_name == result_name)
    rec = db.exec(stmt).first()
    if rec is None:
        rec = models.AnalysisResult(result_name=result_name, payload=payload)
        db.add(rec)
    else:
        rec.payload = payload
    db.commit()
    db.refresh(rec)
    return rec


def load_analysis_result(db: Session, result_name: str) -> Dict | None:
    """Return the JSON payload for *result_name*, or *None* if not found."""
    stmt = select(models.AnalysisResult.payload).where(models.AnalysisResult.result_name == result_name)
    result = db.exec(stmt).first()
    return result if result is None else result[0]


# --------------------------------------------------------------------------- #
#  House‑keeping utilities                                                    #
# --------------------------------------------------------------------------- #


def clear_all_tables(db: Session) -> None:
    """
    Dangerous helper for development / tests: TRUNCATE every table.
    """
    for model in (models.Receipt, models.Shipment, models.SKUMaster, models.AnalysisResult):
        db.execute(delete(model.__table__))
    db.commit()