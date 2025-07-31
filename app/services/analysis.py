"""
app/services/analysis.py
~~~~~~~~~~~~~~~~~~~~~~~~
Batch job that derives *movement‑based* statistics for every SKU and stores
the result in the ``analysis_results`` table.

The analysis is intentionally simple for the PoC:

* Look at the last *horizon* days (default **180 days**) of receipts
  and shipments.
* Aggregate **total in‑qty**, **total out‑qty**, earliest / latest movement
  date and a crude **average daily velocity**.
* Persist one row per SKU in :class:`~app.db.models.AnalysisResult`.

The helper returns the number of SKUs processed so the caller can log /
surface progress.

Usage example
-------------
>>> from app.services.analysis import run_analysis
>>> run_analysis()          # uses default 180‑day horizon
123
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Iterable

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db import SessionLocal, models as core_models
from ..db.models_analysis import AnalysisRun, SKUStat

__all__ = ["run_analysis", "fetch_latest_result"]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Analysis core                                                               #
# --------------------------------------------------------------------------- #


def _receipt_df(session: Session, since: _dt.date) -> pd.DataFrame:
    """Return a DataFrame [sku_id, qty, date] for receipts since *since*."""
    rows = (
        session.execute(
            select(
                core_models.Receipt.sku_id,
                core_models.Receipt.qty,
                core_models.Receipt.receipt_date.label("date"),
            ).where(core_models.Receipt.receipt_date >= since)
        )
        .all()
    )
    return pd.DataFrame(rows, columns=["sku_id", "qty", "date"])


def _shipment_df(session: Session, since: _dt.date) -> pd.DataFrame:
    """Return a DataFrame [sku_id, qty, date] for shipments since *since*."""
    rows = (
        session.execute(
            select(
                core_models.Shipment.sku_id,
                core_models.Shipment.qty,
                core_models.Shipment.ship_date.label("date"),
            ).where(core_models.Shipment.ship_date >= since)
        )
        .all()
    )
    return pd.DataFrame(rows, columns=["sku_id", "qty", "date"])


def run_analysis(
    db_session: Session | None = None,
    horizon_days: int = 180,
) -> int:
    """
    Execute the movement analysis and persist the results.

    Parameters
    ----------
    db_session : sqlalchemy.orm.Session, optional
        Existing DB session.  If *None* a short‑lived session is created.
    horizon_days : int
        How many *days back* to look at receipts / shipments.

    Returns
    -------
    int
        Number of SKUs processed.
    """
    managed: bool = db_session is None
    session: Session = db_session or SessionLocal()

    since = _dt.date.today() - _dt.timedelta(days=horizon_days)
    logger.info("Running movement analysis for the last %s days (since %s)", horizon_days, since)

    # --- Load movement data -------------------------------------------------
    rec_df = _receipt_df(session, since)
    ship_df = _shipment_df(session, since)

    if rec_df.empty and ship_df.empty:
        logger.warning("No movement data in the selected horizon — skipping analysis")
        if managed:
            session.close()
        return 0

    # --- Aggregate ----------------------------------------------------------
    def _aggregate(df: pd.DataFrame, label: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["sku_id", label, f"first_{label}", f"last_{label}"])
        grouped = (
            df.groupby("sku_id")
            .agg(
                **{
                    label: ("qty", "sum"),
                    f"first_{label}": ("date", "min"),
                    f"last_{label}": ("date", "max"),
                }
            )
            .reset_index()
        )
        return grouped

    in_df = _aggregate(rec_df, "in_qty")
    out_df = _aggregate(ship_df, "out_qty")

    # Merge – outer join so we keep SKUs that only moved one way
    merged = pd.merge(in_df, out_df, on="sku_id", how="outer").fillna({
        "in_qty": 0,
        "out_qty": 0,
    })

    # Velocity calculations
    horizon = _dt.timedelta(days=horizon_days).days
    merged["avg_daily_inbound"] = merged["in_qty"] / horizon
    merged["avg_daily_outbound"] = merged["out_qty"] / horizon

    # Replace NaT with None for SQL compatibility
    for col in merged.columns:
        if "first_" in col or "last_" in col:
            merged[col] = merged[col].where(merged[col].notna(), None)

    # --- Persist ------------------------------------------------------------
    # start a new run header row so we can FK stats
    run = AnalysisRun()        # run_datetime has default=func.now()
    session.add(run)
    session.flush()                   # materialise run.id

    for row in merged.to_dict(orient="records"):
        stat = SKUStat(
            run_id=run.id,
            sku_id=row["sku_id"],
            first_receipt_date=row.get("first_in_qty"),
            last_receipt_date=row.get("last_in_qty"),
            first_shipment_date=row.get("first_out_qty"),
            last_shipment_date=row.get("last_out_qty"),
            in_qty=int(row["in_qty"]),
            out_qty=int(row["out_qty"]),
            avg_daily_inbound=row["avg_daily_inbound"],
            avg_daily_outbound=row["avg_daily_outbound"],
        )
        session.add(stat)

    if managed:
        session.commit()
        session.close()
    else:
        session.flush()

    logger.info("Movement analysis persisted for %s SKUs", len(merged))
    return len(merged)


def fetch_latest_result(session: Session) -> list[dict]:
    """
    Return the full contents of the latest analysis run as a list of
    plain dictionaries, joined with the SKU master to expose `sku_code`.
    """
    run = session.scalar(
        select(AnalysisRun)
        .order_by(AnalysisRun.run_datetime.desc())
        .limit(1)
    )
    if run is None:
        return []

    rows = (
        session.execute(
            select(
                SKUStat,
                core_models.SKUMaster.sku_code
            )
            .where(SKUStat.run_id == run.id)
            .join(core_models.SKUMaster, SKUStat.sku_id == core_models.SKUMaster.id)
        )
        .all()
    )

    results: list[dict] = []
    for stat, sku_code in rows:
        results.append(
            {
                "sku": sku_code,
                "in_qty": stat.in_qty,
                "out_qty": stat.out_qty,
                "avg_daily_inbound": stat.avg_daily_inbound,
                "avg_daily_outbound": stat.avg_daily_outbound,
                "first_receipt_date": stat.first_receipt_date.isoformat() if stat.first_receipt_date else None,
                "last_receipt_date": stat.last_receipt_date.isoformat() if stat.last_receipt_date else None,
                "first_shipment_date": stat.first_shipment_date.isoformat() if stat.first_shipment_date else None,
                "last_shipment_date": stat.last_shipment_date.isoformat() if stat.last_shipment_date else None,
            }
        )
    return results