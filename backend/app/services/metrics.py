

"""
helpers for metric calculation.

The module is intentionally *pure* – every public helper receives an
explicit ``sqlmodel.Session`` and never touches FastAPI / Celery.  This
keeps the code re‑usable from a notebook, a unit‑test or a background
worker alike.

Currently implemented metrics
─────────────────────────────
* **oldness_days** – days since the *oldest* lot currently on hand
* **turnover_ratio** – `(ship_qty_last_N / onhand_qty)`  (N defaults 30 d)

Both metrics are persisted on the **Sku** record itself because they are
needed frequently by the optimiser.  We keep the schema narrow on
purpose – add new columns only when truly necessary.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional, Sequence

from sqlmodel import Session, select, func
from sqlalchemy import literal

from app.models import Inventory, ShipTx, Sku


# --------------------------------------------------------------------------- #
# public helpers                                                              #
# --------------------------------------------------------------------------- #
def recompute_all_sku_metrics(
    session: Session,
    *,
    turnover_window_days: int = 30,
    block_filter: Optional[Sequence[str]] = None,
) -> int:
    """
    Re‑calculate SKU metrics.

    Notes:
        - Always updates ``oldness_days``.
        - Stores 30‑day turnover into ``turnover_30d`` if the column exists,
          otherwise falls back to ``turnover_ratio`` (backward‑compat).
        - Also stores 90‑day turnover into ``turnover_90d`` if the column exists.
        - If `block_filter` is given, inventory aggregation (on-hand & oldest lot) is limited to those blocks.
    """
    today: date = date.today()
    window_start_30: date = today - timedelta(days=turnover_window_days or 30)
    window_start_90: date = today - timedelta(days=90)

    # --- ship qty subqueries (30d and 90d) ---------------------------------
    ship_30 = (
        select(
            ShipTx.sku_id,
            func.coalesce(func.sum(ShipTx.qty), 0).label("ship_qty_30"),
        )
        .where(ShipTx.trandate >= window_start_30)
        .group_by(ShipTx.sku_id)
        .subquery()
    )

    ship_90 = (
        select(
            ShipTx.sku_id,
            func.coalesce(func.sum(ShipTx.qty), 0).label("ship_qty_90"),
        )
        .where(ShipTx.trandate >= window_start_90)
        .group_by(ShipTx.sku_id)
        .subquery()
    )

    # --- on‑hand and oldest lot per SKU ------------------------------------
    oldest_expr = (
        func.min(Inventory.lot_date) if hasattr(Inventory, "lot_date") else literal(None)
    )
    inv_stmt = select(
        Inventory.sku_id,
        func.coalesce(func.sum(Inventory.qty), 0).label("onhand_qty"),
        oldest_expr.label("oldest_lot_date"),
    )
    if block_filter:
        inv_stmt = inv_stmt.where(Inventory.block_code.in_(list(block_filter)))
    inv_subq = inv_stmt.group_by(Inventory.sku_id).subquery()

    # --- join all together --------------------------------------------------
    stmt = (
        select(
            Sku,
            inv_subq.c.onhand_qty,
            inv_subq.c.oldest_lot_date,
            ship_30.c.ship_qty_30,
            ship_90.c.ship_qty_90,
        )
        .select_from(Sku)
        .outerjoin(inv_subq, inv_subq.c.sku_id == Sku.sku_id)
        .outerjoin(ship_30, ship_30.c.sku_id == Sku.sku_id)
        .outerjoin(ship_90, ship_90.c.sku_id == Sku.sku_id)
    )

    updated = 0
    for sku, onhand, oldest_lot, ship_30q, ship_90q in session.exec(stmt):
        onhand = onhand or 0
        ship_30q = ship_30q or 0
        ship_90q = ship_90q or 0

        # ----- oldness -----------------------------------------------------
        if oldest_lot:
            oldness_days = (today - oldest_lot).days
        else:
            # No lot information → treat as 0 (= brand‑new)
            oldness_days = 0

        # ----- turnover ----------------------------------------------------
        t30 = float(ship_30q) / float(onhand) if onhand > 0 else 0.0
        t90 = float(ship_90q) / float(onhand) if onhand > 0 else 0.0

        changed = False
        if sku.oldness_days != oldness_days:
            sku.oldness_days = oldness_days
            changed = True

        # Prefer explicit 30d/90d columns if present; else keep backward compat
        if hasattr(sku, "turnover_30d"):
            if float(sku.turnover_30d or 0.0) != t30:
                sku.turnover_30d = t30
                changed = True
        elif hasattr(sku, "turnover_ratio"):
            if float(sku.turnover_ratio or 0.0) != t30:
                sku.turnover_ratio = t30
                changed = True

        if hasattr(sku, "turnover_90d"):
            if float(sku.turnover_90d or 0.0) != t90:
                sku.turnover_90d = t90
                changed = True

        if changed:
            updated += 1

    session.commit()
    return updated