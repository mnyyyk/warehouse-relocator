

"""
Celery background tasks for analytics.

This task delegates metric computation to `app.services.metrics` so that
all business logic lives in one place (no drift between codepaths).

Exposed task:
* ``sku_metrics(window_days=90)`` – recomputes per‑SKU metrics.

Recomputed metrics (if columns exist on the `sku` table):
- oldness_days
- turnover_30d (or legacy: turnover_ratio)
- turnover_90d
"""

from __future__ import annotations

import datetime as _dt
from celery import shared_task
from sqlmodel import Session

from app.core.database import engine
from app.services.metrics import recompute_all_sku_metrics

from typing import Sequence, Union
from sqlalchemy import select, func
from app.models.sku import Sku
import time

def _normalize_blocks(block_codes: Union[Sequence[str], str, None]) -> list[str] | None:
    """Accept list/tuple or comma-separated string and normalize to list[str] or None."""
    if block_codes is None:
        return None
    if isinstance(block_codes, str):
        parts = [p.strip() for p in block_codes.split(",") if p.strip()]
        return parts or None
    # Sequence[str]
    parts = [str(p).strip() for p in block_codes if str(p).strip()]
    return parts or None


@shared_task(name="analysis.sku_metrics")
def sku_metrics(
    window_days: int = 90,
    block_codes: Union[Sequence[str], str, None] = None,
) -> dict:
    """
    Celery task: recompute SKU metrics (optionally for specific blocks).

    Args:
        window_days: turnover window in days (default: 90)
        block_codes: list/tuple of block codes or comma-separated string (e.g.,
            ["B"] or "B,B2") to restrict inventory-based metrics.
    """
    start = time.perf_counter()
    blocks = _normalize_blocks(block_codes)
    with Session(engine) as ses:
        updated = recompute_all_sku_metrics(
            ses,
            turnover_window_days=window_days,
            block_filter=blocks,
        )
        total_skus = ses.exec(select(func.count()).select_from(Sku)).one()
    elapsed = round(time.perf_counter() - start, 3)
    return {
        "updated": int(updated),
        "total_skus": int(total_skus or 0),
        "window_days": int(window_days),
        "blocks": list(blocks or []),
        "elapsed_sec": elapsed,
        "run_at": _dt.datetime.utcnow().isoformat(),
    }