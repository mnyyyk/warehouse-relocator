"""
app/db/crud_analysis.py
-----------------------

Thin CRUD helpers for the :class:`~app.db.models.AnalysisResult` table that
stores computed warehouse‑level KPIs such as cube utilisation, turnover ratio,
etc.  These helpers are **intentionally minimal** so that higher‑level service
code (e.g. in ``app/services/analysis.py``) can remain database‑agnostic.

All functions expect a *SQLAlchemy* ``Session`` that is typically provided
by :pydata:`app.db.SessionLocal`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .models import AnalysisResult


# --------------------------------------------------------------------------- #
# Write helpers                                                               #
# --------------------------------------------------------------------------- #
def insert_analysis_results(
    session: Session,
    metrics: Dict[str, float],
    run_datetime: Optional[datetime] = None,
) -> None:
    """
    Persist a *set* of metrics produced by a single analysis run.

    Parameters
    ----------
    session :
        An active SQLAlchemy session.
    metrics :
        Mapping ``{metric_name: metric_value}``.
    run_datetime :
        Timestamp of the analysis run.  Defaults to ``utcnow()`` when omitted.
    """
    if run_datetime is None:
        run_datetime = datetime.utcnow()

    for name, value in metrics.items():
        session.add(
            AnalysisResult(
                run_datetime=run_datetime,
                metric_name=name,
                metric_value=value,
            )
        )
    # Caller is responsible for committing/rolling‑back.


# --------------------------------------------------------------------------- #
# Read helpers                                                                #
# --------------------------------------------------------------------------- #
def get_latest_analysis(session: Session) -> Dict[str, float]:
    """
    Return the **most recent** set of metrics as a ``{name: value}`` dict.

    If the table is empty the function returns an *empty dict*.
    """
    latest_ts = session.execute(
        select(func.max(AnalysisResult.run_datetime))
    ).scalar_one_or_none()

    if latest_ts is None:
        return {}

    rows = session.execute(
        select(AnalysisResult)
        .where(AnalysisResult.run_datetime == latest_ts)
    ).scalars()

    return {row.metric_name: row.metric_value for row in rows}


def get_metric_history(
    session: Session,
    metric_name: str,
    limit: int = 100,
) -> List[AnalysisResult]:
    """
    Retrieve the time‑series history for a given *metric* (most recent first).

    Parameters
    ----------
    metric_name :
        Name of the metric to retrieve, e.g. ``"cube_utilisation"``.
    limit :
        Maximum number of rows to return (default 100).

    Returns
    -------
    list[AnalysisResult]
        ORM objects ordered by ``run_datetime`` *descending*.
    """
    result = session.execute(
        select(AnalysisResult)
        .where(AnalysisResult.metric_name == metric_name)
        .order_by(AnalysisResult.run_datetime.desc())
        .limit(limit)
    ).scalars().all()

    return list(result)
