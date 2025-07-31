

"""
app/routers/analysis.py
~~~~~~~~~~~~~~~~~~~~~~~
FastAPI endpoints for kicking‑off the offline analysis step and
retrieving the latest stored results.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from typing import Generator
from ..db import SessionLocal
from ..services import analysis as analysis_service
from ..db.models_analysis import AnalysisResult  # for response typing / mypy only

 # --------------------------------------------------------------------------- #
# Database session dependency                                                 #
# --------------------------------------------------------------------------- #
def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a SQLAlchemy session and guarantees it is
    closed after the request.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

router = APIRouter(
    prefix="/analysis",
    tags=["analysis"],
)


# --------------------------------------------------------------------------- #
# Run analysis                                                                #
# --------------------------------------------------------------------------- #
@router.post(
    "/run",
    summary="Trigger the background analysis that ranks SKUs by activity.",
    response_model=dict,
)
def run_analysis(db: Session = Depends(get_db)) -> dict[str, str | int]:
    """
    Kick off the analysis service.

    It reads **all** receipts & shipments currently in the DB, computes the
    inbound / outbound turnover, stores the results, and returns the primary
    key of the new :class:`~app.db.models_analysis.AnalysisResult` row.
    """
    try:
        result: AnalysisResult = analysis_service.run_analysis(db)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc))

    return {"status": "ok", "analysis_id": result.id}


# --------------------------------------------------------------------------- #
# Fetch latest analysis result                                                #
# --------------------------------------------------------------------------- #
@router.get(
    "/latest",
    summary="Return the most recent analysis result (as JSON).",
)
def get_latest(db: Session = Depends(get_db)):
    """
    Convenience helper so the front‑end can retrieve the last completed
    analysis run without knowing its primary key.
    """
    result = analysis_service.fetch_latest_result(db)
    if result is None:
        raise HTTPException(status_code=404, detail="No analysis results yet")

    # `model_dump()` is the SQLModel / Pydantic‑v2 helper that turns the row
    # into a plain JSON‑serialisable dict.
    return result.model_dump()