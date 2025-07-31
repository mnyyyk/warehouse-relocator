"""
app/db/models_analysis.py
-------------------------
SQLModel table classes for *analysis* artefacts.

They live in a separate module so that the core transactional models in
app/db/models.py stay lean.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, TYPE_CHECKING

from sqlalchemy import Column, DateTime, ForeignKey, func
#
from sqlmodel import Field, SQLModel, Relationship

if TYPE_CHECKING:  # circular‑import safe‑guard
    from .models import SKUMaster  # noqa: F401


class AnalysisRun(SQLModel, table=True):
    """One row per execution of the analysis job."""

    __tablename__ = "analysis_runs"

    id: Optional[int] = Field(default=None, primary_key=True)
    run_datetime: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(
            DateTime(timezone=True),
            server_default=func.now(),
            nullable=False,
            index=True,
        ),
    )
    comment: Optional[str] = Field(
        default=None, description="Free‑form note (e.g. commit hash)."
    )

    # ───────────────── relationships ───────────────── #
    sku_stats: list["SKUStat"] = Relationship(
        back_populates="run",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )

    # ---------------- convenience -------------------- #
    def __repr__(self) -> str:  # pragma: no cover
        return f"<AnalysisRun(id={self.id}, at={self.run_datetime:%F %T})>"


class SKUStat(SQLModel, table=True):
    """Per‑SKU statistics produced by an AnalysisRun."""

    __tablename__ = "sku_stats"

    id: Optional[int] = Field(default=None, primary_key=True)

    run_id: int = Field(
        sa_column=Column(
            ForeignKey("analysis_runs.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        description="FK → analysis_runs.id",
    )
    sku_id: int = Field(
        sa_column=Column(ForeignKey("skus.id", ondelete="CASCADE"), nullable=False, index=True),
        description="FK → skus.id",
    )

    # Raw metrics (all *cases*)
    inbound_30d: int = Field(nullable=False, description="Receipts last 30 days")
    outbound_30d: int = Field(nullable=False, description="Shipments last 30 days")

    # Derived
    turnover_ratio: Optional[float] = Field(default=None)
    percentile_rank: Optional[float] = Field(default=None)
    abc_class: Optional[str] = Field(default=None, max_length=1)

    # ───────────────── relationships ───────────────── #
    run: "AnalysisRun" = Relationship(back_populates="sku_stats")
    sku: "SKUMaster" = Relationship(back_populates="analysis_stats")

    # ---------------- convenience -------------------- #
    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<SKUStat(sku={self.sku_id}, run={self.run_id}, "
            f"out30={self.outbound_30d}, class={self.abc_class})>"
        )


# -----------------------------------------------------------------------------
# Backwards‑compatibility alias
# -----------------------------------------------------------------------------
AnalysisResult = SKUStat

# Late model rebuild to wire up the reverse relationship on SKUMaster
from .models import SKUMaster  # noqa: E402

if not hasattr(SKUMaster, "analysis_stats"):
    SKUMaster.analysis_stats: list["SKUStat"] = Relationship(
        back_populates="sku",
    )
    SKUMaster.model_rebuild()

# Ensure forward references are resolved
AnalysisRun.model_rebuild()
SKUStat.model_rebuild()
