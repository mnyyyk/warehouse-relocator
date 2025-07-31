"""
app/db/models.py
----------------
Typed SQLAlchemy 2.0 ORM models for the local SQLite (later RDS) database.
"""

from typing import List, Optional
from datetime import date, datetime

from sqlalchemy import Column, JSON, func, String, Date, DateTime
from sqlmodel import SQLModel, Field, Relationship


class SKUMaster(SQLModel, table=True):
    __tablename__ = "skus"
    """
    Master data for each SKU (stock keeping unit).

    NOTE: We use SQLModel's Field / Relationship helpers instead of the
    SQLAlchemy‑2.0 typed ORM (`Mapped[...]`) because SQLModel still relies
    on Pydantic for schema generation.  `Mapped` types cause
    ``PydanticSchemaGenerationError`` with SQLModel.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    sku_code: str = Field(sa_column=Column("sku_code", String(64), unique=True, nullable=False, index=True))
    description: Optional[str] = Field(default=None, nullable=True)
    volume_l: float = Field(nullable=False, description="Per‑case volume in litres")

    # Relationships
    receipts: List["Receipt"] = Relationship(back_populates="sku")
    shipments: List["Shipment"] = Relationship(back_populates="sku")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<SKU(code={self.sku_code}, vol={self.volume_l}L)>"

# --------------------------------------------------------------------- #

class Receipt(SQLModel, table=True):
    __tablename__ = "receipts"
    """Inbound transactions (cases received)."""

    id: Optional[int] = Field(default=None, primary_key=True)
    sku_id: int = Field(foreign_key="skus.id", index=True)
    receipt_date: date = Field(sa_column=Column("receipt_date", Date, nullable=False, index=True))
    qty: int = Field(nullable=False)

    # Relationship
    sku: SKUMaster = Relationship(back_populates="receipts")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Receipt(sku={self.sku_id}, receipt_date={self.receipt_date}, qty={self.qty})>"

# --------------------------------------------------------------------- #

class Shipment(SQLModel, table=True):
    __tablename__ = "shipments"
    """Outbound transactions (cases shipped)."""

    id: Optional[int] = Field(default=None, primary_key=True)
    sku_id: int = Field(foreign_key="skus.id", index=True)
    ship_date: date = Field(sa_column=Column("ship_date", Date, nullable=False, index=True))
    qty: int = Field(nullable=False)

    sku: SKUMaster = Relationship(back_populates="shipments")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Shipment(sku={self.sku_id}, ship_date={self.ship_date}, qty={self.qty})>"

# --------------------------------------------------------------------- #

class AnalysisResult(SQLModel, table=True):
    """
    Stores aggregated metrics per analysis run so that they can be reused
    by subsequent optimisation requests.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    run_datetime: datetime = Field(sa_column=Column(DateTime, server_default=func.now(), nullable=False, index=True))
    metrics: dict = Field(sa_column=Column(JSON, nullable=False))

    def __repr__(self) -> str:  # pragma: no cover
        return f"<AnalysisResult(id={self.id}, run={self.run_datetime})>"

SKUMaster.update_forward_refs()
Receipt.update_forward_refs()
Shipment.update_forward_refs()