from __future__ import annotations

"""Stock movement (shipping / receiving) transaction models.

Each transaction references a SKU at the level of `(sku_id, pack_qty)` so that
variants of the same SKU with different inner‐pack quantities are tracked
separately.
"""

from datetime import date

from sqlalchemy import ForeignKeyConstraint
from sqlmodel import Field, SQLModel


class ShipTx(SQLModel, table=True):
    """Outbound (shipping) transaction."""

    __tablename__ = "ship_tx"
    __table_args__ = (
        ForeignKeyConstraint(
            ["sku_id"], ["sku.sku_id"], name="fk_shiptx_sku"
        ),
    )

    # primary key (surrogate)
    id: int | None = Field(default=None, primary_key=True)

    # composite foreign key to sku
    sku_id: str = Field(description="SKU identifier")
    pack_qty: int = Field(description="Inner pack quantity for the SKU")

    # payload
    qty: int = Field(description="Shipped quantity (number of cases)")
    trandate: date = Field(description="Transaction date (shipping date)")


class RecvTx(SQLModel, table=True):
    """Inbound (receiving) transaction."""

    __tablename__ = "recv_tx"
    __table_args__ = (
        ForeignKeyConstraint(
            ["sku_id"], ["sku.sku_id"], name="fk_recvtx_sku"
        ),
    )

    # primary key (surrogate)
    id: int | None = Field(default=None, primary_key=True)

    # foreign key to sku
    sku_id: str = Field(description="SKU identifier")

    # payload
    qty: int = Field(description="Received quantity (number of cases)")
    trandate: date = Field(description="Transaction date (receiving date)")
    lot: str | None = Field(default=None, description="Lot code for traceability")
