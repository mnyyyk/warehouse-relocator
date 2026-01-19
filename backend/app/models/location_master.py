

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Integer,
    JSON,
    Numeric,
    String,
    UniqueConstraint,
    Index,
    func,
    event,
)
# Prefer app.db.Base; fall back to a local declarative base in dev
try:
    from app.core.database import Base
except Exception:
    from sqlalchemy.orm import declarative_base
    Base = declarative_base()


class LocationMaster(Base):
    """Location master defining all slots (including empty ones).

    Key characteristics:
      * (block_code, quality_name, level, column, depth) is unique
      * numeric_id: 8-digit string f"{level:03d}{column:03d}{depth:02d}" for joining with inventory.location_id
      * can_receive: whether the slot can be used as a destination
      * highness: if True, height is 1500mm; otherwise 1300mm
      * capacity_m3 is auto-computed from bay_*_mm
    """

    __tablename__ = "location_master"

    id = Column(Integer, primary_key=True, index=True)

    # Zone attributes
    block_code = Column(String(8), nullable=False, index=True)
    quality_name = Column(String(16), nullable=False, default="良品", index=True)

    # Coordinates
    level = Column(Integer, nullable=False)   # 列（段）
    column = Column(Integer, nullable=False)  # 連（列）
    depth = Column(Integer, nullable=False)   # 段（連）

    # Derived identifiers
    numeric_id = Column(String(8), nullable=False, index=True)  # e.g. 00100101
    display_code = Column(String(64))  # e.g. B-001-019-01

    # Availability / attributes
    can_receive = Column(Boolean, nullable=False, default=True)
    highness = Column(Boolean, nullable=False, default=False)

    # Physical dimensions (mm)
    bay_width_mm = Column(Integer, nullable=False, default=1000)
    bay_depth_mm = Column(Integer, nullable=False, default=1000)
    bay_height_mm = Column(Integer, nullable=False, default=1300)

    # Capacity (m^3) – auto-computed from bay_*_mm
    capacity_m3 = Column(Numeric(12, 6), nullable=False)

    # Optional metadata / lifecycle
    effective_from = Column(Date)
    disabled_from = Column(Date)
    meta_json = Column(JSON)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint(
            "block_code",
            "quality_name",
            "level",
            "column",
            "depth",
            name="uq_loc_master_zone",
        ),
        Index(
            "ix_loc_master_zone",
            "block_code",
            "quality_name",
            "can_receive",
            "column",
        ),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<LocationMaster {self.block_code}-{int(self.column):03d}-"
            f"{int(self.depth):02d}-{int(self.level):03d} high={self.highness} "
            f"can_recv={self.can_receive}>"
        )


# --- Event helpers to keep derived fields consistent ------------------------

def _compute_numeric_id(target: LocationMaster) -> None:
    target.numeric_id = f"{int(target.level):03d}{int(target.column):03d}{int(target.depth):02d}"


def _apply_height_by_highness(target: LocationMaster) -> None:
    # Enforce height policy: 1500mm for highness, else 1300mm
    target.bay_width_mm = target.bay_width_mm or 1000
    target.bay_depth_mm = target.bay_depth_mm or 1000
    if bool(target.highness):
        target.bay_height_mm = 1500
    else:
        # If unset/None, default to normal height
        target.bay_height_mm = target.bay_height_mm or 1300


def _compute_capacity(target: LocationMaster) -> None:
    # Convert mm^3 -> m^3 using 1e9 factor
    w = int(target.bay_width_mm or 1000)
    d = int(target.bay_depth_mm or 1000)
    h = int(target.bay_height_mm or (1500 if target.highness else 1300))
    cap = (Decimal(w) * Decimal(d) * Decimal(h)) / Decimal(1_000_000_000)
    target.capacity_m3 = cap.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
    # Display code for human readability
    if not target.display_code:
        target.display_code = (
            f"{target.block_code}-{int(target.column):03d}-{int(target.depth):02d}-{int(target.level):03d}"
        )


@event.listens_for(LocationMaster, "before_insert")
def _before_insert(mapper, connection, target: LocationMaster) -> None:  # pragma: no cover
    _apply_height_by_highness(target)
    _compute_numeric_id(target)
    _compute_capacity(target)


@event.listens_for(LocationMaster, "before_update")
def _before_update(mapper, connection, target: LocationMaster) -> None:  # pragma: no cover
    _apply_height_by_highness(target)
    _compute_numeric_id(target)
    _compute_capacity(target)