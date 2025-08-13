"""add lot_date to inventory

Revision ID: a74f1cde4037
Revises: 425357a7886b
Create Date: 2025-08-12 00:13:46.923839

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a74f1cde4037'
down_revision: Union[str, Sequence[str], None] = '425357a7886b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: add lot_date column + index; optionally backfill on PostgreSQL."""
    # add column & index (batch mode for SQLite compatibility)
    with op.batch_alter_table("inventory") as batch_op:
        batch_op.add_column(sa.Column("lot_date", sa.Date(), nullable=True))
        batch_op.create_index("ix_inventory_lot_date", ["lot_date"], unique=False)

    # Optional: lightweight backfill for PostgreSQL (YYYYMMDD / YYYYMM→01補完)
    bind = op.get_bind()
    if bind is not None and getattr(bind, "dialect", None) and bind.dialect.name == "postgresql":
        op.execute(
            """
            UPDATE inventory i
            SET lot_date = to_date(m.m, 'YYYYMMDD')
            FROM (
              SELECT location_id, sku_id, lot, pack_qty,
                     substring(lot FROM '(20[0-9]{2})(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])') AS m
              FROM inventory
            ) m
            WHERE i.location_id=m.location_id AND i.sku_id=m.sku_id
              AND i.lot=m.lot AND i.pack_qty=m.pack_qty
              AND i.lot_date IS NULL AND m.m IS NOT NULL;
            """
        )
        op.execute(
            """
            UPDATE inventory i
            SET lot_date = to_date(m.m || '01', 'YYYYMMDD')
            FROM (
              SELECT location_id, sku_id, lot, pack_qty,
                     substring(lot FROM '(20[0-9]{2})(0[1-9]|1[0-2])') AS m
              FROM inventory
            ) m
            WHERE i.location_id=m.location_id AND i.sku_id=m.sku_id
              AND i.lot=m.lot AND i.pack_qty=m.pack_qty
              AND i.lot_date IS NULL AND m.m IS NOT NULL;
            """
        )


def downgrade() -> None:
    """Downgrade schema: drop index and column."""
    with op.batch_alter_table("inventory") as batch_op:
        try:
            batch_op.drop_index("ix_inventory_lot_date")
        except Exception:
            pass
        batch_op.drop_column("lot_date")
