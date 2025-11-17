"""
File‑upload router.

Provides four endpoints:

* POST /v1/upload/sku
* POST /v1/upload/inventory
* POST /v1/upload/ship_tx
* POST /v1/upload/recv_tx

Each accepts a single CSV / Excel file (multipart/form‑data) and
inserts rows into PostgreSQL via SQLModel.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Callable
from typing import Any, Dict, List, Optional, Literal, Generator
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, Query, Request
from fastapi.responses import StreamingResponse
from sqlmodel import Session, SQLModel

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy import inspect, select as _sa_select, text, insert as sa_insert
import inspect as pyinspect  # stdlib inspect (do not shadow SQLAlchemy.inspect)
from pathlib import Path

from app.utils.file_parser import read_dataframe
from uuid import uuid4

import os
import logging
import math
import importlib
import io

# ==== AI trace JSONL logger (file-based, best-effort) ====
import json, time
AI_LOG_PATH = os.getenv(
    "AI_LOG_JSONL",
    os.path.join(os.path.dirname(__file__), "../../logs/ai_planner.jsonl"),
)

def _jsonl_log(event: dict) -> None:
    """Write one JSON object per line. Never raise."""
    try:
        path = os.path.abspath(AI_LOG_PATH)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        base = {"ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "app": "warehouse-optimizer"}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({**base, **event}, ensure_ascii=False) + "\n")
    except Exception:
        # swallow any logging errors
        pass

# Additional imports for analysis/relocation endpoints
from pydantic import BaseModel
from app.services.metrics import recompute_all_sku_metrics
from app.services.optimizer import (
    plan_relocation,
    OptimizerConfig,
    get_summary_report,
    get_last_summary_report,
    get_last_rejection_debug,
    get_last_relocation_debug,
    get_current_trace_id,
    sse_events,
)


# --------------------------------------------------------------------------- #
# util: SKU ID canonicalisation                                               #
# --------------------------------------------------------------------------- #
import unicodedata
import re
_SKU_ID_RE = re.compile(r"\s+")

def _normalize_sku_id(value: str | int) -> str:
    """
    Return a canonical SKU code:
      1. convert to str and strip outer whitespace
      2. Unicode NFKC normalize (full‑width → half‑width, etc.)
      3. remove any internal whitespace
      4. strip non‑alphanumerics **except hyphen '-'**
      5. upper‑case
    """
    txt = unicodedata.normalize("NFKC", str(value).strip())
    txt = _SKU_ID_RE.sub("", txt)

    # Strip any residual non‑alphanumeric glyphs (e.g. NBSP, weird control codes), but preserve hyphen '-'
    txt = re.sub(r"[^A-Za-z0-9-]", "", txt)

    return txt.upper()

from app.core.database import engine
from app.models import Inventory, RecvTx, ShipTx, Sku
# Prefer the dedicated SQLModel in app.models; fall back to an inline dev stub
try:
    from app.models.location_master import LocationMaster  # type: ignore
except Exception:
    from sqlmodel import Field
    from typing import Optional
    from sqlalchemy import Column, Integer, UniqueConstraint
    from sqlalchemy.dialects.postgresql import JSONB

    class LocationMaster(SQLModel, table=True):
        __tablename__ = "location_master"
        __table_args__ = (
            UniqueConstraint("block_code", "quality_name", "level", "column", "depth", name="uq_location_master_bq_lcd"),
        )

        id: Optional[int] = Field(default=None, primary_key=True)
        block_code: str = Field(index=True, description="ブロック略称")
        quality_name: Optional[str] = Field(default=None, index=True, description="品質区分名")
        level: int = Field(index=True, description="列（段）= level")
        column: int = Field(index=True, description="連（列）= column")
        depth: int = Field(index=True, description="段（連）= depth")

        # Derived / display
        numeric_id: Optional[str] = Field(default=None, description="LLLCCCDD 文字列など")
        display_code: Optional[str] = Field(default=None, description="例: B-001-19-01")

        # Policy flags
        can_receive: bool = Field(default=True, index=True)
        highness: bool = Field(default=False, index=True)

        # Physical dimensions (mm) and capacity (m^3)
        bay_width_mm: Optional[int] = None
        bay_depth_mm: Optional[int] = None
        bay_height_mm: Optional[int] = None
        capacity_m3: Optional[float] = None

        # Lifecycle & misc
        effective_from: Optional[datetime] = None
        disabled_from: Optional[datetime] = None
        meta_json: Optional[dict] = Field(default=None, sa_column=Column(JSONB))


# --------------------------------------------------------------------------- #
# dev‑only: ensure required tables exist                                       #
# --------------------------------------------------------------------------- #


def _ensure_tables() -> None:
    """
    Dev‑convenience: auto‑create any tables that are missing.

    In production, run Alembic migrations instead.
    """
    insp = inspect(engine)
    missing = [
        mdl.__tablename__
        for mdl in (Sku, Inventory, ShipTx, RecvTx, LocationMaster)
        if not insp.has_table(mdl.__tablename__)
    ]
    if missing:
        SQLModel.metadata.create_all(
            engine,
            tables=[
                mdl.__table__
                for mdl in (Sku, Inventory, ShipTx, RecvTx, LocationMaster)
                if mdl.__tablename__ in missing
            ],
        )

# Auto‑create missing tables only when explicitly enabled.
# Set environment variable AUTO_CREATE_TABLES=true (default) to allow this in dev.
# In production where Alembic migrations are applied, set it to false/0.
if (os.getenv("AUTO_CREATE_TABLES", "true") or "true").strip().lower() in ("1", "true", "yes", "on", "y", "t"):
    _ensure_tables()
    def _ensure_optional_columns() -> None:
        """Dev‑convenience: ensure optional columns exist (non‑breaking).

        In production, manage schema via Alembic migrations instead.
        """
        try:
            with engine.begin() as conn:
                # inventory.cases (double precision) – 箱数を保持
                conn.exec_driver_sql(
                    "ALTER TABLE inventory ADD COLUMN IF NOT EXISTS cases double precision"
                )
                # inventory.quality_name – 品質区分名
                conn.exec_driver_sql(
                    "ALTER TABLE inventory ADD COLUMN IF NOT EXISTS quality_name text"
                )
                # inventory.pack_qty – 入数のスナップショット
                conn.exec_driver_sql(
                    "ALTER TABLE inventory ADD COLUMN IF NOT EXISTS pack_qty integer"
                )
                # inventory.level/column/depth – ロケの分解（必要に応じて）
                conn.exec_driver_sql(
                    "ALTER TABLE inventory ADD COLUMN IF NOT EXISTS level integer"
                )
                conn.exec_driver_sql(
                    'ALTER TABLE inventory ADD COLUMN IF NOT EXISTS "column" integer'
                )
                conn.exec_driver_sql(
                    "ALTER TABLE inventory ADD COLUMN IF NOT EXISTS depth integer"
                )
                # inventory.lot_date – 派生日付 (DATE)
                conn.exec_driver_sql(
                    "ALTER TABLE inventory ADD COLUMN IF NOT EXISTS lot_date date"
                )
                # inventory.allocated_qty – 引当数（個）: 無ければ0で追加
                conn.exec_driver_sql(
                    "ALTER TABLE inventory ADD COLUMN IF NOT EXISTS allocated_qty integer DEFAULT 0"
                )
        except Exception:
            # Dev 用の安全弁: 存在しないテーブル等での失敗を握りつぶす。
            # 本番では Alembic 側で確実に適用すること。
            pass
    _ensure_optional_columns()
# NOTE: In production, DB schema is managed by Alembic migrations.


router = APIRouter(prefix="/v1/upload", tags=["upload"])
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# error CSV helper                                                            #
# --------------------------------------------------------------------------- #

ERROR_DIR = Path("/tmp/upload_errors")
ERROR_DIR.mkdir(parents=True, exist_ok=True)

def _save_error_csv(errors: list[dict], base_url: str = "") -> str:
    """
    Save a list of {'row': int, 'message': str} dicts as CSV and
    return the full URL (e.g. ``https://api.example.com/files/err_<uuid>.csv``).
    The FastAPI app must mount ``StaticFiles`` at ``/files`` separately.
    
    Args:
        errors: List of error dictionaries
        base_url: Base URL of the API (e.g., "https://api.warehouse-optimizer.net")
    """
    if not errors:
        return ""
    df = pd.DataFrame(errors)
    fname = f"err_{uuid4().hex}.csv"
    fpath = ERROR_DIR / fname
    df.to_csv(fpath, index=False, encoding="utf-8-sig")
    logger.info("Saved error CSV: %s (%d errors)", fpath, len(errors))
    # Verify file was created
    if fpath.exists():
        logger.info("Error CSV file exists, size: %d bytes", fpath.stat().st_size)
    else:
        logger.error("Error CSV file was NOT created at %s", fpath)
    
    # Force HTTPS for production (App Runner terminates SSL at load balancer)
    if base_url:
        base_url = base_url.replace("http://", "https://")
        return f"{base_url.rstrip('/')}/files/{fname}"
    return f"/files/{fname}"

# --------------------------------------------------------------------------- #
# helper: robust upsert for location_master (handles constraint order/name)   #
# --------------------------------------------------------------------------- #

def _upsert_location_master(session: Session, rows: list[dict], set_dict: dict | None) -> None:
    """Dialect-safe UPSERT for `location_master`.

    - Detects DB dialect (PostgreSQL vs SQLite) and uses the appropriate
      insert(...).on_conflict_do_update API.
    - If `set_dict` is provided by caller, it will be used for PostgreSQL.
      For SQLite, the update mapping is recomputed from `stmt.excluded` to
      avoid backend-specific `text('excluded.col')` constructs.
    - If unique constraint name varies, falls back to index_elements with the
      canonical column list [block_code, quality_name, level, column, depth].
    """
    if not rows:
        return

    from sqlalchemy.dialects.postgresql import insert as _pg_insert
    from sqlalchemy.dialects.sqlite import insert as _sqlite_insert
    from sqlalchemy import insert as _sa_insert

    bind = session.get_bind()
    dialect_name = getattr(getattr(bind, "dialect", None), "name", "") if bind is not None else ""

    # Try to discover a named unique constraint matching the key set
    try:
        insp = inspect(bind)
        uqs = insp.get_unique_constraints("location_master") or []
    except Exception:
        uqs = []

    target_cols = {"block_code", "quality_name", "level", "column", "depth"}
    on_constraint_name: str | None = None
    for uq in uqs:
        cols = set((uq.get("column_names") or []))
        if cols == target_cols:
            on_constraint_name = uq.get("name")
            break

    tbl = LocationMaster.__table__

    if dialect_name == "postgresql":
        stmt = _pg_insert(tbl).values(rows)
        # Prefer using provided set_dict if available; otherwise update all non-key columns
        if set_dict is None:
            update_cols = {
                c.name: getattr(stmt.excluded, c.name)
                for c in tbl.c
                if c.name not in target_cols
            }
        else:
            update_cols = set_dict

        if on_constraint_name:
            try:
                stmt = stmt.on_conflict_do_update(constraint=on_constraint_name, set_=update_cols)
            except TypeError:
                stmt = stmt.on_conflict_do_update(
                    index_elements=["block_code", "quality_name", "level", "column", "depth"],
                    set_=update_cols,
                )
        else:
            stmt = stmt.on_conflict_do_update(
                index_elements=["block_code", "quality_name", "level", "column", "depth"],
                set_=update_cols,
            )
    elif dialect_name == "sqlite":
        # SQLite supports ON CONFLICT DO UPDATE via SQLAlchemy's sqlite_insert
        stmt = _sqlite_insert(tbl).values(rows)
        update_cols = {
            c.name: getattr(stmt.excluded, c.name)
            for c in tbl.c
            if c.name not in target_cols
        }
        try:
            stmt = stmt.on_conflict_do_update(
                index_elements=["block_code", "quality_name", "level", "column", "depth"],
                set_=update_cols,
            )
        except Exception:
            # Fallback: if the driver is too old, do plain INSERT (may duplicate)
            stmt = _sa_insert(tbl).values(rows)
    else:
        # Other dialects: plain INSERT; duplicates may be ignored by DB policy
        stmt = _sa_insert(tbl).values(rows)

    session.execute(stmt)
    session.commit()

# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #


def get_session() -> Generator[Session, None, None]:  # dependency
    with Session(engine) as ses:
        yield ses




def _generic_insert(
    df: pd.DataFrame,
    model: type[SQLModel],
    mapper: Callable[[pd.Series], object],
    session: Session,
    use_upsert: bool = True,
) -> dict:
    """Insert rows with upsert (ON CONFLICT DO UPDATE).

    - 同じアップロード内で **一次キーが重複** している行は
      「最後に現れた行」を優先して 1 行にまとめる。
    - まとめた行を INSERT … ON CONFLICT DO UPDATE で upsert する。
    - 実際に upsert した行数を `success_rows` / `upserted_rows` として返す。
    
    Performance optimizations:
    - Uses df.itertuples() instead of df.iterrows() for 3-5x faster iteration
    - Increased default batch size for fewer SQL round trips
    - Optimized deduplication logic
    """
    from time import perf_counter
    t0 = perf_counter()
    total = len(df)
    objects: list[SQLModel] = []
    errors: list[dict] = []

    # DataFrame → モデルオブジェクトへ変換
    for idx, row in df.iterrows():
        try:
            objects.append(mapper(row))
        except Exception as exc:
            errors.append({"row": int(idx) + 2, "message": str(exc)})
    t_map = perf_counter()
    try:
        logger.info("_generic_insert[%s]: mapping done rows=%s errors=%s elapsed=%.3fs", getattr(model, "__tablename__", str(model)), len(objects), len(errors), (t_map - t0))
    except Exception:
        pass
    # Performance debug output
    print(f"[PERF] _generic_insert mapping: {(t_map - t0):.3f}s for {len(objects)} rows ({model.__tablename__ if hasattr(model, '__tablename__') else model})")

    # ---------- 0‑b.  外部キー整合性チェック -------------------------------
    #
    # Inventory / ShipTx  では (sku_id, pack_qty) の複合キーで SKU マスタを
    # 参照する。一方、 RecvTx (入荷実績) は「入り数」列がなく pack_qty を
    # 判別できないため、**sku_id だけ** で突合する。
    #
    if model is RecvTx:
        # Ensure referenced SKUs exist; auto-create minimal SKU rows for unknowns.
        # Pull a scalar list of sku_id values and normalise them.
        sku_values = session.exec(_sa_select(Sku.sku_id)).all()
        existing: set[str] = {_normalize_sku_id(v) for v in sku_values}

        incoming_ids: set[str] = {_normalize_sku_id(o.sku_id) for o in objects}
        missing = sorted(incoming_ids - existing)
        if missing:
            # Auto-create with default pack_qty=1 and no dimensions
            session.bulk_save_objects([Sku(sku_id=m, pack_qty=1) for m in missing])
            session.commit()
            existing.update(missing)

        # Canonicalise all objects now that SKUs are ensured
        for obj in objects:
            obj.sku_id = _normalize_sku_id(obj.sku_id)

    elif model is ShipTx:
        # ShipTx は CSV に「入数」が含まれないケースが多い。
        # そのため、SKU マスタから sku_id → pack_qty を引き当てて埋める。
        sku_pack_map: dict[str, int] = {
            _normalize_sku_id(s): pq
            for (s, pq) in session.exec(_sa_select(Sku.sku_id, Sku.pack_qty)).all()
        }

        # Auto-create unknown SKUs with default pack_qty=1
        incoming_ids: set[str] = {_normalize_sku_id(o.sku_id) for o in objects}
        missing = sorted(incoming_ids - set(sku_pack_map.keys()))
        if missing:
            session.bulk_save_objects([Sku(sku_id=m, pack_qty=1) for m in missing])
            session.commit()
            # refresh map including new entries
            sku_pack_map.update({m: 1 for m in missing})

        filtered: list[SQLModel] = []
        for obj in objects:
            canon = _normalize_sku_id(obj.sku_id)
            pack = sku_pack_map.get(canon, 1)
            obj.sku_id = canon
            obj.pack_qty = pack  # CSV 値は無視し、SKU マスタの入数を採用（未知は1）
            filtered.append(obj)
        objects = filtered

    elif model is Inventory:
        # Inventory CSVには入数が無い前提。
        # SKUマスタから sku_id → pack_qty を引き当てて埋める（ShipTxと同様）。
        sku_pack_map: dict[str, int] = {
            _normalize_sku_id(s): pq
            for (s, pq) in session.exec(_sa_select(Sku.sku_id, Sku.pack_qty)).all()
        }

        # Auto-create unknown SKUs with default pack_qty=1
        incoming_ids: set[str] = {_normalize_sku_id(o.sku_id) for o in objects}
        missing = sorted(incoming_ids - set(sku_pack_map.keys()))
        if missing:
            session.bulk_save_objects([Sku(sku_id=m, pack_qty=1) for m in missing])
            session.commit()
            sku_pack_map.update({m: 1 for m in missing})

        filtered: list[SQLModel] = []
        for obj in objects:
            canon = _normalize_sku_id(obj.sku_id)
            pack = sku_pack_map.get(canon, 1)

            obj.sku_id = canon
            obj.pack_qty = pack  # CSV値は無視し、SKUマスタの入数を採用（未知は1）

            # cases 補完: 入力に正の cases があれば優先、
            # 無ければ ceil(qty/pack) で補完（端数切り上げ）
            qv = getattr(obj, "qty", None)
            c_in = getattr(obj, "cases", None)
            if c_in is not None and float(c_in) > 0:
                obj.cases = float(c_in)
            elif qv is not None and pack and pack > 0:
                obj.cases = float(math.ceil(float(qv) / float(pack)))
            else:
                # pack が不明/0 の異常系は 0 として扱う（後続の最適化ではスキップ対象）
                obj.cases = 0.0

            filtered.append(obj)
        objects = filtered

    # ---------- ① 一次キーで重複を排除（後勝ち） ---------------------------
    # Resolve PK columns preferring DB inspector (handles composite PK even if model metadata is stale)
    try:
        bind = session.get_bind()
        insp = inspect(bind)
        pk_info = insp.get_pk_constraint(model.__tablename__) or {}
        pk_cols_db = list(pk_info.get("constrained_columns") or [])
        # Obtain actual columns present in the physical table
        real_cols = [c["name"] for c in insp.get_columns(model.__tablename__)]
    except Exception:
        pk_cols_db = []
        # Fall back to model columns if inspector fails
        real_cols = [c.name for c in model.__table__.columns]

    if pk_cols_db:
        pk_cols = pk_cols_db
        pk_src = "db"
    else:
        pk_cols = [c.name for c in model.__table__.primary_key.columns]
        pk_src = "model"

    logger.debug("_generic_insert: table=%s pk_cols=%s (src=%s) upsert=%s", model.__tablename__, pk_cols, pk_src, bool(use_upsert))
    dedup_map: dict[tuple, dict] = {}
    for obj in objects:
        key = tuple(getattr(obj, col) for col in pk_cols)

        # If every primary‑key column is None, this model is using an
        # autoincrement surrogate key (e.g. `id`).  All incoming rows would
        # otherwise share the same `(None,)` key and be deduplicated down to
        # a single record.  To preserve every row, generate a per‑object
        # sentinel that is guaranteed to be unique for the current process.
        if all(v is None for v in key):
            key = (id(obj),)  # `id()` provides a unique identifier per object

        # Convert to plain dict and **remove** autoincrement PK columns that are None
        data = obj.model_dump()
        for pk in pk_cols:
            if data.get(pk) is None:
                data.pop(pk, None)
        # Keep only columns that actually exist in the table. This avoids
        # attempting to insert values for non-existent columns like a
        # surrogate "id" on composite-PK tables (e.g., inventory).
        data = {k: v for k, v in data.items() if k in real_cols}

        dedup_map[key] = data

    rows_data = list(dedup_map.values())
    t_dedup = perf_counter()
    try:
        logger.info("_generic_insert[%s]: dedup done unique_rows=%s elapsed=%.3fs", getattr(model, "__tablename__", str(model)), len(rows_data), (t_dedup - t_map))
    except Exception:
        pass
    # Performance debug output
    print(f"[PERF] _generic_insert dedup: {(t_dedup - t_map):.3f}s for {len(rows_data)} rows")
    affected = len(rows_data)          # 実際に upsert/skip 判定対象の行数（成功数ではない）
    successes = 0                      # 実際にDBに反映できた行数
    if len(objects) != affected:
        logger.info("_generic_insert: dedup %s -> %s by PK %s for %s", len(objects), affected, pk_cols, model.__tablename__)

    # ---------- ② UPSERT  (split into chunks to avoid 32 767‑parameter limit) ----
    if rows_data:
        # --- Calculate a safe batch size ----------------------------------
        #
        # PostgreSQL has a hard ceiling of 32 767 bind parameters per
        # prepared statement.  We work out how many parameters *this*
        # particular model/row will generate and then pick a chunk size
        # that always stays well below that ceiling, with an additional
        # safety margin.
        #
        # 1.  Count only the columns that will actually be bound for this
        #     bulk‑insert (i.e. the keys present in the first row dict),
        #     not every column defined on the model/table.
        # 2.  Compute how many rows fit into a 25 000‑parameter budget
        #     (provides ample headroom under 32 767).
        # 3.  Put an extra hard cap of 500 rows per statement.  This keeps
        #     the SQL string length reasonable and avoids edge cases with
        #     older drivers / proxies that have their own limits.
        #
        # Parameter budget depends on dialect (SQLite has a low 999 limit)
        try:
            bind = session.get_bind()
            _dialect_name = getattr(bind.dialect, "name", "") if bind is not None else ""
        except Exception:
            _dialect_name = ""
        if _dialect_name == "sqlite":
            param_budget = 900  # leave headroom under 999
        elif _dialect_name == "postgresql":
            param_budget = 25_000
        else:
            param_budget = 5_000
        # Optional override via env for tuning (e.g., UPLOAD_PARAM_BUDGET=1200)
        try:
            _env_budget = int(os.getenv("UPLOAD_PARAM_BUDGET", "0") or 0)
            if _env_budget > 0:
                param_budget = int(_env_budget)
        except Exception:
            pass
        params_per_row = max(1, len(rows_data[0]))           # keys actually present
        # OPTIMIZATION: Increase default batch size for better performance
        # Use 1000 as base instead of 500, capped by parameter budget
        max_rows       = min(1000, param_budget // params_per_row)

        total_chunks = 0
        t_sql0 = perf_counter()
        # 方言検出（PRAGMA等は1回だけ適用）
        bind = session.get_bind()
        dialect = getattr(bind.dialect, "name", "") if bind is not None else ""
        dropped_indexes: list[tuple[str, str]] = []  # (name, sql)
        if dialect == "sqlite":
            try:
                session.exec(text("PRAGMA synchronous = OFF"))
            except Exception:
                pass
            try:
                session.exec(text("PRAGMA temp_store = MEMORY"))
            except Exception:
                pass
            # 大量行のときは一時的に二次インデックスを落として速度向上（後で復元）
            try:
                row_threshold = int(os.getenv("UPLOAD_INDEX_DROP_THRESHOLD", "20000") or 20000)
            except Exception:
                row_threshold = 20000
            try:
                if len(rows_data) >= row_threshold:
                    tbl = getattr(model, "__tablename__", None)
                    if tbl:
                        idx_rows = session.exec(
                            text("SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name = :t AND sql IS NOT NULL")
                            .bindparams(t=tbl)
                        ).all()
                        for name, sql in idx_rows:
                            # 主キー由来の index は sqlite_master.sql が NULL のことが多いので上の WHERE で除外済み
                            try:
                                session.exec(text(f"DROP INDEX IF EXISTS {name}"))
                                dropped_indexes.append((str(name), str(sql)))
                            except Exception:
                                pass
                        session.commit()
            except Exception:
                pass
            try:
                session.exec(text("PRAGMA cache_size = -200000"))
            except Exception:
                pass
        for i in range(0, len(rows_data), max_rows):
            chunk = rows_data[i : i + max_rows]
            total_chunks += 1

            do_upsert = bool(use_upsert and pk_cols)
            if use_upsert and not pk_cols:
                logger.warning("_generic_insert: upsert requested but no PK detected for table=%s; falling back to plain INSERT", model.__tablename__)
                do_upsert = False

            # INSERT 文の構築（upsert 無しなら汎用 insert を使用）
            if not do_upsert:
                stmt = sa_insert(model.__table__).values(chunk)
            else:
                if dialect == "postgresql":
                    stmt = pg_insert(model.__table__).values(chunk)
                    # 構築後に excluded を参照して更新カラムを決定
                    update_cols_dict = {
                        c.name: getattr(stmt.excluded, c.name)
                        for c in model.__table__.c
                        if (c.name not in pk_cols) and (c.name in real_cols)
                    }
                    if not update_cols_dict:
                        logger.warning(
                            "_generic_insert: no updatable columns detected for table=%s (pk=%s)",
                            model.__tablename__, pk_cols
                        )
                    stmt = stmt.on_conflict_do_update(
                        index_elements=pk_cols,
                        set_=update_cols_dict,
                    )
                elif dialect == "sqlite":
                    # SQLite でも SQLAlchemy の on_conflict_do_update が利用可能
                    stmt = sqlite_insert(model.__table__).values(chunk)
                    update_cols_dict = {
                        c.name: getattr(stmt.excluded, c.name)
                        for c in model.__table__.c
                        if (c.name not in pk_cols) and (c.name in real_cols)
                    }
                    if not update_cols_dict:
                        logger.warning(
                            "_generic_insert(sqlite): no updatable columns detected for table=%s (pk=%s)",
                            model.__tablename__, pk_cols
                        )
                    try:
                        stmt = stmt.on_conflict_do_update(
                            index_elements=pk_cols,
                            set_=update_cols_dict,
                        )
                    except Exception:
                        # 古いSQLite/ドライバでは on_conflict_do_update が未対応のことがある
                        # その場合は素直に汎用 INSERT にフォールバック（重複は最新行に置換されない）
                        logger.exception("_generic_insert: sqlite on_conflict_do_update unsupported; falling back to plain insert")
                        stmt = sa_insert(model.__table__).values(chunk)
                else:
                    # その他の方言は汎用 INSERT のみ
                    stmt = sa_insert(model.__table__).values(chunk)

            try:
                session.execute(stmt)
                successes += len(chunk)
            except Exception as bulk_exc:
                # バルク実行で失敗 → 1行ずつフォールバックし、失敗分を errors へ集約
                try:
                    logger.exception("_generic_insert: bulk execute failed; falling back to row-by-row insertion")
                except Exception:
                    pass
                session.rollback()

                for row in chunk:
                    try:
                        if not do_upsert:
                            stmt1 = sa_insert(model.__table__).values(row)
                        else:
                            if dialect == "postgresql":
                                stmt1 = pg_insert(model.__table__).values([row])
                                update_cols_dict = {
                                    c.name: getattr(stmt1.excluded, c.name)
                                    for c in model.__table__.c
                                    if (c.name not in pk_cols) and (c.name in real_cols)
                                }
                                stmt1 = stmt1.on_conflict_do_update(
                                    index_elements=pk_cols,
                                    set_=update_cols_dict,
                                )
                            elif dialect == "sqlite":
                                stmt1 = sqlite_insert(model.__table__).values([row])
                                update_cols_dict = {
                                    c.name: getattr(stmt1.excluded, c.name)
                                    for c in model.__table__.c
                                    if (c.name not in pk_cols) and (c.name in real_cols)
                                }
                                try:
                                    stmt1 = stmt1.on_conflict_do_update(
                                        index_elements=pk_cols,
                                        set_=update_cols_dict,
                                    )
                                except Exception:
                                    stmt1 = sa_insert(model.__table__).values(row)
                            else:
                                stmt1 = sa_insert(model.__table__).values(row)

                        session.execute(stmt1)
                        successes += 1
                        session.commit()
                    except Exception as row_exc:
                        session.rollback()
                        # エラー内容と一部キー情報のみを記録（個人情報/巨大データのダンプは避ける）
                        try:
                            key_preview = {k: row.get(k) for k in list(row.keys())[:4]}
                        except Exception:
                            key_preview = {}
                        errors.append({
                            "row": None,
                            "message": f"DB error: {str(row_exc)}",
                            "preview": key_preview,
                        })

        session.commit()
        # SQLite: インデックス復元
        try:
            if dialect == "sqlite" and dropped_indexes:
                for name, sql in dropped_indexes:
                    try:
                        session.exec(text(sql))
                    except Exception:
                        pass
                session.commit()
        except Exception:
            pass
        t_sql1 = perf_counter()
        try:
            logger.info("_generic_insert[%s]: sql execute+commit done rows=%s chunks=%s batch_max=%s elapsed=%.3fs", getattr(model, "__tablename__", str(model)), len(rows_data), total_chunks, max_rows, (t_sql1 - t_sql0))
        except Exception:
            pass
        # Performance debug output
        print(f"[PERF] _generic_insert SQL execution: {(t_sql1 - t_sql0):.3f}s for {len(rows_data)} rows in {total_chunks} chunks (batch_max={max_rows})")
        print(f"[PERF] _generic_insert TOTAL: {(t_sql1 - t0):.3f}s")

    return {
        "total_rows": total,           # ファイルに存在した行数
        "success_rows": successes,     # 実際に upsert できた行数
        "upserted_rows": successes,
        "error_rows": len(errors),
        "errors": errors,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Helper: Backfill/derive inventory.lot_date from lot column (SAFE)
# ---------------------------------------------------------------------------
def _update_inventory_lot_dates(session: Session) -> int:
    """
    Backfill/derive inventory.lot_date (DATE) from the text 'lot' column.

    Safety:
      * Only call to_date() on strictly valid patterns:
        - letters + YYYYMMDD / letters + YYYYMM
        - anywhere YYYYMMDD / anywhere YYYYMM
      * YYYYMMDD validates both month (01..12) and day (01..31).
      * YYYYMM validates month (01..12) and pads day as 01.
    """
    # Dialect guard: the SQL below is PostgreSQL-specific (to_date, substring FROM regex, UPDATE .. FROM ..).
    # On SQLite (dev), skip silently and return 0 to avoid noisy stack traces.
    try:
        bind = session.get_bind()
        dialect = getattr(getattr(bind, "dialect", None), "name", "") if bind is not None else ""
    except Exception:
        dialect = ""
    if dialect != "postgresql":
        return 0

    sql = text(
        """
        UPDATE inventory i
        SET lot_date = COALESCE(
            CASE WHEN s.l8a IS NOT NULL THEN to_date(s.l8a, 'YYYYMMDD') END,
            CASE WHEN s.l6a IS NOT NULL THEN to_date(s.l6a || '01', 'YYYYMMDD') END,
            CASE WHEN s.l8  IS NOT NULL THEN to_date(s.l8,  'YYYYMMDD') END,
            CASE WHEN s.l6  IS NOT NULL THEN to_date(s.l6  || '01', 'YYYYMMDD') END
        )
        FROM (
            SELECT location_id, sku_id, pack_qty, lot,
                   substring(lot FROM '(?i)[A-Z]+(20[0-9]{2}(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01]))') AS l8a,
                   substring(lot FROM '(?i)[A-Z]+(20[0-9]{2}(0[1-9]|1[0-2]))')                       AS l6a,
                   substring(lot FROM '(20[0-9]{2}(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01]))')        AS l8,
                   substring(lot FROM '(20[0-9]{2}(0[1-9]|1[0-2]))')                                AS l6
            FROM inventory
        ) AS s
        WHERE i.location_id = s.location_id
          AND i.sku_id      = s.sku_id
          AND i.pack_qty    = s.pack_qty
          AND i.lot         = s.lot
          AND i.lot_date IS NULL;
        """
    )
    result = session.exec(sql)
    session.commit()
    try:
        return int(getattr(result, "rowcount", 0) or 0)
    except Exception:
        return 0

# --------------------------------------------------------------------------- #
# util: flexible column lookup                                                #
# --------------------------------------------------------------------------- #

# Cache of normalised header maps keyed by the tuple of original column names
_HEADER_MAP_CACHE: dict[tuple, dict[str, str]] = {}


def _normalize_header(name: str) -> str:  # noqa: D401
    """Return a canonicalised CSV/Excel header.

    Normalisation steps:

    1. Strip a UTF‑8/UTF‑16 BOM (``\ufeff``) if present.
    2. Unicode NFKC folding (converts full‑width ‹ＡＢＣ› → half‑width ‹ABC› etc.).
    3. Remove *all* ASCII/Unicode whitespace characters (spaces, tabs, NBSP, …).
    4. Lower‑case.

    These rules make the header lookup tolerant to the most common quirks
    seen in Japanese ERP exports (mixed full‑width glyphs, stray spaces,
    invisible BOM at cell A1).
    """
    # Step‑1: strip BOM that sometimes sneaks in as the very first character
    if name and name[0] == "\ufeff":
        name = name.lstrip("\ufeff")

    # Step‑2: full‑width/half‑width, composed/decomposed folding
    name = unicodedata.normalize("NFKC", name)

    # Step‑2.5: strip surrounding quotes occasionally emitted by CSV exports
    if name and name[0] in {"'", '"'} and len(name) >= 2 and name[-1] == name[0]:
        name = name[1:-1]
    name = name.strip("'\"")

    # Step‑3: erase *every* whitespace code‑point
    name = re.sub(r"\s+", "", name)

    # Step‑4: lower‑case for case‑insensitive match
    return name.lower()


def _col(row: pd.Series, *candidates: str):  # noqa: D401
    """Return the first matching column value from *candidates*.

    The comparison is **robust** to the following quirks often found in
    Japanese CSV/Excel exports:

    * Full‑width vs. half‑width characters (e.g. "ＩＤ" vs. "ID").
    * Accidental leading/trailing spaces or tabs.
    * In‑string spaces ("ロケーション ") or non‑breaking spaces.
    * ASCII case differences.

    Raises ``KeyError`` if none of the *candidates* are present.
    """

    # Build once per distinct header set; reuse via a small cache.
    idx_key = tuple(row.index)
    normalised_map = _HEADER_MAP_CACHE.get(idx_key)
    if normalised_map is None:
        normalised_map = {_normalize_header(col): col for col in row.index}
        _HEADER_MAP_CACHE[idx_key] = normalised_map

    for cand in candidates:
        # literal match first (fast‑path)
        if cand in row:
            return row[cand]

        # Attempt NFKC/whitespace/ci match
        n_cand = _normalize_header(cand)
        hit = normalised_map.get(n_cand)
        if hit is not None:
            return row[hit]

        # legacy alt: simple full‑width substitution used earlier
        alt = (
            cand.replace("ID", "ＩＤ")
                .replace("SKU", "ＳＫＵ")
                .replace("sku", "sku")  # no‑op but keeps pattern clear
        )
        n_alt = _normalize_header(alt)
        hit = normalised_map.get(n_alt)
        if hit is not None:
            return row[hit]

    # Enhanced error message with actual CSV headers for debugging
    actual_headers = list(row.index)[:10]  # Show first 10 columns
    raise KeyError(
        f"None of {candidates!r} found in CSV header. "
        f"Actual headers (first 10): {actual_headers!r}"
    )

def _opt_col(row: pd.Series, *candidates: str):
    """Best-effort column getter.

    Returns the first matching column value among candidates, or None if none
    of the candidates exist in the row. This is useful for optional columns
    (e.g., dimensions) where absence shouldn't fail the whole row.
    """
    try:
        return _col(row, *candidates)
    except KeyError:
        return None

# --------------------------------------------------------------------------- #
# mappers                                                                     #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# util                                                                        #
# --------------------------------------------------------------------------- #



def _clean_float(val) -> float | None:
    """Convert value to float; handles strings with commas or blanks."""
    if val is None or (isinstance(val, str) and val.strip() == "") or pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(str(val).replace(",", "").strip())
    except Exception as exc:
        raise ValueError(f"Cannot convert {val!r} to float") from exc

def _clean_decimal(val) -> Decimal | None:
    """Convert '0,0239' or '0.0239' etc. to Decimal; empty→None."""
    if val is None or (isinstance(val, str) and val.strip() == "") or pd.isna(val):
        return None
    if isinstance(val, (int, float, Decimal)):
        return Decimal(str(val))
    # assume string
    return Decimal(str(val).replace(",", ".").strip())

# --------------------------------------------------------------------------- #
# util: safe integer conversion                                              #
# --------------------------------------------------------------------------- #

def _safe_int(val, default: int | None = None) -> int:
    """
    Convert value to ``int`` safely.

    * Treats NaN / None / empty‑string as *default*
      (or raises ``ValueError`` if default is None).
    * Accepts numeric strings that may contain:
        - Thousands separators: "1,234"
        - Full‑width digits: "１２３４"
        - Combined: "１,２３４"
    """
    if pd.isna(val):
        val = None

    import math, unicodedata

    # blank / nan → default handling
    if val is None or (isinstance(val, float) and math.isnan(val)) or (isinstance(val, str) and val.strip() == ""):
        if default is not None:
            return default
        raise ValueError("Value required")

    # Handle numeric types directly (int or float)
    if isinstance(val, (int, float)):
        # Convert float to int (truncates decimal part)
        return int(val)

    # Coerce string to int with cleanup
    if isinstance(val, str):
        # Normalize full‑width → half‑width, remove commas/spaces
        cleaned = unicodedata.normalize("NFKC", val).replace(",", "").strip()
        try:
            # Try direct int conversion first
            return int(cleaned)
        except ValueError:
            # If it fails, it might be a float string like "1.0"
            try:
                return int(float(cleaned))
            except Exception as exc:
                raise ValueError(f"Cannot convert {val!r} to int") from exc
    
    # Fallback for other types
    try:
        return int(val)
    except Exception as exc:
        raise ValueError(f"Cannot convert {val!r} to int") from exc


def _safe_date(val):
    """
    CSVから読み込んだ日付を date オブジェクトに変換する。
    
    - YYYYMMDDの整数または浮動小数点数(20240805, 20240805.0)
    - 文字列 ('20240805', '20240805.0', '2024-08-05', '2024/08/05')
    - すでに datetime/date オブジェクト
    
    Examples:
        _safe_date(20240805) -> date(2024, 8, 5)
        _safe_date(20240805.0) -> date(2024, 8, 5)
        _safe_date('20240805') -> date(2024, 8, 5)
        _safe_date('20240805.0') -> date(2024, 8, 5)
        _safe_date('2024-08-05') -> date(2024, 8, 5)
    """
    import pandas as pd
    from datetime import date, datetime
    
    if pd.isna(val):
        raise ValueError(f"Date value is NaT or None: {val!r}")
    
    # すでに date オブジェクト
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    
    # すでに datetime オブジェクト
    if isinstance(val, datetime):
        return val.date()
    
    # 数値(整数または浮動小数点)の場合、YYYYMMDDと仮定して整数化
    if isinstance(val, (int, float)):
        val = int(val)  # 20240805.0 -> 20240805
        val = str(val)  # '20240805'
    
    # 文字列の場合
    if isinstance(val, str):
        val = val.strip()
        
        # '20240805.0' のような浮動小数点文字列を整数化
        try:
            # まず浮動小数点として解釈を試みる
            float_val = float(val)
            int_val = int(float_val)
            val = str(int_val)  # '20240805.0' -> '20240805'
        except (ValueError, TypeError):
            # 数値でない場合は文字列のまま処理
            pass
        
        # pandas の to_datetime で柔軟にパース
        dt = pd.to_datetime(val, errors='coerce')
        if pd.isna(dt):
            raise ValueError(f"Cannot parse date: {val!r}")
        return dt.date()
    
    # その他の型はそのまま pd.to_datetime に渡してみる
    try:
        dt = pd.to_datetime(val, errors='raise')
        return dt.date()
    except Exception as exc:
        raise ValueError(f"Cannot convert {val!r} to date") from exc


# column aliases
# col = lambda *names: next((row.get(n) for n in names if n in row), None)
def _sku_mapper(row: pd.Series) -> Sku:
    sku_code = _col(
        row,
        "item_internalid",  # ← 入荷CSVと同じIDを最優先で採用
        "商品ID",
        "商品ＩＤ",
        "商品コード",  # 追加: よくある別名
        "商品IDコード",
        "SKU",
        "sku",
    )
    # 必須: SKU が空ならスキップ
    if sku_code is None or str(sku_code).strip() == "" or str(sku_code).lower() == "nan":
        raise ValueError("Missing SKU ID")

    return Sku(
        sku_id=_normalize_sku_id(sku_code),
        pack_qty=_safe_int(_col(row, "入数", "入り数", "入数（個）", "pack_qty"), default=1),
        # 寸法系は任意列（存在しなければ None）
        length_mm=_clean_float(_opt_col(row, "商品予備項目００３", "商品予備項目003", "縦", "長さ(mm)", "length_mm")),
        width_mm=_clean_float(_opt_col(row, "商品予備項目００４", "商品予備項目004", "横", "幅(mm)", "width_mm")),
        height_mm=_clean_float(_opt_col(row, "商品予備項目００５", "商品予備項目005", "高", "高さ(mm)", "height_mm")),
        # volume は float で扱う（Decimal→float 変換）。Pydantic v2 の警告対策。
        volume_m3=_clean_float(_opt_col(row, "商品予備項目００６", "商品予備項目006", "容積", "容積(m3)", "volume_m3")),
    )


def _inventory_mapper(row: pd.Series) -> Inventory:
    # 位置は日本語・英語・略称・表示コードなどゆらぎに対応
    loc = _col(
        row,
        "ロケーション", "location", "ロケーションコード", "ロケ", "ロケコード", "display_code", "location_id",
    )
    sku = _col(row, "商品ID", "SKU", "item_internalid")
    sku_norm = _normalize_sku_id(sku)

    # 在庫数量は表記ゆらぎを広く許容
    qty = _safe_int(
        _col(
            row,
            "在庫数",
            "在庫数(引当数を含む)",
            "在庫数（引当数を含む）",
            "在庫数(引当込み)",
            "在庫数（引当込み）",
            "qty",
        )
    )

    # 引当数（オプション列）。無ければ0。
    allocated = _safe_int(_opt_col(row, "引当数", "引当", "allocated_qty", "allocated"), default=0)

    # ブロックは別名も受け付ける（存在しなければ None のまま）
    try:
        blk = _col(row, "ブロック略称", "ブロック名", "block_code", "ブロック", "block")
    except KeyError:
        blk = None

    # ロット列のゆらぎを許容。欠落時は 'UNKNOWN' を採用して取り込み継続。
    lot_val = _opt_col(row, "ロット", "lot", "ロット番号", "LOT", "Lot")
    if lot_val is None or (isinstance(lot_val, str) and lot_val.strip() == ""):
        lot_val = "UNKNOWN"

    # 品質区分名（列が無い場合は None にフォールバック）
    try:
        quality = _col(row, "品質区分", "品質区分名", "quality_name", "品質", "quality")
    except KeyError:
        quality = None

    # ケース数は任意列。存在すればそのまま採用、無ければ後段で補完。
    try:
        cases_in = _clean_float(_col(row, "cases", "ケース数", "ケース"))
    except KeyError:
        cases_in = None

    # Parse location_id into level(3), column(3), depth(2)
    loc_str = str(loc)
    # Keep only digits; tolerate stray spaces or non-digit characters
    loc_digits = "".join(ch for ch in loc_str if ch.isdigit())

    normalized_location = loc_str
    lvl = col = dep = 0

    if len(loc_digits) == 8:
        # 例: "00102821" → level=1, column=28, depth=21
        normalized_location = loc_digits
        lvl = int(loc_digits[0:3])   # 001-999 → 1..999
        col = int(loc_digits[3:6])   # 001-999 → 1..999
        dep = int(loc_digits[6:8])   # 00-99    → 0..99
    elif len(loc_digits) == 6:
        # 例: "101419" → level=1, column=014, depth=19 → 正規化 "00101419"
        lvl_1d = int(loc_digits[0:1])
        col_3d = int(loc_digits[1:4])
        dep_2d = int(loc_digits[4:6])
        normalized_location = f"{lvl_1d:03d}{col_3d:03d}{dep_2d:02d}"
        lvl, col, dep = lvl_1d, col_3d, dep_2d
    else:
        # 非対応フォーマットは元文字列を保持し、分解値は 0 とする
        normalized_location = loc_str
        lvl = col = dep = 0

    return Inventory(
        location_id=normalized_location,
        sku_id=sku_norm,
        pack_qty=0,  # placeholder; filled from SKU master in _generic_insert
        lot=lot_val,
        qty=qty,
        allocated_qty=int(allocated or 0),
        block_code=blk,
        level=lvl,
        column=col,
        depth=dep,
        cases=cases_in,              # _generic_insert で qty/pack から算出
        quality_name=quality,
    )


def _ship_mapper(row: pd.Series) -> ShipTx:
    sku_raw = _col(row, "item_internalid", "SKU", "商品ID")
    sku_norm = _normalize_sku_id(sku_raw)
    return ShipTx(
        sku_id=sku_norm,
        # 出荷CSVには入数が無い前提。SKUマスタから後段で埋める。
        pack_qty=0,
        qty=_safe_int(_col(row, "item_shipquantity", "出荷数")),
        trandate=_safe_date(_col(row, "trandate", "出荷日")),
    )

def _recv_mapper(row: pd.Series) -> RecvTx:
    sku_raw = _col(row, "item_internalid", "SKU", "商品ID")
    sku_norm = _normalize_sku_id(sku_raw)
    return RecvTx(
        sku_id=sku_norm,
        qty=_safe_int(_col(row, "item_quantity", "入庫数")),
        trandate=_safe_date(_col(row, "trandate", "入庫日")),
        lot=row.get("lot") or row.get("ロット"),
    )

# --------------------------------------------------------------------------- #
# Location master helpers (valid/invalid/highness)                            #
# --------------------------------------------------------------------------- #

def _normalize_location_df(df: pd.DataFrame, block: str | None = None) -> pd.DataFrame:
    """Normalise a location CSV into (block_code, level, column, depth, numeric_id).

    * 日本語ヘッダのゆらぎ（全角/半角・スペース違い）や別名に強い
    * 列・連・段が欠けても、`ロケ` から `B-...-...-...` を解析して補完
    * block は列/ロケ/引数の順で推定
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["block_code", "level", "column", "depth", "numeric_id"])  # minimal schema

    # 正規化済みヘッダマップを作成（NFKC, 空白除去, lower）
    norm_map = { _normalize_header(str(c)): str(c) for c in df.columns }

    def pick(*cands: str) -> str | None:
        for c in cands:
            hit = norm_map.get(_normalize_header(c))
            if hit is not None:
                return hit
        return None

    c_block  = pick("ブロック略称","block_code","ブロック","ブロック名称","block")
    c_level  = pick("列（段）","列(段)","列( 段 )","level","列")
    c_column = pick("連（列）","連(列)","連( 列 )","column","連")
    c_depth  = pick("段（連）","段(連)","段( 連 )","depth","段")
    c_disp   = pick("ロケ","ロケーション","ロケコード","display_code","location")

    def to_int(v) -> int | None:
        try:
            if pd.isna(v):
                return None
            s = str(v).strip().replace(",", "")
            return int(float(s))
        except Exception:
            return None

    rows: list[dict] = []
    for _, r in df.iterrows():
        # --- block 推定 ---
        blk = (str(r.get(c_block)) if c_block else None) or block
        if (not blk) and c_disp and isinstance(r.get(c_disp), str):
            s = r.get(c_disp)
            if "-" in s:
                blk = s.split("-")[0].strip()
        blk = (blk or "").strip()

        # --- level/column/depth の抽出 ---
        lv = to_int(r.get(c_level)) if c_level else None
        co = to_int(r.get(c_column)) if c_column else None
        dp = to_int(r.get(c_depth)) if c_depth else None

        # ロケ（例: B-001-019-001 など）からの補完。
        # 代表パターン A: B-<level>-<column>-<depth>
        # 代表パターン B: B-<column>-<depth>-<level>
        if (lv is None or co is None or dp is None) and c_disp and isinstance(r.get(c_disp), str):
            parts = [p for p in str(r.get(c_disp)).split("-") if p != ""]
            # A: B-L-C-D
            try:
                if lv is None and len(parts) >= 2 and parts[1].isdigit():
                    lv = int(parts[1])
                if co is None and len(parts) >= 3 and parts[2].isdigit():
                    co = int(parts[2])
                if dp is None and len(parts) >= 4 and parts[3].isdigit():
                    dp = int(parts[3])
            except Exception:
                pass
            # B: B-C-D-L
            if lv is None or co is None or dp is None:
                try:
                    if co is None and len(parts) >= 2 and parts[1].isdigit():
                        co = int(parts[1])
                    if dp is None and len(parts) >= 3 and parts[2].isdigit():
                        dp = int(parts[2])
                    if lv is None and len(parts) >= 4 and parts[3].isdigit():
                        lv = int(parts[3])
                except Exception:
                    pass

        if lv is None or co is None or dp is None:
            # 必須欠落はスキップ
            continue

        rows.append({
            "block_code": blk,
            "level": int(lv),
            "column": int(co),
            "depth": int(dp),
        })

    out = pd.DataFrame(rows, columns=["block_code","level","column","depth"])
    if out.empty:
        return pd.DataFrame(columns=["block_code", "level", "column", "depth", "numeric_id"])  # no rows

    out["numeric_id"] = (
        out["level"].astype(int).astype(str).str.zfill(3)
        + out["column"].astype(int).astype(str).str.zfill(3)
        + out["depth"].astype(int).astype(str).str.zfill(2)
    )
    return out


# --- CSV reader for location master with encoding fallback ---
def _read_location_upload(file: UploadFile) -> pd.DataFrame:
    """Read a location CSV/Excel. If read_dataframe() yields an empty/unknown schema,
    retry as raw CSV with common Japanese encodings (utf-8, utf-8-sig, cp932)."""
    # First try the shared helper
    try:
        df = read_dataframe(file)
    except Exception:
        df = pd.DataFrame()

    def _looks_like_location(df: pd.DataFrame) -> bool:
        if df is None or df.empty:
            return False
        cols = set(map(str, df.columns))
        needed_any = (
            ("列（段）" in cols or "列(段)" in cols or "level" in cols)
            and ("連（列）" in cols or "連(列)" in cols or "column" in cols)
            and ("段（連）" in cols or "段(連)" in cols or "depth" in cols)
        )
        # If "ロケ" だけでも後で補完できる
        has_disp = ("ロケ" in cols or "ロケーション" in cols or "display_code" in cols or "location" in cols)
        return needed_any or has_disp

    if _looks_like_location(df):
        return df

    # Fallback: brute-force CSV with encodings
    try:
        file.file.seek(0)
        content = file.file.read()
    except Exception:
        content = None
    finally:
        try:
            file.file.seek(0)
        except Exception:
            pass

    if content:
        for enc in (None, "utf-8", "utf-8-sig", "cp932"):
            try:
                bio = io.BytesIO(content)
                df2 = pd.read_csv(bio) if enc is None else pd.read_csv(bio, encoding=enc)
                if _looks_like_location(df2):
                    return df2
            except Exception:
                continue

    # Give up – return whatever we had (likely empty) and let caller raise
    return df if df is not None else pd.DataFrame()


# --- inference helpers for block/quality from CSV or filename ---

def _infer_block_and_quality(df: pd.DataFrame, filename: str | None = None, default_quality: str = "良品") -> tuple[str, str]:
    """Infer (block_code, quality_name) from columns or filename.

    Priority for block_code:
      1) 'block_code' / 'ブロック略称'
      2) 'display_code' like 'B-001-...'
      3) filename like 'Bネス...' or '... B-...'
    Quality: 'quality_name' / '品質区分名' if present; otherwise default.
    """
    blk = None
    # 1) column based
    for cand in ("block_code", "ブロック略称"):
        if cand in df.columns:
            s = df[cand].dropna().astype(str).str.strip()
            s = s[s != ""]
            if not s.empty:
                blk = str(s.mode().iloc[0])  # most frequent
                break
    # 2) display_code pattern
    if blk is None:
        for cand in ("display_code", "ロケコード", "ロケーション", "location"):
            if cand in df.columns:
                s = df[cand].dropna().astype(str)
                # take first match B-...
                import re as _re
                for v in s:
                    m = _re.match(r"^\s*([A-Za-z])[-]", v)
                    if m:
                        blk = m.group(1).upper()
                        break
                if blk:
                    break
    # 3) filename hint
    if blk is None and filename:
        import re as _re
        m = _re.search(r"([A-Za-z])ネス|\b([A-Za-z])[-_]", filename)
        if m:
            blk = (m.group(1) or m.group(2)).upper()
    # fallback
    if not blk:
        blk = "B"

    # quality
    q = None
    for cand in ("quality_name", "品質区分名"):
        if cand in df.columns:
            s = df[cand].dropna().astype(str).str.strip()
            s = s[s != ""]
            if not s.empty:
                q = str(s.mode().iloc[0])
                break
    if not q:
        q = default_quality

    return blk, q


def _rows_from_single(df: pd.DataFrame, *, block: str, quality: str, can_receive: bool | None, highness: bool | None) -> list[dict]:
    """Build upsert rows from a single sheet with fixed flags.

    - numeric_id: LLLCCCDD
    - display_code: {block}-{CCC}-{DD}-{LLL}
    - capacity: 1000x1000x(1500 if highness else 1300) mm
    """
    rows: list[dict] = []
    if df is None or df.empty:
        return rows

    v = _normalize_location_df(df, block)
    for r in v.itertuples(index=False):
        h = bool(highness) if highness is not None else False
        from decimal import Decimal as _D
        cap = (_D(1000) * _D(1000) * _D(1500 if h else 1300)) / _D(1_000_000_000)
        rows.append({
            "block_code": block,
            "quality_name": quality,
            "level": int(r.level),
            "column": int(r.column),
            "depth": int(r.depth),
            "numeric_id": f"{int(r.level):03d}{int(r.column):03d}{int(r.depth):02d}",
            "display_code": f"{block}-{int(r.column):03d}-{int(r.depth):02d}-{int(r.level):03d}",
            # flags
            **({"can_receive": bool(can_receive)} if can_receive is not None else {}),
            **({"highness": h} if highness is not None else {}),
            # physical
            "bay_width_mm": 1000,
            "bay_depth_mm": 1000,
            "bay_height_mm": 1500 if h else 1300,
            "capacity_m3": cap.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP),
        })
    return rows


def _build_loc_union(
    valid_df: pd.DataFrame,
    invalid_df: pd.DataFrame | None,
    high_df: pd.DataFrame | None,
    *,
    block: str,
    quality: str,
    strict_highness: bool = True,
) -> list[dict]:
    """Merge valid/invalid/highness to a single list of row dicts for upsert.

    Conflict policy:
      * All slots = valid ∪ invalid
      * can_receive = True for valid; False for invalid (invalid overrides valid)
      * highness = True for rows present in high_df (overlay)
      * If strict_highness=True, raise on (high \\ (valid ∪ invalid))
    """
    v = _normalize_location_df(valid_df, block)
    i = _normalize_location_df(invalid_df, block) if invalid_df is not None else pd.DataFrame(columns=v.columns)
    h = _normalize_location_df(high_df, block) if high_df is not None else pd.DataFrame(columns=v.columns)

    # Keyed union map
    key = lambda r: (block, quality, int(r.level), int(r.column), int(r.depth))
    union: dict[tuple, dict] = {}

    # valid → can_receive=True
    for r in v.itertuples(index=False):
        union[key(r)] = {
            "block_code": block,
            "quality_name": quality,
            "level": int(r.level),
            "column": int(r.column),
            "depth": int(r.depth),
            "numeric_id": f"{int(r.level):03d}{int(r.column):03d}{int(r.depth):02d}",
            "display_code": f"{block}-{int(r.column):03d}-{int(r.depth):02d}-{int(r.level):03d}",
            "can_receive": True,
            "highness": False,
        }

    # invalid → can_receive=False (override)
    for r in i.itertuples(index=False):
        k = key(r)
        row = union.get(k) or {
            "block_code": block,
            "quality_name": quality,
            "level": int(r.level),
            "column": int(r.column),
            "depth": int(r.depth),
            "numeric_id": f"{int(r.level):03d}{int(r.column):03d}{int(r.depth):02d}",
            "display_code": f"{block}-{int(r.column):03d}-{int(r.depth):02d}-{int(r.level):03d}",
            "can_receive": True,
            "highness": False,
        }
        row["can_receive"] = False
        union[k] = row

    # highness overlay
    outside: list[tuple] = []
    for r in h.itertuples(index=False):
        k = key(r)
        if k not in union:
            outside.append(k)
            if not strict_highness:
                union[k] = {
                    "block_code": block,
                    "quality_name": quality,
                    "level": int(r.level),
                    "column": int(r.column),
                    "depth": int(r.depth),
                    "numeric_id": f"{int(r.level):03d}{int(r.column):03d}{int(r.depth):02d}",
                    "display_code": f"{block}-{int(r.column):03d}-{int(r.depth):02d}-{int(r.level):03d}",
                    "can_receive": True,
                    "highness": True,
                }
                continue
        else:
            union[k]["highness"] = True

    if strict_highness and outside:
        # Return a compact error showing counts and a few samples
        samples = [f"{b}-L{lv}-C{co}-D{dp}" for (b, q, lv, co, dp) in outside[:5]]
        raise HTTPException(
            status_code=400,
            detail=f"Highness rows found outside valid/invalid union: count={len(outside)}, sample={samples}",
        )

    # Compute physical dimensions and capacity for each row
    rows: list[dict] = []
    for r in union.values():
        w, d = 1000, 1000
        h_mm = 1500 if r.get("highness") else 1300
        cap = (Decimal(w) * Decimal(d) * Decimal(h_mm)) / Decimal(1_000_000_000)
        r["bay_width_mm"], r["bay_depth_mm"], r["bay_height_mm"] = w, d, h_mm
        r["capacity_m3"] = cap.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
        rows.append(r)

    return rows


@router.post("/location_master")

def upload_location_master(
    valid_csv: UploadDep,
    invalid_csv: UploadFile | None = File(None),
    high_csv: UploadFile | None = File(None),
    block: str = Query(..., description="ブロック略称（例: B）"),
    quality: str = Query("良品", description="品質区分名（例: 良品）"),
    mode: Literal["replace", "patch"] = Query("replace"),
    strict_highness: bool = Query(True, description="ハイネスが有効/無効の集合外にあればエラーにする"),
    *,
    ses: SesDep,
):
    """Import location master from three CSVs (valid/invalid/highness).

    * All slots = valid ∪ invalid
    * can_receive=True for valid; False for invalid
    * highness overlay from high_csv
    * Capacity is computed as 1000x1000x(1500 if highness else 1300) mm

    Endpoint path: **/v1/upload/location_master** (this router has prefix /v1/upload)
    Form fields: valid_csv (required), invalid_csv (optional), high_csv (optional)
    Query params: block, quality, mode, strict_highness
    """
    try:
        vdf = read_dataframe(valid_csv)
        idf = read_dataframe(invalid_csv) if invalid_csv is not None else pd.DataFrame()
        hdf = read_dataframe(high_csv) if high_csv is not None else pd.DataFrame()

        rows = _build_loc_union(vdf, idf, hdf, block=block, quality=quality, strict_highness=bool(strict_highness))

        # Replace or patch mode
        if mode == "replace":
            ses.exec(text("DELETE FROM location_master WHERE block_code = :b AND quality_name = :q").bindparams(b=block, q=quality))
            ses.commit()

        from sqlalchemy.dialects.postgresql import insert as _pg_insert
        tbl = LocationMaster.__table__

        if rows:
            _upsert_location_master(
                ses,
                rows,
                set_dict={
                    "numeric_id": text("excluded.numeric_id"),
                    "display_code": text("excluded.display_code"),
                    "can_receive": text("excluded.can_receive"),
                    "highness": text("excluded.highness"),
                    "bay_width_mm": text("excluded.bay_width_mm"),
                    "bay_depth_mm": text("excluded.bay_depth_mm"),
                    "bay_height_mm": text("excluded.bay_height_mm"),
                    "capacity_m3": text("excluded.capacity_m3"),
                    "effective_from": text("excluded.effective_from"),
                    "disabled_from": text("excluded.disabled_from"),
                    "meta_json": text("excluded.meta_json"),
                },
            )

        # Stats
        total = ses.exec(text("SELECT COUNT(*) FROM location_master WHERE block_code=:b AND quality_name=:q").bindparams(b=block, q=quality)).one()[0]
        can_recv = ses.exec(text("SELECT COUNT(*) FROM location_master WHERE block_code=:b AND quality_name=:q AND can_receive").bindparams(b=block, q=quality)).one()[0]
        high_cnt = ses.exec(text("SELECT COUNT(*) FROM location_master WHERE block_code=:b AND quality_name=:q AND highness").bindparams(b=block, q=quality)).one()[0]

        return {
            "block": block,
            "quality": quality,
            "mode": mode,
            "total_slots": int(total),
            "can_receive": int(can_recv),
            "cannot_receive": int(total) - int(can_recv),
            "highness": int(high_cnt),
            "height_policy_mm": {"highness": 1500, "normal": 1300},
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("location_master upload failed")
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------------------------------------------------------- #
# endpoints                                                                   #
# --------------------------------------------------------------------------- #


# --- Single-file location master endpoints (valid/invalid/highness) ---


@router.post("/location_master/valid")
def upload_location_master_valid(
    file: UploadDep,
    *,
    ses: SesDep,
):
    """Upload **valid** locations only. Flags: can_receive=True.
    Block / quality are inferred from the CSV/filename.
    """
    try:
        df = _read_location_upload(file)
        block, quality = _infer_block_and_quality(df, getattr(file, "filename", None))
        rows = _rows_from_single(df, block=block, quality=quality, can_receive=True, highness=False)
        from sqlalchemy.dialects.postgresql import insert as _pg_insert
        tbl = LocationMaster.__table__
        if rows:
            _upsert_location_master(
                ses,
                rows,
                set_dict={
                    "numeric_id": text("excluded.numeric_id"),
                    "display_code": text("excluded.display_code"),
                    "can_receive": text("excluded.can_receive"),
                    # keep existing highness
                    "bay_width_mm": text("excluded.bay_width_mm"),
                    "bay_depth_mm": text("excluded.bay_depth_mm"),
                    "bay_height_mm": text("excluded.bay_height_mm"),
                    "capacity_m3": text("excluded.capacity_m3"),
                },
            )
        total = ses.exec(text("SELECT COUNT(*) FROM location_master WHERE block_code=:b AND quality_name=:q").bindparams(b=block, q=quality)).one()[0]
        can_recv = ses.exec(text("SELECT COUNT(*) FROM location_master WHERE block_code=:b AND quality_name=:q AND can_receive").bindparams(b=block, q=quality)).one()[0]
        high_cnt = ses.exec(text("SELECT COUNT(*) FROM location_master WHERE block_code=:b AND quality_name=:q AND highness").bindparams(b=block, q=quality)).one()[0]
        return {"block": block, "quality": quality, "mode": "patch", "total_slots": int(total), "can_receive": int(can_recv), "cannot_receive": int(total) - int(can_recv), "highness": int(high_cnt)}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("location_master/valid upload failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/location_master/invalid")
def upload_location_master_invalid(
    file: UploadDep,
    *,
    ses: SesDep,
):
    """Upload **invalid** locations only. Flags: can_receive=False.
    Block / quality are inferred from the CSV/filename.
    """
    try:
        df = _read_location_upload(file)
        block, quality = _infer_block_and_quality(df, getattr(file, "filename", None))
        rows = _rows_from_single(df, block=block, quality=quality, can_receive=False, highness=False)
        from sqlalchemy.dialects.postgresql import insert as _pg_insert
        tbl = LocationMaster.__table__
        if rows:
            _upsert_location_master(
                ses,
                rows,
                set_dict={
                    "numeric_id": text("excluded.numeric_id"),
                    "display_code": text("excluded.display_code"),
                    "can_receive": text("excluded.can_receive"),
                    # do not touch highness
                    "bay_width_mm": text("excluded.bay_width_mm"),
                    "bay_depth_mm": text("excluded.bay_depth_mm"),
                    "bay_height_mm": text("excluded.bay_height_mm"),
                    "capacity_m3": text("excluded.capacity_m3"),
                },
            )
        total = ses.exec(text("SELECT COUNT(*) FROM location_master WHERE block_code=:b AND quality_name=:q").bindparams(b=block, q=quality)).one()[0]
        can_recv = ses.exec(text("SELECT COUNT(*) FROM location_master WHERE block_code=:b AND quality_name=:q AND can_receive").bindparams(b=block, q=quality)).one()[0]
        high_cnt = ses.exec(text("SELECT COUNT(*) FROM location_master WHERE block_code=:b AND quality_name=:q AND highness").bindparams(b=block, q=quality)).one()[0]
        return {"block": block, "quality": quality, "mode": "patch", "total_slots": int(total), "can_receive": int(can_recv), "cannot_receive": int(total) - int(can_recv), "highness": int(high_cnt)}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("location_master/invalid upload failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/location_master/highness")
def upload_location_master_highness(
    file: UploadDep,
    *,
    ses: SesDep,
):
    """Upload **highness** overlay. Flags: highness=True.
    If a slot does not exist yet, it will be created with can_receive=True by default.
    Block / quality are inferred from the CSV/filename.
    """
    try:
        df = _read_location_upload(file)
        block, quality = _infer_block_and_quality(df, getattr(file, "filename", None))
        rows = _rows_from_single(df, block=block, quality=quality, can_receive=True, highness=True)
        from sqlalchemy.dialects.postgresql import insert as _pg_insert
        tbl = LocationMaster.__table__
        if rows:
            _upsert_location_master(
                ses,
                rows,
                set_dict={
                    "numeric_id": text("excluded.numeric_id"),
                    "display_code": text("excluded.display_code"),
                    "highness": text("excluded.highness"),
                    # do not touch can_receive here
                    "bay_width_mm": text("excluded.bay_width_mm"),
                    "bay_depth_mm": text("excluded.bay_depth_mm"),
                    "bay_height_mm": text("excluded.bay_height_mm"),
                    "capacity_m3": text("excluded.capacity_m3"),
                },
            )
        total = ses.exec(text("SELECT COUNT(*) FROM location_master WHERE block_code=:b AND quality_name=:q").bindparams(b=block, q=quality)).one()[0]
        can_recv = ses.exec(text("SELECT COUNT(*) FROM location_master WHERE block_code=:b AND quality_name=:q AND can_receive").bindparams(b=block, q=quality)).one()[0]
        high_cnt = ses.exec(text("SELECT COUNT(*) FROM location_master WHERE block_code=:b AND quality_name=:q AND highness").bindparams(b=block, q=quality)).one()[0]
        return {"block": block, "quality": quality, "mode": "patch", "total_slots": int(total), "can_receive": int(can_recv), "cannot_receive": int(total) - int(can_recv), "highness": int(high_cnt)}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("location_master/highness upload failed")
        raise HTTPException(status_code=500, detail=str(e))


UploadDep = Annotated[UploadFile, File(...)]
SesDep = Annotated[Session, Depends(get_session)]


# --------------------------------------------------------------------------- #
# DB helper: dialect-safe table truncation/cleanup                             #
# --------------------------------------------------------------------------- #
def _truncate_tables(
    session: Session,
    table_names: list[str],
    *,
    restart_identity: bool = False,
    cascade: bool = False,
) -> None:
    """
    Clear rows from given tables in a dialect-safe way.

    - PostgreSQL: TRUNCATE TABLE ... [RESTART IDENTITY] [CASCADE]
    - SQLite/other: DELETE FROM table; and optionally reset sqlite_sequence.

    When `cascade=True` and `sku` is targeted on non-Postgres, also clears
    dependent tables (`ship_tx`, `recv_tx`) before removing `sku` to mimic
    CASCADE semantics.
    """
    bind = session.get_bind()
    dialect = getattr(bind.dialect, "name", "") if bind is not None else ""

    # Expand minimal dependency knowledge for non-cascade engines
    expanded: list[str] = []
    deps: dict[str, list[str]] = {
        "sku": ["ship_tx", "recv_tx"],
    }
    seen: set[str] = set()
    for t in table_names:
        if cascade and dialect != "postgresql" and t in deps:
            for dep in deps[t]:
                if dep not in seen:
                    expanded.append(dep)
                    seen.add(dep)
        if t not in seen:
            expanded.append(t)
            seen.add(t)

    targets = expanded

    if dialect == "postgresql":
        # Build TRUNCATE statement with options
        opts: list[str] = []
        if restart_identity:
            opts.append("RESTART IDENTITY")
        if cascade:
            opts.append("CASCADE")
        opt_sql = (" " + " ".join(opts)) if opts else ""
        tbl_sql = ", ".join(targets)
        session.exec(text(f"TRUNCATE TABLE {tbl_sql}{opt_sql}"))
        session.commit()
        return

    # Fallback: DELETE rows per table (pre-check existence to avoid noisy tracebacks)
    try:
        insp = inspect(session.get_bind())
        existing = set(insp.get_table_names())
    except Exception:
        existing = set()

    for name in targets:
        if existing and (name not in existing):
            # Table does not exist – skip silently (dev convenience)
            continue
        try:
            session.exec(text(f"DELETE FROM {name}"))
        except Exception as e:
            # Dev 環境でテーブル未作成の場合でも落ちないようにする（後続の create_all で補う）
            if "no such table" in str(e).lower():
                continue
            raise
    # Reset SQLite autoincrement sequence if requested
    if restart_identity and dialect == "sqlite":
        for name in targets:
            try:
                session.exec(text("DELETE FROM sqlite_sequence WHERE name = :n").bindparams(n=name))
            except Exception:
                # sqlite_sequence may not exist (older SQLite or no AUTOINCREMENT)
                pass
    session.commit()


@router.post("/sku")
async def upload_sku(file: UploadDep, ses: SesDep, request: Request):
    """Upload SKU master (置き換えモード・常時)。

    アップロードのたびに `sku` テーブルを **TRUNCATE RESTART IDENTITY CASCADE** し、
    データを丸ごと置き換えます（参照する在庫/入出荷も CASCADE で消去）。
    その後、ファイル内容を upsert 方針で投入します（初回は実質 INSERT）。
    """
    try:
        from time import perf_counter
        t0 = perf_counter()
        
        # Get base URL from request
        base_url = str(request.base_url).rstrip('/')
        
        try:
            logger.info(
                "upload_sku: start filename=%s content_type=%s",
                getattr(file, "filename", None),
                getattr(file, "content_type", None),
            )
        except Exception:
            pass
        # 1) ファイルを先に検証・読込（失敗時は既存データを壊さない）
        df = read_dataframe(file)
        t1 = perf_counter()
        try:
            logger.info("upload_sku: read_dataframe done rows=%s elapsed=%.3fs", len(df), (t1 - t0))
        except Exception:
            pass
        print(f"[PERF] upload_sku read_dataframe: {(t1 - t0):.3f}s for {len(df)} rows")
        # 2) 問題なければ置き換え
        _truncate_tables(ses, ["sku"], restart_identity=True, cascade=True)
        t2 = perf_counter()
        try:
            logger.info("upload_sku: truncate done elapsed=%.3fs", (t2 - t1))
        except Exception:
            pass
        print(f"[PERF] upload_sku truncate: {(t2 - t1):.3f}s")
        # 3) 挿入
        summary = _generic_insert(df, Sku, _sku_mapper, ses, use_upsert=False)
        t3 = perf_counter()
        try:
            logger.info("upload_sku: insert done elapsed=%.3fs total_elapsed=%.3fs", (t3 - t2), (t3 - t0))
        except Exception:
            pass
        print(f"[PERF] upload_sku insert: {(t3 - t2):.3f}s, TOTAL: {(t3 - t0):.3f}s")
        try:
            logger.info("upload_sku: done total=%s success=%s errors=%s", summary.get("total_rows"), summary.get("success_rows"), summary.get("error_rows"))
        except Exception:
            pass
        # attach error CSV url if any
        if summary["error_rows"] > 0:
            summary["error_csv_url"] = _save_error_csv(summary["errors"], base_url)
            # Include first few errors in response for debugging
            summary["sample_errors"] = summary["errors"][:5] if len(summary["errors"]) > 0 else []
        else:
            summary["error_csv_url"] = None
            summary["sample_errors"] = []
        # Add CSV header info for debugging
        summary["csv_headers"] = list(df.columns)
        summary["csv_sample_row"] = df.head(1).to_dict(orient="records")[0] if len(df) > 0 else {}
        return summary
    except ValueError as e:
        logger.exception("sku upload failed: invalid file")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("sku upload failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inventory")
async def upload_inventory(file: UploadDep, ses: SesDep, request: Request):
    try:
        base_url = str(request.base_url).rstrip('/')
        # 1) ファイルを先に検証・読込
        df = read_dataframe(file)
        # 2) 置き換え
        _truncate_tables(ses, ["inventory"], restart_identity=True)
        # 3) 挿入
        summary = _generic_insert(df, Inventory, _inventory_mapper, ses, use_upsert=True)
        # Backfill/derive lot_date column from lot text（SQLiteでは失敗しても無害に握りつぶす）
        try:
            updated_cnt = _update_inventory_lot_dates(ses)
            summary["lot_date_updated_rows"] = int(updated_cnt)
        except Exception:
            logger.exception("inventory lot_date backfill failed")
            summary["lot_date_updated_rows"] = 0
        # attach error CSV url if any
        if summary["error_rows"] > 0:
            summary["error_csv_url"] = _save_error_csv(summary["errors"], base_url)
        else:
            summary["error_csv_url"] = None
        return summary
    except ValueError as e:
        logger.exception("inventory upload failed: invalid file")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("inventory upload failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ship_tx")
async def upload_ship_tx(file: UploadDep, ses: SesDep, request: Request):
    try:
        base_url = str(request.base_url).rstrip('/')
        # 1) ファイル検証
        df = read_dataframe(file)
        # 2) 置き換え
        _truncate_tables(ses, ["ship_tx"], restart_identity=True)
        # 3) 挿入
        summary = _generic_insert(df, ShipTx, _ship_mapper, ses, use_upsert=False)
        # attach error CSV url if any
        if summary["error_rows"] > 0:
            summary["error_csv_url"] = _save_error_csv(summary["errors"], base_url)
        else:
            summary["error_csv_url"] = None
        return summary
    except ValueError as e:
        logger.exception("ship_tx upload failed: invalid file")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("ship_tx upload failed")
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/recv_tx")
async def upload_recv_tx(file: UploadDep, ses: SesDep, request: Request):
    try:
        base_url = str(request.base_url).rstrip('/')
        # 1) ファイル検証
        df = read_dataframe(file)
        # 2) 置き換え
        _truncate_tables(ses, ["recv_tx"], restart_identity=True)
        # 3) 挿入
        summary = _generic_insert(df, RecvTx, _recv_mapper, ses, use_upsert=False)
        # attach error CSV url if any
        if summary["error_rows"] > 0:
            summary["error_csv_url"] = _save_error_csv(summary["errors"], base_url)
        else:
            summary["error_csv_url"] = None
        return summary
    except ValueError as e:
        logger.exception("recv_tx upload failed: invalid file")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("recv_tx upload failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Additional inventory endpoints
# ---------------------------------------------------------------------------

@router.post("/inventory/rebuild_lot_dates")
def inventory_rebuild_lot_dates(session: Session = Depends(get_session)):
    """
    Recompute inventory.lot_date for all rows based on text 'lot'.
    Safe to run multiple times.
    """
    try:
        updated = _update_inventory_lot_dates(session)
        return {"updated_rows": int(updated)}
    except Exception as e:
        logger.exception("rebuild lot_date failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/inventory/preview")
def inventory_preview(
    session: Session = Depends(get_session),
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    block_codes: list[str] | None = Query(None),
    quality_names: list[str] | None = Query(None),
):
    """
    Lightweight inventory browser for the frontend.
    Returns both `lot` and parsed `lot_date`.
    """
    try:
        stmt = _sa_select(Inventory)
        if block_codes:
            stmt = stmt.where(Inventory.block_code.in_(block_codes))
        if quality_names:
            stmt = stmt.where(Inventory.quality_name.in_(quality_names))
        stmt = stmt.offset(offset).limit(limit)

        items: list[dict] = []
        rows = session.exec(stmt).all()
        for inv in rows:
            # inv is an Inventory model instance
            ld = getattr(inv, "lot_date", None)
            items.append({
                "sku_id": inv.sku_id,
                "lot": inv.lot,
                "lot_date": (ld.isoformat() if ld else None),
                "location_id": inv.location_id,
                "pack_qty": inv.pack_qty,
                "qty": inv.qty,
                "allocated_qty": getattr(inv, "allocated_qty", 0),
                "column": inv.column,
                "cases": inv.cases,
                "block_code": inv.block_code,
                "level": inv.level,
                "depth": inv.depth,
                "quality_name": inv.quality_name,
            })
        return {
            "count": len(items),
            "offset": int(offset),
            "limit": int(limit),
            "items": items,
        }
    except Exception as e:
        logger.exception("inventory preview failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Analysis (sync) – block‑scoped SKU metrics recompute for easier testing
# NOTE: MVP用途のため upload.py に同居させています。将来的には analysis.py へ分離予定。
# ---------------------------------------------------------------------------
class AnalysisStartRequest(BaseModel):
    # Primary param (backend native)
    window_days: int | None = None
    # Frontend alias (analyze.tsx uses this)
    rotation_window_days: int | None = None
    # Optional filters (currently blockのみ対応。qualityは将来拡張のため受理だけして無視)
    block_codes: list[str] | None = None
    quality_names: list[str] | None = None

@router.post("/analysis/start")
def analysis_start(req: AnalysisStartRequest, session: Session = Depends(get_session)):
    try:
        # Accept both window_days and rotation_window_days
        wd_in = req.window_days if req.window_days is not None else req.rotation_window_days
        logger.info(f"analysis_start: window_days={req.window_days}, rotation_window_days={req.rotation_window_days}, wd_in={wd_in}")
        wd = int(wd_in or 0)
        if wd <= 0:
            wd = 99_999  # 全期間扱い
        logger.info(f"analysis_start: effective window_days={wd}")
        updated = recompute_all_sku_metrics(
            session,
            turnover_window_days=wd,
            block_filter=req.block_codes,
        )
        return {
            "updated": int(updated),
            "window_days_requested": int(wd_in or 0),
            "window_days_effective": int(wd),
            "blocks": list(req.block_codes or []),
            # quality_names は現状のメトリクス計算では未使用（将来拡張用）
            "quality_names": list(req.quality_names or []),
        }
    except Exception as e:
        logger.exception("analysis_start failed")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")



#
# ---------------------------------------------------------------------------
# Compat: robustly call optimizer.plan_relocation regardless of parameter names
# ---------------------------------------------------------------------------

def _call_plan_relocation_compat(*, cfg, sku_df, inv_df, ship_df, recv_df, location_master_df):
    """Call planner with kwargs filtered/mapped to its signature.

    Supports older/newer variants where parameters may be named
    (ship|ship_df|ship_tx), (recv|recv_df|recv_tx), etc. If keyword mapping
    fails or the call raises, gracefully fall back to the current known
    signature without raising from this shim.
    """
    fn = plan_relocation
    # Try kwargs based on signature
    try:
        sig = pyinspect.signature(fn)
        params = sig.parameters
        kw: dict = {}

        # cfg/config
        if "cfg" in params:
            kw["cfg"] = cfg
        elif "config" in params:
            kw["config"] = cfg

        def _bind(preferred, value, aliases):
            for nm in [preferred] + list(aliases):
                if nm in params:
                    kw[nm] = value
                    return True
            return False

        _bind("sku_master", sku_df, ["sku_df", "sku", "sku_master_df"])  # sku master
        _bind("inventory",  inv_df, ["inv", "inventory_df", "inv_df"])   # inventory
        _bind("ship",       ship_df, ["ship_df", "ship_tx", "ship_tx_df", "shipments", "shipments_df"])  # ship
        _bind("recv",       recv_df, ["recv_df", "recv_tx", "recv_tx_df", "receipts", "receipts_df", "receiving", "receiving_df"])  # recv
        _bind("location_master", location_master_df, ["location_master_df", "loc_master", "locations", "locations_df", "capacity", "capacity_df", "location_capacity"])  # capacity

        has_varkw = any(p.kind == pyinspect.Parameter.VAR_KEYWORD for p in params.values())
        if kw or has_varkw:
            try:
                return fn(**kw)
            except Exception:
                # fall through to direct call
                pass
    except Exception:
        # signature introspection failed; fall through to direct call
        pass

    # Final fallback: call with the current optimizer signature (keyword-only)
    try:
        return fn(
            sku_master=sku_df,
            inventory=inv_df,
            cfg=cfg,
            ai_col_hints={},
            loc_master=location_master_df,
        )
    except Exception:
        # As a last resort, return an empty plan rather than crashing the server
        try:
            # Prefer returning [] to keep API responsive even if planner wiring changes
            return []
        except Exception:
            return []

# ---------------------------------------------------------------------------
# Relocation (sync) – call optimizer.plan_relocation and return moves as JSON
# フロントの簡易検証用エンドポイント（MVP）。
# ※ 本番では Celery ジョブ + CSV/Excel エクスポートに置き換え予定。
# ---------------------------------------------------------------------------
class RelocationStartRequest(BaseModel):
    block_codes: list[str] | None = None
    quality_names: list[str] | None = None  # 追加: 品質区分で対象を限定
    max_moves: int = 200
    # 容量上限は棚容量の90%（要件に合わせてデフォルトを0.90へ）
    fill_rate: float = 0.90
    # --- AI 配線オプション ---
    use_ai: bool = True                    # AIヒントを使うか
    ai_max_candidates: int = 3             # SKUごとの優先「列」候補の上限
    # --- 対象SKUの制限 ---
    require_volume: bool = True            # 商品予備項目006(volume_m3)があるSKUのみ
    # --- AIメイン切替 & 回転期間 ---
    use_ai_main: bool = False
    rotation_window_days: int = 90
    # --- 連鎖退避の有効化と予算（optimizer.py で実装済みの新機能） ---
    chain_depth: int = 2           # 0=無効, 1..N=最大連鎖段数（デフォルト2）
    eviction_budget: int = 50      # 退避移動の総数上限（デフォルト有効）
    touch_budget: int = 100        # 連鎖で触ってよいユニークロケ数の上限（デフォルト有効）
    # --- 高コストパスのトグル（高速化に有効） ---
    enable_pass0_area_rebalance: bool | None = None  # Trueで帯リバランスを試行
    enable_pass1_swap: bool | None = None            # Trueでスワップ最適化を試行
    pass1_swap_budget_per_group: int | None = None   # スワップ上限（同一SKU×同一列グループ）
    # --- 帯嗜好パラメータ（optimizer.py の OptimizerConfig と連動） ---
    pack_low_max: int | None = None
    pack_high_min: int | None = None
    near_cols: list[int] | None = None
    far_cols: list[int] | None = None
    band_pref_weight: float | None = None
    # --- 奥行きの優先度（front=浅い順, center=山形） ---
    depth_preference: str | None = None  # 'front' or 'center'
    center_depth_weight: float | None = None
    promo_quality_keywords: list[str] | None = None  # 例: ["販促資材","販促","什器","資材"]
    # --- trace / debug ---
    trace_id: str | None = None          # Frontend-provided trace id (optional)
    include_debug: bool = False          # If true, include a compact rejection summary in the response
# Endpoint for relocation
@router.post("/relocation/start")
def relocation_start(
    req: RelocationStartRequest,
    session: Session = Depends(get_session),
):
    # Build optimizer config defensively: avoid passing unsupported kwargs
    try:
        cfg = OptimizerConfig()  # prefer no-arg init; attach fields via setattr
    except TypeError:
        # If OptimizerConfig requires args or isn't importable as expected,
        # fall back to a dynamic namespace with attribute access.
        from types import SimpleNamespace
        cfg = SimpleNamespace()

    # Apply user-provided knobs uniformly; ignore if OptimizerConfig doesn't have them
    for _k in (
        "max_moves",
        "fill_rate",
        "use_ai",
        "ai_max_candidates",
        "require_volume",
        "use_ai_main",
        "rotation_window_days",
        "chain_depth",
        "eviction_budget",
        "touch_budget",
        # --- high-cost pass toggles ---
        "enable_pass0_area_rebalance",
        "enable_pass1_swap",
        "pass1_swap_budget_per_group",
        # --- band preference knobs passed through 1:1 ---
        "pack_low_max",
        "pack_high_min",
        "band_pref_weight",
        # --- depth preference knobs ---
        "depth_preference",
        "center_depth_weight",
    ):
        try:
            _v = getattr(req, _k, None)
            if _v is not None:
                setattr(cfg, _k, _v)
        except Exception:
            pass

    # Attach scoping fields even if OptimizerConfig doesn't declare them
    setattr(cfg, "block_codes", req.block_codes)
    setattr(cfg, "quality_names", req.quality_names)
    # Column band / keyword preferences that need type adaptation
    try:
        if getattr(req, "near_cols", None):
            setattr(cfg, "near_cols", tuple(int(x) for x in req.near_cols))
        if getattr(req, "far_cols", None):
            setattr(cfg, "far_cols", tuple(int(x) for x in req.far_cols))
        if getattr(req, "promo_quality_keywords", None):
            setattr(cfg, "promo_quality_keywords", tuple(str(x) for x in req.promo_quality_keywords))
    except Exception:
        pass
    # --- Bind trace id so drop logs match the frontend id ---
    # Use the client-provided id when available; otherwise generate a short hex.
    try:
        req_trace = (req.trace_id or "").strip() if hasattr(req, "trace_id") else ""
        bound_trace_id = req_trace if req_trace else uuid4().hex[:12]
        # Attach to config even if OptimizerConfig doesn't declare the field
        setattr(cfg, "trace_id", bound_trace_id)
    except Exception:
        bound_trace_id = None

    # -- Build source DataFrames from DB --
    sku_df  = _df_from_sku(session)
    inv_df  = _df_from_inventory(session)
    ship_df = _df_from_ship(session)
    recv_df = _df_from_recv(session)

    # Optional: location_master DataFrame (capacity & can_receive flags)
    try:
        lm_rows = session.exec(_sa_select(
            LocationMaster.block_code,
            LocationMaster.quality_name,
            LocationMaster.level,
            LocationMaster.column,
            LocationMaster.depth,
            LocationMaster.numeric_id,
            LocationMaster.display_code,
            LocationMaster.can_receive,
            LocationMaster.highness,
            LocationMaster.capacity_m3,
        )).all()
        location_master_df = pd.DataFrame(lm_rows, columns=[
            "block_code","quality_name","level","column","depth",
            "numeric_id","display_code","can_receive","highness","capacity_m3"
        ])
    except Exception:
        location_master_df = pd.DataFrame(columns=[
            "block_code","quality_name","level","column","depth",
            "numeric_id","display_code","can_receive","highness","capacity_m3"
        ])

    # --- Strict pre-filtering by block/quality & can_receive (destination) ---
    try:
        _blocks = {str(b).strip() for b in (req.block_codes or []) if str(b).strip()}
        _quals  = {str(q).strip() for q in (req.quality_names or []) if str(q).strip()}
    except Exception:
        _blocks, _quals = set(), set()

    # Inventory: limit to selected blocks/qualities only (source population)
    if not inv_df.empty:
        if _blocks and "ブロック略称" in inv_df.columns:
            inv_df = inv_df[inv_df["ブロック略称"].astype(str).isin(_blocks)]
        if _quals:
            qcol = "品質区分名" if "品質区分名" in inv_df.columns else ("quality_name" if "quality_name" in inv_df.columns else None)
            if qcol:
                inv_df = inv_df[inv_df[qcol].astype(str).isin(_quals)]

    # Location master: destinations must be registered AND receivable
    if not location_master_df.empty:
        # 1) only receivable slots
        if "can_receive" in location_master_df.columns:
            location_master_df = location_master_df[location_master_df["can_receive"] == True]
        # 2) restrict to requested blocks / qualities
        if _blocks and "block_code" in location_master_df.columns:
            location_master_df = location_master_df[location_master_df["block_code"].astype(str).isin(_blocks)]
        if _quals and "quality_name" in location_master_df.columns:
            location_master_df = location_master_df[location_master_df["quality_name"].astype(str).isin(_quals)]

    # Optional: small log for sanity
    try:
        logger.info(
            "relocation_start filters: blocks=%s qualities=%s | inv=%s rows -> %s, loc_master=%s rows",
            sorted(list(_blocks)) if _blocks else [],
            sorted(list(_quals)) if _quals else [],
            len(_df_from_inventory(session)) if 'session' in locals() else 'n/a',
            len(inv_df) if 'inv_df' in locals() else 'n/a',
            len(location_master_df) if 'location_master_df' in locals() else 'n/a',
        )
    except Exception:
        pass

    # Call optimizer with signature-agnostic shim; if it fails, fall back to
    # a direct call using the current optimizer signature.
    try:
        moves = _call_plan_relocation_compat(
            cfg=cfg,
            sku_df=sku_df,
            inv_df=inv_df,
            ship_df=ship_df,
            recv_df=recv_df,
            location_master_df=location_master_df,
        )
    except TypeError as _compat_err:
        try:
            logger.warning("relocation_start: compat shim failed (%s); using direct call", str(_compat_err))
        except Exception:
            pass
        # Direct call aligned to optimizer.plan_relocation signature
        try:
            moves = plan_relocation(
                sku_master=sku_df,
                inventory=inv_df,
                cfg=cfg,
                block_filter=req.block_codes,
                quality_filter=req.quality_names,
                ai_col_hints={},
                loc_master=location_master_df,
            )
        except Exception as e:
            logger.exception("relocation_start: direct plan call failed")
            raise HTTPException(status_code=500, detail=str(e))
    # Determine the *actual* trace id used by optimizer (in case it overrode/auto-generated)
    try:
        used_trace_id = get_current_trace_id() or getattr(cfg, "trace_id", None)
    except Exception:
        used_trace_id = getattr(cfg, "trace_id", None)

    # Always include a compact rejection summary in the response
    try:
        _rej = get_last_rejection_debug() or {}
        rej = {
            "planned": _rej.get("planned"),
            "accepted": _rej.get("accepted"),
            "rejections": _rej.get("rejections", {}),
            "examples": _rej.get("examples", {}),
        }
    except Exception:
        rej = {}

    # --- Build a compact efficiency summary for UI ---
    def _parse_loc8(loc: str):
        s = str(loc or "")
        if len(s) == 8 and s.isdigit():
            try:
                return int(s[0:3]), int(s[3:6]), int(s[6:8])
            except Exception:
                return 0, 0, 0
        return 0, 0, 0

    def _ease_key(lv: int, col: int, dep: int) -> int:
        # mirror of services.optimizer._ease_key
        try:
            col_rev = 42 - int(col)
            return int(lv) * 10000 + int(col_rev) * 100 + int(dep)
        except Exception:
            return 99_999_999

    def _build_summary(inv_df: pd.DataFrame, moves_data: list[dict]) -> dict:
        if not isinstance(inv_df, pd.DataFrame) or not isinstance(moves_data, list) or not moves_data:
            return {"moves": 0}
        # Map (sku, lot, from_loc) -> before coords
        before = {}
        if not inv_df.empty:
            for _, r in inv_df.iterrows():
                try:
                    key = (str(r.get("商品ID")), str(r.get("ロット")), str(r.get("ロケーション")))
                    lv, col, dep = _parse_loc8(str(r.get("ロケーション")))
                    before[key] = (lv, col, dep)
                except Exception:
                    pass
        total = len(moves_data)
        touched = set()
        uniq_skus = set()
        qty_total = 0
        lvl_delta_sum = 0
        col_dist_sum = 0
        dep_dist_sum = 0
        ease_gain_sum = 0
        to_pick = 0
        examples = []
        for m in moves_data:
            sku = str(m.get("sku_id"))
            lot = str(m.get("lot") or m.get("lot_date_key") or "")
            fr = str(m.get("from_loc") or m.get("from") or "")
            to = str(m.get("to_loc") or m.get("to") or "")
            qty = int(m.get("qty") or m.get("qty_cases") or m.get("ケース") or 0)
            uniq_skus.add(sku)
            touched.add(fr); touched.add(to)
            qty_total += max(0, qty)
            flv,fcol,fdep = _parse_loc8(fr)
            tlv,tcol,tdep = _parse_loc8(to)
            if tlv in (1,2):
                to_pick += 1
            lvl_delta_sum += (flv - tlv)
            col_dist_sum += abs(tcol - fcol)
            dep_dist_sum += abs(tdep - fdep)
            ease_gain_sum += max(0, _ease_key(flv,fcol,fdep) - _ease_key(tlv,tcol,tdep))
            if len(examples) < 5:
                examples.append({
                    "sku_id": sku,
                    "lot": lot,
                    "from_loc": fr,
                    "to_loc": to,
                    "qty": qty,
                    "level_change": (flv - tlv),
                    "column_change": (tcol - fcol),
                    "depth_change": (tdep - fdep),
                    "ease_gain": max(0, _ease_key(flv,fcol,fdep) - _ease_key(tlv,tcol,tdep)),
                })
        return {
            "moves": total,
            "unique_skus": len(uniq_skus),
            "locations_touched": len([x for x in touched if x and x not in ("", "00000000")]),
            "qty_cases_total": qty_total,
            "avg_level_improvement": (lvl_delta_sum/total) if total else 0.0,
            "avg_column_distance": (col_dist_sum/total) if total else 0.0,
            "avg_depth_distance": (dep_dist_sum/total) if total else 0.0,
            "moves_to_pick_levels": to_pick,
            "ease_gain_sum": ease_gain_sum,
            "ease_gain_avg": (ease_gain_sum/total) if total else 0.0,
            "highlights": examples,
        }

    try:
        # normalize moves to list[dict]
        moves_dicts = []
        for m in (moves or []):
            if isinstance(m, dict):
                # Ensure from_loc and to_loc are strings with proper padding
                dict_copy = m.copy()
                if "from_loc" in dict_copy and dict_copy["from_loc"]:
                    dict_copy["from_loc"] = str(dict_copy["from_loc"]).split('.')[0].zfill(8)
                if "to_loc" in dict_copy and dict_copy["to_loc"]:
                    dict_copy["to_loc"] = str(dict_copy["to_loc"]).split('.')[0].zfill(8)
                moves_dicts.append(dict_copy)
            else:
                try:
                    from_loc_val = getattr(m, "from_loc", getattr(m, "from", None))
                    to_loc_val = getattr(m, "to_loc", getattr(m, "to", None))
                    # Ensure locations are strings with 8 digits, remove any decimal points
                    if from_loc_val:
                        from_loc_val = str(from_loc_val).split('.')[0].zfill(8)
                    if to_loc_val:
                        to_loc_val = str(to_loc_val).split('.')[0].zfill(8)
                    
                    moves_dicts.append({
                        "sku_id": getattr(m, "sku_id", None),
                        "lot": getattr(m, "lot", None),
                        "qty": getattr(m, "qty", getattr(m, "qty_cases", None)),
                        "from_loc": from_loc_val,
                        "to_loc": to_loc_val,
                        "lot_date": getattr(m, "lot_date", None),
                        "distance": getattr(m, "distance", None),
                        "reason": getattr(m, "reason", None),
                    })
                except Exception:
                    pass
        summary = _build_summary(inv_df, moves_dicts)
    except Exception:
        summary = {"moves": len(moves or [])}

    # Compose response with summary - use normalized dicts instead of raw objects
    return {
        "moves": moves_dicts,
        "trace_id": used_trace_id,
        "rejection_summary": rej,
        "use_ai": req.use_ai,
        "use_ai_main": req.use_ai_main,
        "count": len(moves) if moves else 0,
        "max_moves": req.max_moves,
        "fill_rate": req.fill_rate,
        "chain_depth": req.chain_depth,
        "eviction_budget": req.eviction_budget,
        "touch_budget": req.touch_budget,
        "block_codes": req.block_codes,
        "quality_names": req.quality_names,
        "rotation_window_days": req.rotation_window_days,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Relocation progress stream (SSE)
# ---------------------------------------------------------------------------
@router.get("/relocation/stream")
async def relocation_progress_stream(trace_id: str | None = Query(None, description="Trace ID to subscribe to")):
    """Server-Sent Events stream of relocation progress for a given trace.

    If trace_id is omitted, the currently active trace (if any) will be used.
    """
    # SSE events feature temporarily disabled - return empty stream
    return StreamingResponse(iter(()), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Relocation debug endpoint (latest breakdown and diagnostics)
# ---------------------------------------------------------------------------
@router.get("/relocation/debug")
def relocation_last_debug(trace_id: Optional[str] = None):
    """
    Return the breakdown of dropped relocation candidates and other debug info
    captured during the most recent relocation planning run.
    
    Args:
        trace_id: Optional trace ID to retrieve specific optimization result.
                  If not provided, returns the latest result.
    """
    try:
        rej = get_last_rejection_debug() or {}
    except Exception:
        rej = {}
    try:
        rel = get_last_relocation_debug() or {}
    except Exception:
        rel = {}
    try:
        from app.services.optimizer import get_last_summary_report, get_summary_report
        if trace_id:
            summary_report = get_summary_report(trace_id)
        else:
            summary_report = get_last_summary_report()
    except Exception:
        summary_report = None

    # Compact, UI‑friendly payload
    return {
        "planned": rej.get("planned"),
        "accepted": rej.get("accepted"),
        "rejections": rej.get("rejections", {}),
        # Show up to 10 sample rows per reason (as recorded by optimizer)
        "examples": rej.get("examples", {}),
        # Additional planner‑level diagnostics if available
        "relocation": rel,
        # サマリーレポートを追加
        "summary_report": summary_report,
    }

def _df_from_sku(session: Session) -> pd.DataFrame:
    rows = session.exec(
        _sa_select(Sku.sku_id, Sku.pack_qty, Sku.volume_m3)
    ).all()
    df = pd.DataFrame(rows, columns=["商品ID", "入数", "volume_m3"]) if rows else pd.DataFrame(
        columns=["商品ID", "入数", "volume_m3"]
    )
    if df.empty:
        return df
    # 型整形
    df["入数"] = pd.to_numeric(df["入数"], errors="coerce").fillna(0).astype(int)
    df["volume_m3"] = pd.to_numeric(df["volume_m3"], errors="coerce").fillna(0.0).astype(float)
    # Optimizer 互換: 006 と同値として重複列を持たせる
    df["商品予備項目００６"] = df["volume_m3"]
    return df

def _df_from_inventory(session: Session) -> pd.DataFrame:
    rows = session.exec(
        _sa_select(
            Inventory.location_id,
            Inventory.sku_id,
            Inventory.lot,
            Inventory.lot_date,
            Inventory.qty,
            Inventory.block_code,
            Inventory.cases,
            Inventory.quality_name,
            Inventory.level,
            Inventory.column,
            Inventory.depth,
        )
    ).all()
    df = pd.DataFrame(
        rows,
        columns=[
            "ロケーション",
            "商品ID",
            "ロット",
            "lot_date",
            "在庫数(引当数を含む)",
            "ブロック略称",
            "cases",
            "品質区分名",
            "level",
            "column",
            "depth",
        ],
    ) if rows else pd.DataFrame(
        columns=[
            "ロケーション",
            "商品ID",
            "ロット",
            "lot_date",
            "在庫数(引当数を含む)",
            "ブロック略称",
            "cases",
            "品質区分名",
            "level",
            "column",
            "depth",
        ]
    )
    if df.empty:
        return df
    # 型整形
    df["在庫数(引当数を含む)"] = pd.to_numeric(df["在庫数(引当数を含む)"], errors="coerce").fillna(0).astype(int)
    df["cases"] = pd.to_numeric(df["cases"], errors="coerce").fillna(0.0).astype(float)
    df["ロケーション"] = df["ロケーション"].astype(str)
    df["商品ID"] = df["商品ID"].astype(str)
    # level/column/depth は欠損を 0 扱いでint化
    for _c in ("level", "column", "depth"):
        if _c in df.columns:
            df[_c] = pd.to_numeric(df[_c], errors="coerce").fillna(0).astype(int)
    return df

def _df_from_ship(session: Session) -> pd.DataFrame:
    rows = session.exec(
        _sa_select(ShipTx.sku_id, ShipTx.pack_qty, ShipTx.qty, ShipTx.trandate)
    ).all()
    df = pd.DataFrame(rows, columns=["item_internalid", "pack_qty", "item_shipquantity", "trandate"]) if rows else pd.DataFrame(
        columns=["item_internalid", "pack_qty", "item_shipquantity", "trandate"]
    )
    if df.empty:
        return df
    df["pack_qty"] = pd.to_numeric(df["pack_qty"], errors="coerce").fillna(0).astype(int)
    df["item_shipquantity"] = pd.to_numeric(df["item_shipquantity"], errors="coerce").fillna(0).astype(int)
    df["trandate"] = pd.to_datetime(df["trandate"], errors="coerce")
    return df


def _df_from_recv(session: Session) -> pd.DataFrame:
    rows = session.exec(
        _sa_select(RecvTx.sku_id, RecvTx.qty, RecvTx.trandate, RecvTx.lot)
    ).all()
    df = pd.DataFrame(rows, columns=["item_internalid", "item_quantity", "trandate", "lot"]) if rows else pd.DataFrame(
        columns=["item_internalid", "item_quantity", "trandate", "lot"]
    )
    if df.empty:
        return df
    df["item_quantity"] = pd.to_numeric(df["item_quantity"], errors="coerce").fillna(0).astype(int)
    df["trandate"] = pd.to_datetime(df["trandate"], errors="coerce")
    return df

# ---------------------------------------------------------------------------
# Capacity audit helper (棚容量 before/after) for relocation moves
# ---------------------------------------------------------------------------
CAP_BASE_M3 = 1.3  # 1000mm x 1000mm x 1300mm = 1.3 m^3 per shelf

def _attach_capacity_audit(
    moves: list[dict],
    sku_df: pd.DataFrame,
    inv_df: pd.DataFrame,
    fill_rate: float,
) -> list[dict]:
    """Attach per-move capacity audit (before/after usage in m^3).

    改良点:
      * ロケーションごとに `location_master.capacity_m3` を参照し、
        fill_rate を掛けた上限で判定・監査を行う（従来は固定 1.3m3）。
    """
    # Build case-volume map: SKU → ケース容積(m3) = volume_m3(単品) × 入数
    case_volume: dict[str, float] = {}
    if not sku_df.empty:
        for _, r in sku_df.iterrows():
            try:
                sku = str(r["商品ID"]) if r.get("商品ID") is not None else ""
                pack = int(r.get("入数", 0) or 0)
                vol_item = float(r.get("volume_m3", r.get("商品予備項目００６", 0.0)) or 0.0)
                case_volume[sku] = vol_item * max(pack, 0)
            except Exception:
                continue

    # Load per-location capacity map (numeric_id → capacity_m3)
    try:
        with Session(engine) as _ses:
            rows = _ses.exec(_sa_select(LocationMaster.numeric_id, LocationMaster.capacity_m3)).all()
        cap_map: dict[str, float] = {str(n): float(c or 0.0) for (n, c) in rows if n}
    except Exception:
        cap_map = {}

    def _num8(loc: str | None) -> str | None:
        if not loc:
            return None
        s = "".join(ch for ch in str(loc) if ch.isdigit())
        return s if len(s) == 8 else None

    # Initial shelf occupancy (m3)
    occ: dict[str, float] = {}
    if not inv_df.empty:
        for _, r in inv_df.iterrows():
            sku = str(r["商品ID"]) if r.get("商品ID") is not None else ""
            loc = str(r["ロケーション"]) if r.get("ロケーション") is not None else ""
            cases = float(r.get("cases", 0.0) or 0.0)
            v_case = case_volume.get(sku, 0.0)
            if loc:
                occ[loc] = occ.get(loc, 0.0) + cases * v_case

    enriched: list[dict] = []
    for m in moves:
        sku = str(m.get("sku_id", ""))
        fr = str(m.get("from_loc", ""))
        to = str(m.get("to_loc", ""))
        qty = int(m.get("qty") or 0)
        v_case = case_volume.get(sku, 0.0)

        from_before = occ.get(fr, 0.0)
        to_before = occ.get(to, 0.0)

        # apply move (for auditing state)
        occ[fr] = max(0.0, from_before - qty * v_case)
        occ[to] = to_before + qty * v_case

        # Determine destination cap using per-location capacity
        num_to = _num8(to)
        cap_base = cap_map.get(num_to, CAP_BASE_M3)
        cap_to = float(cap_base) * float(fill_rate)

        mm = {**m}
        mm["audit"] = {
            "cap_m3": cap_to,                  # destination cap after fill_rate
            "case_volume_m3": v_case,
            "from_before_m3": from_before,
            "from_after_m3": occ[fr],
            "to_before_m3": to_before,
            "to_after_m3": occ[to],
            "within_cap_to": occ[to] <= cap_to,
        }
        enriched.append(mm)

    return enriched

# ---------------------------------------------------------------------------
# Capacity enforcement (hard cap) + audit
# ---------------------------------------------------------------------------

def _enforce_capacity_and_attach_audit(
    moves: list[dict],
    sku_df: pd.DataFrame,
    inv_df: pd.DataFrame,
    fill_rate: float,
) -> list[dict]:
    """Greedily drop moves that would overflow destination capacity.

    改良点:
      * 宛先ロケーションごとに `location_master.capacity_m3` を参照し、
        fill_rate を掛けた上限で判定する（従来は固定 1.3m3）。
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"_enforce_capacity_and_attach_audit: 開始 - 移動数: {len(moves)}, fill_rate: {fill_rate}")
    # Build case-volume map: SKU → ケース容積(m3) = volume_m3(単品) × 入数
    case_volume: dict[str, float] = {}
    if not sku_df.empty:
        for _, r in sku_df.iterrows():
            try:
                sku = str(r["商品ID"]) if r.get("商品ID") is not None else ""
                pack = int(r.get("入数", 0) or 0)
                vol_item = float(r.get("volume_m3", r.get("商品予備項目００６", 0.0)) or 0.0)
                case_volume[sku] = vol_item * max(pack, 0)
            except Exception:
                continue

    # Load per-location capacity map (numeric_id → capacity_m3)
    try:
        with Session(engine) as _ses:
            rows = _ses.exec(_sa_select(LocationMaster.numeric_id, LocationMaster.capacity_m3)).all()
        cap_map: dict[str, float] = {str(n): float(c or 0.0) for (n, c) in rows if n}
    except Exception:
        cap_map = {}

    def _num8(loc: str | None) -> str | None:
        if not loc:
            return None
        s = "".join(ch for ch in str(loc) if ch.isdigit())
        return s if len(s) == 8 else None

    # Initial shelf occupancy (m3)
    occ: dict[str, float] = {}
    if not inv_df.empty:
        for _, r in inv_df.iterrows():
            sku = str(r["商品ID"]) if r.get("商品ID") is not None else ""
            loc = str(r["ロケーション"]) if r.get("ロケーション") is not None else ""
            cases = float(r.get("cases", 0.0) or 0.0)
            v_case = case_volume.get(sku, 0.0)
            if loc:
                occ[loc] = occ.get(loc, 0.0) + cases * v_case

    accepted: list[dict] = []
    for m in moves:
        try:
            sku = str(m.get("sku_id", ""))
            fr = str(m.get("from_loc", ""))
            to = str(m.get("to_loc", ""))
            qty = int(m.get("qty") or 0)
        except Exception:
            # Skip malformed rows silently
            continue

        v_case = case_volume.get(sku, 0.0)

        from_before = occ.get(fr, 0.0)
        to_before = occ.get(to, 0.0)

        # Prospective state **if** we accepted the move
        from_after = max(0.0, from_before - qty * v_case)
        to_after = to_before + qty * v_case

        # Determine destination cap using per-location capacity
        num_to = _num8(to)
        cap_base = cap_map.get(num_to, CAP_BASE_M3)
        cap_to = float(cap_base) * float(fill_rate)

        # Enforce destination hard cap
        if to_after <= cap_to:
            # Accept and commit the occupancy change
            occ[fr] = from_after
            occ[to] = to_after

            mm = {**m}
            mm["audit"] = {
                "cap_m3": cap_to,
                "case_volume_m3": v_case,
                "from_before_m3": from_before,
                "from_after_m3": from_after,
                "to_before_m3": to_before,
                "to_after_m3": to_after,
                "within_cap_to": True,
            }
            accepted.append(mm)
        else:
            # Reject move; do not mutate occupancy
            logger.debug(f"移動を拒否: {sku} {fr}→{to} (容量超過) - 移動後容量: {to_after:.3f}m3 > 上限: {cap_to:.3f}m3 (base:{cap_base:.3f} * fill_rate:{fill_rate})")
            continue

    # --- Priority ordering -------------------------------------------------
    # 目的:
    # - 効果が高い（段が下がる、ケース数が多い）移動を先に
    # - 依存関係（to_loc を空けるために from_loc=to_loc の移動）がある場合は、
    #   その“空ける側”を先に実行
    import re
    import heapq

    def _parse_loc8_tuple(loc: str) -> tuple[int, int, int]:
        s = str(loc or "").strip()
        m = re.match(r"^(\d{3})(\d{3})(\d{2})$", s)
        if not m:
            return (999, 999, 99)
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))

    # まず各移動の効果スコアを見積もる
    for mm in accepted:
        try:
            lv_from, col_from, dep_from = _parse_loc8_tuple(mm.get("from_loc"))
            lv_to, col_to, dep_to = _parse_loc8_tuple(mm.get("to_loc"))
            qty = int(mm.get("qty") or 0)
            # 段の改善を最重要（1段下げ = 1000点）+ ケース数も寄与（1ケース=10点）
            delta_lv = max(0, lv_from - lv_to)
            # 取りやすさの微小寄与（奥行き減も僅かに評価）
            delta_dep = max(0, dep_from - dep_to)
            effect = 1000 * delta_lv + 10 * qty + 1 * delta_dep
            mm.setdefault("audit", {})["priority_score"] = float(effect)
        except Exception:
            mm.setdefault("audit", {})["priority_score"] = 0.0

    # 依存関係グラフ（k -> m: k が先。条件: k.from_loc == m.to_loc）
    n = len(accepted)
    adj: list[list[int]] = [[] for _ in range(n)]
    indeg: list[int] = [0] * n
    # インデックスに安定IDを振る
    idx_map = {i: i for i in range(n)}
    for j, m in enumerate(accepted):
        to_loc = str(m.get("to_loc", ""))
        if not to_loc:
            continue
        for i, k in enumerate(accepted):
            if i == j:
                continue
            if str(k.get("from_loc", "")) == to_loc:
                adj[i].append(j)
                indeg[j] += 1

    # 優先度付きトポロジカルソート（優先度=priority_score 降順）
    heap: list[tuple[float, int]] = []
    for i in range(n):
        if indeg[i] == 0:
            pr = float(accepted[i].get("audit", {}).get("priority_score", 0.0))
            heapq.heappush(heap, (-pr, i))

    ordered: list[int] = []
    while heap:
        _neg_pr, u = heapq.heappop(heap)
        ordered.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                pr = float(accepted[v].get("audit", {}).get("priority_score", 0.0))
                heapq.heappush(heap, (-pr, v))

    # サイクル等で取り切れなかったノードは、効果スコア順で後ろに追加
    if len(ordered) < n:
        remain = [i for i in range(n) if i not in set(ordered)]
        remain.sort(key=lambda i: float(accepted[i].get("audit", {}).get("priority_score", 0.0)), reverse=True)
        ordered.extend(remain)

    # 並べ替えと sequence 付与
    accepted_sorted = [accepted[i] for i in ordered]
    for seq, mm in enumerate(accepted_sorted, start=1):
        mm.setdefault("audit", {})["sequence"] = int(seq)

    logger.info(f"_enforce_capacity_and_attach_audit: 完了 - 受け入れた移動: {len(accepted)}/{len(moves)}（優先度順に {len(accepted_sorted)} 件）")
    return accepted_sorted

# --- helper: 列番号から代表ロケーション(8桁)を選ぶ（最小使用量のロケを選択） ---
_def_col = "column"
_def_loc = "ロケーション"


def _pick_to_loc_for_column(inv_df: pd.DataFrame, col: int) -> Optional[str]:
    try:
        tmp = inv_df.copy()
        # 列列の有無にかかわらず抽出を試みる
        if _def_col not in tmp.columns:
            def _parse_col(loc: str) -> Optional[int]:
                s = "".join(ch for ch in str(loc) if ch.isdigit())
                if len(s) == 8:
                    return int(s[3:6])
                return None
            tmp[_def_col] = tmp[_def_loc].map(_parse_col)
        tmp = tmp.dropna(subset=[_def_col])
        tmp = tmp[tmp[_def_col].astype(int) == int(col)]
        if tmp.empty:
            return None
        # ロケーション単位の cases 総量が最小の場所を返す
        grp = tmp.groupby(_def_loc)["cases"].sum().reset_index().sort_values("cases", ascending=True)
        return str(grp.iloc[0][_def_loc]) if not grp.empty else None
    except Exception:
        return None


# --- helper: AIメイン（draft_moves_with_ai）を動的呼び出し ---

def _maybe_call_ai_main(
    sku_df: pd.DataFrame,
    inv_df: pd.DataFrame,
    ship_df: pd.DataFrame,
    recv_df: pd.DataFrame,
    cfg: OptimizerConfig,
    *,
    block_codes: list[str] | None,
    quality_names: list[str] | None,
    rotation_window_days: int,
    max_moves: int,
) -> list[dict]:
    try:
        mod = importlib.import_module("app.services.ai_planner")
    except Exception:
        return []
    fn = getattr(mod, "draft_moves_with_ai", None)
    if not callable(fn):
        return []
    try:
        res = fn(
            sku_master=sku_df,
            inventory=inv_df,
            ship=ship_df,
            recv=recv_df,
            cfg=cfg,
            block_codes=block_codes,
            quality_names=quality_names,
            rotation_window_days=int(rotation_window_days),
            max_moves=int(max_moves),
        )
        if isinstance(res, list):
            return res
    except Exception:
        logger.exception("ai_planner.draft_moves_with_ai failed")
        return []
    return []


# --- helper: AI応答の正規化（qty/lot/to_loc補完・厳密版） ---
PLACEHOLDER_LOCS = {"00000000", "22222222"}

def _is_placeholder_loc(loc: str | None) -> bool:
    if not loc:
        return True
    s = "".join(ch for ch in str(loc) if ch.isdigit())
    if len(s) != 8:
        return True
    if s in PLACEHOLDER_LOCS:
        return True
    # LLLCCCdd の L/C が 000 は実棚では使わない想定 → プレースホルダ扱い
    return (s[0:3] == "000") or (s[3:6] == "000")

def _normalize_ai_moves(
    ai_moves: List[dict],
    inv_df: pd.DataFrame,
    trace_id: str | None = None,
) -> List[dict]:
    """AI応答の move 配列を内部形式へ正規化。to_col→代表ロケ補完、容量チェックは別関数で実施。

    仕様:
      - (sku, lot, from_loc) が在庫に存在し、かつ lot_date が既知である行のみ採用
      - from_loc がプレースホルダ（00000000/22222222 or L/C=000）なら棄却
      - qty は from 側の cases を上限に切り詰め（0 なら棄却）
      - to_loc が未指定で to_col がある場合、同列の最小使用ロケを代表として補完
    """
    out: List[dict] = []
    drops = {
        "missing_fields": 0,
        "placeholder_from": 0,
        "unknown_lot": 0,
        "src_missing": 0,
        "to_resolve_failed": 0,
        "same_from_to": 0,
        "cut_to_zero": 0,
    }
    if inv_df is None or inv_df.empty:
        if trace_id:
            _jsonl_log({"event": "ai.normalize", "trace_id": trace_id, "raw_count": len(ai_moves or []), "normalized": 0, "drops": drops})
        return out

    # --- Build lookup maps from the scoped inventory --------------------
    # total cases by (sku, lot, loc)
    src_cases: dict[tuple[str, str, str], float] = {}
    # lot_date known?
    src_lot_date: dict[tuple[str, str, str], bool] = {}
    # reverse: (sku, loc, yyyymmdd) -> lot string
    date_to_lot: dict[tuple[str, str, str], str] = {}

    def _yyyymmdd(v) -> str | None:
        if v is None or pd.isna(v):
            return None
        try:
            dt = pd.to_datetime(v, errors="coerce")
            if pd.isna(dt):
                return None
            return dt.strftime("%Y%m%d")
        except Exception:
            return None

    tmp = inv_df.loc[:, ["商品ID", "ロット", "ロケーション", "cases", "lot_date"]].copy()
    tmp["cases"] = pd.to_numeric(tmp["cases"], errors="coerce").fillna(0.0).astype(float)
    for _, r in tmp.iterrows():
        sku = str(r["商品ID"]); lot = str(r["ロット"]); loc = str(r["ロケーション"])
        key = (sku, lot, loc)
        src_cases[key] = src_cases.get(key, 0.0) + float(r["cases"])
        has_date = _yyyymmdd(r.get("lot_date")) is not None
        src_lot_date[key] = has_date
        if has_date:
            date_key = _yyyymmdd(r.get("lot_date"))
            if date_key:
                date_to_lot[(sku, loc, date_key)] = lot

    def _pick_to_loc_for_col(col: int) -> str | None:
        try:
            return _pick_to_loc_for_column(inv_df, int(col))
        except Exception:
            return None

    # --- Normalize each proposed move ----------------------------------
    for m in (ai_moves or []):
        try:
            sku = str(m.get("sku_id") or m.get("SKU") or m.get("sku") or "").strip()
            lot = str(m.get("lot") or "").strip()
            fr  = str(m.get("from_loc") or m.get("from") or "").strip()
            toL = str(m.get("to_loc") or "").strip()
            toC = m.get("to_col", None)
            qty = m.get("qty", m.get("qty_cases"))
            qty = int(qty or 0)
        except Exception:
            drops["missing_fields"] += 1
            continue

        if not sku or not fr or qty <= 0:
            drops["missing_fields"] += 1
            continue
        if _is_placeholder_loc(fr):
            drops["placeholder_from"] += 1
            continue

        # lot が 8桁日付（YYYYMMDD）で、在庫キーに見つからない場合は逆引き補完を試す
        key = (sku, lot, fr)
        if (not src_cases.get(key, 0.0)) and re.fullmatch(r"\d{8}", lot or ""):
            alt = date_to_lot.get((sku, fr, lot))
            if alt:
                lot = alt
                key = (sku, lot, fr)

        avail = float(src_cases.get(key, 0.0))
        if avail <= 0:
            drops["src_missing"] += 1
            continue
        if not src_lot_date.get(key, False):
            drops["unknown_lot"] += 1
            continue

        # to_loc 補完
        to = toL
        if not to and (toC is not None):
            to = _pick_to_loc_for_col(toC) or ""
        if not to:
            drops["to_resolve_failed"] += 1
            continue
        if to == fr:
            drops["same_from_to"] += 1
            continue

        if qty > int(avail):
            qty = int(avail)
        if qty <= 0:
            drops["cut_to_zero"] += 1
            continue

        out.append({
            "sku_id": sku,
            "lot": lot,
            "qty": int(qty),
            "from_loc": fr,
            "to_loc": to,
        })

    if trace_id:
        _jsonl_log({
            "event": "ai.normalize",
            "trace_id": trace_id,
            "raw_count": len(ai_moves or []),
            "normalized": len(out),
            "drops": drops,
        })
    return out


# --- helper: 基本監査（FIFO/容量/入数帯域/アンカー/工数の概算） ---

def _audit_basic(moves: List[dict], sku_df: pd.DataFrame, inv_df: pd.DataFrame, fill_rate: float) -> Dict[str, Any]:
    # 事前状態の構築
    df = inv_df.copy()
    # loc→SKU集合（退避発生の近似判定に使用）
    loc_skus: Dict[str, set] = {}
    for _, r in df.iterrows():
        loc = str(r.get("ロケーション") or "")
        sku = str(r.get("商品ID") or "")
        if loc:
            loc_skus.setdefault(loc, set()).add(sku)

    # move適用（casesは近似。qtyのみ更新）
    for m in moves:
        fr = str(m.get("from_loc") or "")
        to = str(m.get("to_loc") or "")
        sku = str(m.get("sku_id") or "")
        lot = str(m.get("lot") or "")
        qty = int(m.get("qty") or 0)
        if not fr or not to or qty <= 0:
            continue
        # from 側の cases を減らし、to 側に行を追加（必要最小限）
        mask_fr = (df["ロケーション"] == fr) & (df["商品ID"] == sku) & (df["ロット"] == lot)
        if mask_fr.any():
            idx = df.index[mask_fr][0]
            df.at[idx, "cases"] = max(0.0, float(df.at[idx, "cases"] or 0.0) - qty)
        # to 行を追加（簡略化: 既存と同一キーがあれば cases 加算）
        mask_to = (df["ロケーション"] == to) & (df["商品ID"] == sku) & (df["ロット"] == lot)
        if mask_to.any():
            idx2 = df.index[mask_to][0]
            df.at[idx2, "cases"] = float(df.at[idx2, "cases"] or 0.0) + qty
        else:
            # level/column/depth をロケーションから再算出
            s = "".join(ch for ch in to if ch.isdigit())
            lvl = int(s[0:3]) if len(s) == 8 else 0
            col = int(s[3:6]) if len(s) == 8 else 0
            dep = int(s[6:8]) if len(s) == 8 else 0
            df = pd.concat([
                df,
                pd.DataFrame([{ "ロケーション": to, "商品ID": sku, "ロット": lot, "在庫数(引当数を含む)": 0, "ブロック略称": None, "cases": float(qty), "品質区分名": None, "level": lvl, "column": col, "depth": dep }])
            ], ignore_index=True)

    # FIFO違反の近似判定（同一SKU×列ごとに、lot_date昇順→level非減少をチェック）
    fifo_viol = 0
    if "lot_date" in inv_df.columns:
        # 終状態に lot_date をマージ（from 側・to 側の lot_date は元データから流用）
        lot_map: Dict[tuple, Any] = {}
        tmp = inv_df.loc[:, ["商品ID", "ロット", "ロケーション", "lot_date"]].copy()
        for _, r in tmp.iterrows():
            lot_map[(str(r["商品ID"]), str(r["ロット"]), str(r["ロケーション"]))] = r["lot_date"]
        df["lot_date_final"] = [ lot_map.get((str(r["商品ID"]), str(r["ロット"]), str(r["ロケーション"]))) for _, r in df.iterrows() ]
        sub = df.dropna(subset=["lot_date_final"]).copy()
        if not sub.empty:
            sub["lot_key"] = pd.to_datetime(sub["lot_date_final"], errors="coerce")
            grp = sub.groupby(["商品ID", "column"], dropna=False)
            for _, g in grp:
                g = g.sort_values("lot_key")
                levels = g["level"].astype(int).tolist()
                # 古いほど level が小さい（低段）であるべき → 非減少性チェック
                if any(levels[i] > levels[i+1] for i in range(len(levels)-1)):
                    fifo_viol += 1

    # 入数帯域の誤差（列代表入数との相対誤差P90）
    pack_map: Dict[str, int] = {}
    for _, r in sku_df.iterrows():
        sku = str(r.get("商品ID") or "")
        pack_map[sku] = int(r.get("入数", 0) or 0)
    # 列の代表入数（中央値）
    rep_pack: Dict[int, float] = {}
    if not df.empty and "column" in df.columns:
        tmp2 = df.copy()
        tmp2["pack"] = tmp2["商品ID"].map(pack_map).fillna(0).astype(int)
        grp2 = tmp2[tmp2["pack"] > 0].groupby("column")["pack"].median().to_dict()
        rep_pack.update({int(k): float(v) for k, v in grp2.items()})
    errors: List[float] = []
    for m in moves:
        to = str(m.get("to_loc") or "")
        s = "".join(ch for ch in to if ch.isdigit())
        col = int(s[3:6]) if len(s) == 8 else None
        sku = str(m.get("sku_id") or "")
        p = pack_map.get(sku, 0)
        rp = rep_pack.get(col, 0.0) if col is not None else 0.0
        if p > 0 and rp > 0:
            errors.append(abs(p - rp) / rp)
    pack_error_p90 = (pd.Series(errors).quantile(0.9) if errors else 0.0)

    # アンカー列（入数同質性×使用率）を推定し、そこからの退避をカウント
    anchor_cols: set = set()
    try:
        # 使用率: 列ごとのm3使用量 / (スロット数×cap_per_slot)
        cap = CAP_BASE_M3 * float(fill_rate)
        # 列ごとのスロット数（(level, depth) のユニーク数）
        slots = df.groupby("column")["depth"].nunique()
        # ケース容積マップ
        case_v: Dict[str, float] = {}
        for _, r in sku_df.iterrows():
            case_v[str(r.get("商品ID") or "")] = float(r.get("volume_m3", r.get("商品予備項目００６", 0.0)) or 0.0) * max(int(r.get("入数", 0) or 0), 0)
        # 列のm3使用量
        use_m3 = df.assign(_v=df.apply(lambda r: case_v.get(str(r["商品ID"]), 0.0) * float(r.get("cases", 0.0) or 0.0), axis=1)).groupby("column")["_v"].sum()
        util = (use_m3 / (slots.replace(0, 1) * cap)).fillna(0.0)
        # 入数同質性（±10%帯の占有率近似）: 列代表入数±10%内の割合
        tmp3 = df.copy()
        tmp3["pack"] = tmp3["商品ID"].map(pack_map).fillna(0).astype(int)
        med = tmp3[tmp3["pack"] > 0].groupby("column")["pack"].median()
        merged = tmp3.merge(med.rename("rep"), left_on="column", right_index=True, how="left")
        band = (merged["rep"] * 0.10).fillna(0.0)
        in_band = ((merged["pack"] >= (merged["rep"] - band)) & (merged["pack"] <= (merged["rep"] + band)))
        hom = in_band.groupby(merged["column"]).mean().fillna(0.0)
        # 閾値（暫定）
        anchor_cols = set(hom[(hom >= 0.7) & (util >= 0.6)].index.astype(int).tolist())
    except Exception:
        anchor_cols = set()

    # アンカー破壊＝ from側の列がアンカーで、しかも to側列が異なる場合
    anchor_breaks = 0
    if anchor_cols:
        for m in moves:
            fr = str(m.get("from_loc") or "")
            to = str(m.get("to_loc") or "")
            s1 = "".join(ch for ch in fr if ch.isdigit())
            s2 = "".join(ch for ch in to if ch.isdigit())
            c1 = int(s1[3:6]) if len(s1) == 8 else None
            c2 = int(s2[3:6]) if len(s2) == 8 else None
            if (c1 is not None) and (c1 in anchor_cols) and (c2 is not None) and (c1 != c2):
                anchor_breaks += 1

    # 工数近似
    touched = set([str(m.get("from_loc") or "") for m in moves] + [str(m.get("to_loc") or "") for m in moves])
    moved_cases_total = int(sum(int(m.get("qty") or 0) for m in moves))
    evictions = 0
    for m in moves:
        to = str(m.get("to_loc") or "")
        sku = str(m.get("sku_id") or "")
        skus_at_to = loc_skus.get(to, set())
        if to and skus_at_to and (sku not in skus_at_to):
            evictions += 1

    return {
        "fifo_violations": int(fifo_viol),
        "pack_error_p90": float(pack_error_p90) if pack_error_p90 is not None else 0.0,
        "anchor_breaks": int(anchor_breaks),
        "touched_locations": int(len(touched)),
        "evictions": int(evictions),
        "moved_cases_total": int(moved_cases_total),
    }

# --- inventory scope helper ---
def _inventory_scope(inv_df: pd.DataFrame, block_codes: list[str] | None) -> pd.DataFrame:
    """Blockコードで在庫DataFrameを絞り込む（None/空はそのまま返す）。"""
    if not isinstance(block_codes, list) or len(block_codes) == 0:
        return inv_df
    return inv_df[inv_df["ブロック略称"].isin(block_codes)].copy()

# --- AI-less fallback: build column hints by SKU ---
def _build_ai_col_hints(inv_df: pd.DataFrame, max_candidates: int = 3) -> dict[str, list[int]]:
    """
    **AI不使用のフォールバック**:
    SKUごとに現在配置されている「列(column)」の出現頻度上位を候補とする。
    """
    hints: dict[str, list[int]] = {}
    if inv_df is None or inv_df.empty:
        return hints

    # 列番号を確実に用意（column列が無い場合はロケーションから抽出）
    if "column" not in inv_df.columns:
        def _parse_col(loc: str) -> int | None:
            s = "".join(ch for ch in str(loc) if ch.isdigit())
            if len(s) == 8:
                return int(s[3:6])  # LLLCCCdd → CCC
            return None
        tmp = inv_df.copy()
        tmp["column"] = tmp["ロケーション"].map(_parse_col)
    else:
        tmp = inv_df

    tmp = tmp.dropna(subset=["column"])
    if tmp.empty:
        return hints

    # SKU×列の出現回数を集計して上位を候補に
    grp = tmp.groupby(["商品ID", "column"]).size().reset_index(name="cnt")
    grp = grp.sort_values(["商品ID", "cnt"], ascending=[True, False])

    for sku, sub in grp.groupby("商品ID"):
        top_cols = sub.head(int(max_candidates)).loc[:, "column"].astype(int).tolist()
        if top_cols:
            hints[str(sku)] = top_cols
    return hints

# --- Optionally call AI planner module if present ---
def _maybe_call_ai_planner(
    sku_df: pd.DataFrame,
    inv_df: pd.DataFrame,
    block_codes: list[str] | None,
    quality_names: list[str] | None,
    max_candidates: int,
) -> dict[str, list[int]] | None:
    """
    `app.services.ai_planner` が存在すれば動的に呼び出す。
    代表的な関数名のいずれかが見つかれば、それを用いて列ヒントdictを生成する。
    失敗時は None を返す。
    """
    try:
        mod = importlib.import_module("app.services.ai_planner")
    except Exception:
        return None

    cand_funcs = (
        "ai_rank_columns_for_skus",
        "rank_columns_for_skus",
        "rank_columns",
        "make_column_hints",
    )
    for fname in cand_funcs:
        fn = getattr(mod, fname, None)
        if callable(fn):
            try:
                # もっとも情報量の多い呼び方をトライ
                res = fn(
                    sku_df=sku_df,
                    inv_df=inv_df,
                    block_codes=block_codes,
                    quality_names=quality_names,
                    max_candidates=max_candidates,
                )
                if isinstance(res, dict):
                    return res
            except TypeError:
                # シグネチャが異なる場合に簡易版で再トライ
                try:
                    res = fn(inv_df, sku_df, block_codes, max_candidates)  # type: ignore[misc]
                    if isinstance(res, dict):
                        return res
                except Exception:
                    continue
            except Exception:
                logger.exception("ai_planner.%s failed; fallback to heuristic", fname)
                return None
    return None

# Note: The main /relocation/start endpoint is defined above (around line 2103).
# The duplicate implementation that was here (lines 3305-3625) has been removed
# to avoid routing conflicts. The current implementation supports:
# - location_master integration
# - compat shim for optimizer signature changes
# - detailed summary generation  
# - trace_id support
# - rejection_summary


def _consolidate_eviction_chains(moves: List[Dict]) -> List[Dict]:
    """
    エビクションチェーンの中間移動を統合し、最終的な移動のみを返す。
    
    同一SKU+ロットが複数回移動される場合、最初の移動元と最後の移動先のみを残し、
    中間移動を除去する。
    
    Args:
        moves: 元の移動リスト（エビクションチェーン含む）
        
    Returns:
        統合された最終移動のみのリスト
    """
    from collections import defaultdict
    
    # SKU+ロット別に移動チェーンを構築
    chains = defaultdict(list)  # (sku_id, lot) -> [(from_loc, to_loc, move_data)]
    
    for move in moves:
        sku_id = move.get('sku_id')
        lot = move.get('lot')
        from_loc = move.get('from_loc')
        to_loc = move.get('to_loc')
        
        key = (sku_id, lot)
        chains[key].append((from_loc, to_loc, move))
    
    # 各チェーンで最初と最後の移動のみを保持
    final_moves = []
    
    for (sku_id, lot), chain in chains.items():
        if len(chain) == 1:
            # 単一移動の場合はそのまま
            final_moves.append(chain[0][2])
        else:
            # 複数移動の場合は最初の移動元と最後の移動先を使用
            first_from = chain[0][0]
            last_to = chain[-1][1] 
            
            # 最後の移動データをベースに、移動元を修正
            final_move = chain[-1][2].copy()
            final_move['from_loc'] = first_from
            
            # 中間移動情報を追加
            final_move['chain_info'] = {
                'is_consolidated': True,
                'original_steps': len(chain),
                'intermediate_locations': [step[1] for step in chain[:-1]]
            }
            
            final_moves.append(final_move)
    
    return final_moves


@router.post("/relocation/start/final-moves")
def relocation_start_final_moves(req: RelocationStartRequest, session: Session = Depends(get_session)):
    """
    リロケーション最適化を実行し、エビクションチェーンを統合した最終移動のみを返す。
    
    作業員向けに、同一SKU+ロットの複数回移動を統合し、
    実際に必要な移動（開始位置→最終位置）のみを提供する。
    """
    try:
        logger = logging.getLogger(__name__)

        # 入力の軽いバリデーション
        if req.max_moves < 0:
            raise HTTPException(status_code=400, detail="max_moves must be >= 0")
        if not (0.0 <= req.fill_rate <= 1.0):
            raise HTTPException(status_code=400, detail="fill_rate must be in [0.0, 1.0]")

        # 既存のリロケーション実装を再利用して一貫した move 形式を得る
        base = relocation_start(req, session)  # dict を返す（"moves" フィールドあり）
        moves = list(base.get("moves") or [])

        # エビクションチェーンを統合（開始→最終だけ残す）
        final_moves = _consolidate_eviction_chains(moves)
        logger.info(
            "relocation/start/final-moves: consolidated %d moves to %d final moves",
            len(moves), len(final_moves)
        )

        consolidated_count = sum(1 for m in final_moves if m.get("chain_info", {}).get("is_consolidated", False))
        efficiency = (len(final_moves) / len(moves) * 100.0) if moves else 100.0

        # 返却（base のコンテキストを尊重しつつ moves を差し替え）
        return {
            "count": len(final_moves),
            "original_count": len(moves),
            "consolidated_chains": consolidated_count,
            "efficiency_percent": efficiency,
            "blocks": base.get("blocks", list(req.block_codes or [])),
            "quality_names": base.get("quality_names", list(req.quality_names or [])),
            "max_moves": base.get("max_moves", int(req.max_moves)),
            "fill_rate": base.get("fill_rate", float(req.fill_rate)),
            "chain_depth": base.get("chain_depth", int(getattr(req, "chain_depth", 0) or 0)),
            "eviction_budget": base.get("eviction_budget", int(getattr(req, "eviction_budget", 0) or 0)),
            "touch_budget": base.get("touch_budget", int(getattr(req, "touch_budget", 0) or 0)),
            "moves": final_moves,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("relocation/start/final-moves failed")
        raise HTTPException(status_code=500, detail=str(e))