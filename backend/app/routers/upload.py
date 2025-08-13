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
from typing import Any, Dict, List, Optional
from decimal import Decimal

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, Query
from sqlmodel import Session, SQLModel

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy import inspect, select as _sa_select, text
from pathlib import Path

from app.utils.file_parser import read_dataframe
from uuid import uuid4

import os
import logging
import math
import importlib

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
from app.services.optimizer import plan_relocation, OptimizerConfig


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
        for mdl in (Sku, Inventory, ShipTx, RecvTx)
        if not insp.has_table(mdl.__tablename__)
    ]
    if missing:
        SQLModel.metadata.create_all(
            engine,
            tables=[
                mdl.__table__
                for mdl in (Sku, Inventory, ShipTx, RecvTx)
                if mdl.__tablename__ in missing
            ],
        )

# Auto‑create missing tables only when explicitly enabled.
# Set environment variable AUTO_CREATE_TABLES=true (default) to allow this in dev.
# In production where Alembic migrations are applied, set it to false/0.
if os.getenv("AUTO_CREATE_TABLES", "true").lower() in ("1", "true", "yes"):
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

def _save_error_csv(errors: list[dict]) -> str:
    """
    Save a list of {'row': int, 'message': str} dicts as CSV and
    return the relative URL (e.g. ``/files/err_<uuid>.csv``).
    The FastAPI app must mount ``StaticFiles`` at ``/files`` separately.
    """
    if not errors:
        return ""
    df = pd.DataFrame(errors)
    fname = f"err_{uuid4().hex}.csv"
    fpath = ERROR_DIR / fname
    df.to_csv(fpath, index=False, encoding="utf-8-sig")
    return f"/files/{fname}"

# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #


def get_session() -> Session:  # dependency
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
    """
    total = len(df)
    objects: list[SQLModel] = []
    errors: list[dict] = []

    # DataFrame → モデルオブジェクトへ変換
    for idx, row in df.iterrows():
        try:
            objects.append(mapper(row))
        except Exception as exc:
            errors.append({"row": int(idx) + 2, "message": str(exc)})

    # ---------- 0‑b.  外部キー整合性チェック -------------------------------
    #
    # Inventory / ShipTx  では (sku_id, pack_qty) の複合キーで SKU マスタを
    # 参照する。一方、 RecvTx (入荷実績) は「入り数」列がなく pack_qty を
    # 判別できないため、**sku_id だけ** で突合する。
    #
    if model is RecvTx:
        # Pull a scalar list of sku_id values and normalise them.
        sku_values = session.exec(_sa_select(Sku.sku_id)).all()
        valid_sku_ids: set[str] = {_normalize_sku_id(v) for v in sku_values}

        filtered: list[SQLModel] = []
        for obj in objects:
            norm_id = _normalize_sku_id(obj.sku_id)
            if norm_id in valid_sku_ids:
                obj.sku_id = norm_id  # store canonical form
                filtered.append(obj)
            else:
                errors.append(
                    {
                        "row": "n/a",
                        "message": f"Unknown SKU:{obj.sku_id} – skipped",
                    }
                )
        objects = filtered

    elif model is ShipTx:
        # ShipTx は CSV に「入数」が含まれないケースが多い。
        # そのため、SKU マスタから sku_id → pack_qty を引き当てて埋める。
        sku_pack_map: dict[str, int] = {
            _normalize_sku_id(s): pq
            for (s, pq) in session.exec(_sa_select(Sku.sku_id, Sku.pack_qty)).all()
        }

        filtered: list[SQLModel] = []
        for obj in objects:
            canon = _normalize_sku_id(obj.sku_id)
            pack = sku_pack_map.get(canon)
            if pack is None:
                errors.append(
                    {
                        "row": "n/a",
                        "message": f"Unknown SKU:{obj.sku_id} – skipped",
                    }
                )
            else:
                obj.sku_id = canon
                obj.pack_qty = pack  # CSV 値は無視し、SKU マスタの入数を採用
                filtered.append(obj)
        objects = filtered

    elif model is Inventory:
        # Inventory CSVには入数が無い前提。
        # SKUマスタから sku_id → pack_qty を引き当てて埋める（ShipTxと同様）。
        sku_pack_map: dict[str, int] = {
            _normalize_sku_id(s): pq
            for (s, pq) in session.exec(_sa_select(Sku.sku_id, Sku.pack_qty)).all()
        }

        filtered: list[SQLModel] = []
        for obj in objects:
            canon = _normalize_sku_id(obj.sku_id)
            pack = sku_pack_map.get(canon)
            if pack is None:
                errors.append(
                    {
                        "row": "n/a",
                        "message": f"Unknown SKU:{obj.sku_id} – skipped",
                    }
                )
                continue

            obj.sku_id = canon
            obj.pack_qty = pack  # CSV値は無視し、SKUマスタの入数を採用

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
    affected = len(rows_data)          # 実際に upsert/skip 判定した行数
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
        param_budget   = 25_000
        params_per_row = max(1, len(rows_data[0]))           # keys actually present
        max_rows       = min(500, param_budget // params_per_row)

        for i in range(0, len(rows_data), max_rows):
            chunk = rows_data[i : i + max_rows]

            stmt = pg_insert(model.__table__).values(chunk)
            do_upsert = bool(use_upsert and pk_cols)
            if use_upsert and not pk_cols:
                logger.warning("_generic_insert: upsert requested but no PK detected for table=%s; falling back to plain INSERT", model.__tablename__)

            if do_upsert:
                # Only include columns that both:
                #  1) are not part of the primary key, and
                #  2) physically exist in the table (as per inspector).
                # This guards against model-only columns (e.g., a stale "id") that
                # would generate "excluded.id does not exist" in ON CONFLICT DO UPDATE.
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

            session.execute(stmt)

        session.commit()

    return {
        "total_rows": total,           # ファイルに存在した行数
        "success_rows": affected,      # upsert できた行数
        "upserted_rows": affected,
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

    # Build once per call – row.index is small.
    normalised_map = {_normalize_header(col): col for col in row.index}

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

    raise KeyError(f"None of {candidates!r} found in CSV header")

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

    # Coerce to str for cleanup
    if not isinstance(val, str):
        return int(val)

    # Normalize full‑width → half‑width, remove commas/spaces
    cleaned = unicodedata.normalize("NFKC", val).replace(",", "").strip()

    try:
        return int(cleaned)
    except Exception as exc:
        raise ValueError(f"Cannot convert {val!r} to int") from exc


# column aliases
# col = lambda *names: next((row.get(n) for n in names if n in row), None)
def _sku_mapper(row: pd.Series) -> Sku:
    sku_code = _col(
        row,
        "item_internalid",  # ← 入荷CSVと同じIDを最優先で採用
        "商品ID",
        "商品ＩＤ",
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
        length_mm=_clean_float(_col(row, "商品予備項目００３", "縦", "長さ(mm)", "length_mm")),
        width_mm=_clean_float(_col(row, "商品予備項目００４", "横", "幅(mm)", "width_mm")),
        height_mm=_clean_float(_col(row, "商品予備項目００５", "高", "高さ(mm)", "height_mm")),
        # volume は float で扱う（Decimal→float 変換）。Pydantic v2 の警告対策。
        volume_m3=_clean_float(_col(row, "商品予備項目００６", "容積", "容積(m3)", "volume_m3")),
    )


def _inventory_mapper(row: pd.Series) -> Inventory:
    loc = _col(row, "ロケーション", "location", "ロケーションコード")
    sku = _col(row, "商品ID", "SKU", "item_internalid")
    sku_norm = _normalize_sku_id(sku)
    qty = _safe_int(_col(row, "在庫数", "在庫数(引当数を含む)", "qty"))
    blk = _col(row, "ブロック略称", "ブロック名", "block_code")

    # --- Skip rows where lot is missing ---------------------------------
    lot_val = row.get("ロット") or row.get("lot")
    if lot_val is None or (isinstance(lot_val, str) and lot_val.strip() == ""):
        # Do not ingest inventory rows that have no lot information.
        raise ValueError("Missing lot")

    # 品質区分名（列が無い場合は None にフォールバック）
    try:
        quality = _col(row, "品質区分名", "quality_name")
    except KeyError:
        quality = None
    
    try:
        cases_in = _clean_float(_col(row, "cases", "ケース数"))
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
        trandate=pd.to_datetime(_col(row, "trandate", "出荷日")).date(),
    )

def _recv_mapper(row: pd.Series) -> RecvTx:
    sku_raw = _col(row, "item_internalid", "SKU", "商品ID")
    sku_norm = _normalize_sku_id(sku_raw)
    return RecvTx(
        sku_id=sku_norm,
        qty=_safe_int(_col(row, "item_quantity", "入庫数")),
        trandate=pd.to_datetime(_col(row, "trandate", "入庫日")).date(),
        lot=row.get("lot") or row.get("ロット"),
    )

# --------------------------------------------------------------------------- #
# endpoints                                                                   #
# --------------------------------------------------------------------------- #


UploadDep = Annotated[UploadFile, File(...)]
SesDep = Annotated[Session, Depends(get_session)]


@router.post("/sku")
async def upload_sku(file: UploadDep, ses: SesDep):
    """Upload SKU master (置き換えモード・常時)。

    アップロードのたびに `sku` テーブルを **TRUNCATE RESTART IDENTITY CASCADE** し、
    データを丸ごと置き換えます（参照する在庫/入出荷も CASCADE で消去）。
    その後、ファイル内容を upsert 方針で投入します（初回は実質 INSERT）。
    """
    # Always replace (truncate) before load
    ses.exec(text("TRUNCATE TABLE sku RESTART IDENTITY CASCADE"))
    ses.commit()
    df = read_dataframe(file)
    summary = _generic_insert(df, Sku, _sku_mapper, ses, use_upsert=False)
    # attach error CSV url if any
    if summary["error_rows"] > 0:
        summary["error_csv_url"] = _save_error_csv(summary["errors"])
    else:
        summary["error_csv_url"] = None
    return summary


@router.post("/inventory")
async def upload_inventory(file: UploadDep, ses: SesDep):
    # Always replace (truncate) before load
    ses.exec(text("TRUNCATE TABLE inventory RESTART IDENTITY"))
    ses.commit()
    df = read_dataframe(file)
    summary = _generic_insert(df, Inventory, _inventory_mapper, ses, use_upsert=True)
    # Backfill/derive lot_date column from lot text
    try:
        updated_cnt = _update_inventory_lot_dates(ses)
        summary["lot_date_updated_rows"] = int(updated_cnt)
    except Exception:
        logger.exception("inventory lot_date backfill failed")
        summary["lot_date_updated_rows"] = 0
    # attach error CSV url if any
    if summary["error_rows"] > 0:
        summary["error_csv_url"] = _save_error_csv(summary["errors"])
    else:
        summary["error_csv_url"] = None
    return summary


@router.post("/ship_tx")
async def upload_ship_tx(file: UploadDep, ses: SesDep):
    # Always replace (truncate) before load
    ses.exec(text("TRUNCATE TABLE ship_tx RESTART IDENTITY"))
    ses.commit()
    df = read_dataframe(file)
    summary = _generic_insert(df, ShipTx, _ship_mapper, ses, use_upsert=False)
    # attach error CSV url if any
    if summary["error_rows"] > 0:
        summary["error_csv_url"] = _save_error_csv(summary["errors"])
    else:
        summary["error_csv_url"] = None
    return summary



@router.post("/recv_tx")
async def upload_recv_tx(file: UploadDep, ses: SesDep):
    # Always replace (truncate) before load
    ses.exec(text("TRUNCATE TABLE recv_tx RESTART IDENTITY"))
    ses.commit()
    df = read_dataframe(file)
    summary = _generic_insert(df, RecvTx, _recv_mapper, ses, use_upsert=False)
    # attach error CSV url if any
    if summary["error_rows"] > 0:
        summary["error_csv_url"] = _save_error_csv(summary["errors"])
    else:
        summary["error_csv_url"] = None
    return summary


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
    window_days: int = 90
    block_codes: list[str] | None = None

@router.post("/analysis/start")
def analysis_start(req: AnalysisStartRequest, session: Session = Depends(get_session)):
    wd = int(req.window_days or 0)
    if wd <= 0:
        wd = 99_999  # 全期間扱い
    updated = recompute_all_sku_metrics(
        session,
        turnover_window_days=wd,
        block_filter=req.block_codes,
    )
    return {
        "updated": int(updated),
        "window_days_requested": int(req.window_days),
        "window_days_effective": int(wd),
        "blocks": list(req.block_codes or []),
    }


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
    chain_depth: int = 0           # 0=無効, 1..N=最大連鎖段数
    eviction_budget: int = 0       # 退避移動の総数上限
    touch_budget: int = 0          # 連鎖で触ってよいユニークロケ数の上限

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
    """Attach per-move capacity audit (before/after usage in m^3)."""
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

    cap = CAP_BASE_M3 * float(fill_rate)

    enriched: list[dict] = []
    for m in moves:
        sku = str(m.get("sku_id", ""))
        fr = str(m.get("from_loc", ""))
        to = str(m.get("to_loc", ""))
        qty = int(m.get("qty") or 0)
        v_case = case_volume.get(sku, 0.0)

        from_before = occ.get(fr, 0.0)
        to_before = occ.get(to, 0.0)

        # apply move
        occ[fr] = max(0.0, from_before - qty * v_case)
        occ[to] = to_before + qty * v_case

        m = {**m}
        m["audit"] = {
            "cap_m3": cap,
            "case_volume_m3": v_case,
            "from_before_m3": from_before,
            "from_after_m3": occ[fr],
            "to_before_m3": to_before,
            "to_after_m3": occ[to],
            "within_cap_to": occ[to] <= cap,
        }
        enriched.append(m)

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

    This replays the *proposed* move sequence while maintaining a running
    occupancy map per location (in m^3).  A move is **accepted** only if its
    application keeps the destination location at or below the allowed cap.
    Accepted moves receive the same per-move audit payload as
    :func:`_attach_capacity_audit`.  Rejected moves are omitted from the
    returned list and do **not** affect subsequent occupancy.
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

    cap = CAP_BASE_M3 * float(fill_rate)

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

        # Enforce destination hard cap
        if to_after <= cap:
            # Accept and commit the occupancy change
            occ[fr] = from_after
            occ[to] = to_after

            mm = {**m}
            mm["audit"] = {
                "cap_m3": cap,
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
            continue

    return accepted

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

@router.post("/relocation/start")
def relocation_start(req: RelocationStartRequest, session: Session = Depends(get_session)):
    try:
        # 入力データをDBからDataFrame化
        sku_df = _df_from_sku(session)
        inv_df = _df_from_inventory(session)
        logger.info("relocation/start: loaded sku_df=%d rows, inv_df=%d rows", len(sku_df), len(inv_df))

        if sku_df.empty:
            raise HTTPException(status_code=400, detail="SKUマスタが空です。先にアップロードしてください。")
        if inv_df.empty:
            raise HTTPException(status_code=400, detail="在庫データが空です。先にアップロードしてください。")

        # 品質フィルタ（良品など）
        if req.quality_names:
            inv_df = inv_df[inv_df["品質区分名"].isin(req.quality_names)].copy()
            logger.info("relocation/start: after quality filter %s → %d rows", req.quality_names, len(inv_df))
            if inv_df.empty:
                return {
                    "count": 0,
                    "blocks": list(req.block_codes or []),
                    "quality_names": list(req.quality_names or []),
                    "max_moves": int(req.max_moves),
                    "fill_rate": float(req.fill_rate),
                    "use_ai": bool(req.use_ai),
                    "ai_hints_skus": 0,
                    "moves": [],
                }

        # 商品予備項目006(volume_m3)があるSKUのみ対象に絞る（オプション）
        if req.require_volume and not sku_df.empty:
            ok_skus = set(sku_df.loc[(pd.to_numeric(sku_df["volume_m3"], errors="coerce").fillna(0.0) > 0.0), "商品ID"])
            if ok_skus:
                inv_df = inv_df[inv_df["商品ID"].isin(ok_skus)].copy()
                sku_df = sku_df[sku_df["商品ID"].isin(ok_skus)].copy()
            logger.info("relocation/start: after volume filter → sku:%d inv:%d", len(sku_df), len(inv_df))
            if inv_df.empty:
                return {
                    "count": 0,
                    "blocks": list(req.block_codes or []),
                    "quality_names": list(req.quality_names or []),
                    "max_moves": int(req.max_moves),
                    "fill_rate": float(req.fill_rate),
                    "use_ai": bool(req.use_ai),
                    "ai_hints_skus": 0,
                    "moves": [],
                }

        # === AIメイン経路（オプション） =====================================
        if getattr(req, "use_ai_main", False):
            # ブロックで在庫をスコープ
            inv_scope = _inventory_scope(inv_df, req.block_codes)
            # 入出荷を取得（回転スコア用の元データ）
            ship_df = _df_from_ship(session)
            recv_df = _df_from_recv(session)

            # ---- tracing ----
            trace_id = uuid4().hex[:12]
            _jsonl_log({
                "event": "ai.start",
                "trace_id": trace_id,
                "params": {
                    "blocks": req.block_codes,
                    "quality_names": req.quality_names,
                    "rotation_window_days": int(getattr(req, "rotation_window_days", 90) or 90),
                    "max_moves": int(req.max_moves),
                    "fill_rate": float(req.fill_rate),
                    "use_ai_main": bool(getattr(req, "use_ai_main", False)),
                    "chain_depth": int(getattr(req, "chain_depth", 0) or 0),
                    "eviction_budget": int(getattr(req, "eviction_budget", 0) or 0),
                    "touch_budget": int(getattr(req, "touch_budget", 0) or 0),
                },
                "scope": {
                    "rows": int(inv_scope.shape[0]),
                    "skus": int(inv_scope["商品ID"].nunique()) if "商品ID" in inv_scope.columns else None,
                    "locations": int(inv_scope["ロケーション"].nunique()) if "ロケーション" in inv_scope.columns else None,
                    "unknown_lot_rows": int(inv_scope["lot_date"].isna().sum()) if "lot_date" in inv_scope.columns else None,
                },
            })
            # ---------------

            cfg = OptimizerConfig(
                max_moves=req.max_moves,
                fill_rate=req.fill_rate,
                chain_depth=getattr(req, "chain_depth", 0),
                eviction_budget=getattr(req, "eviction_budget", 0),
                touch_budget=getattr(req, "touch_budget", 0),
            )
            ai_moves_raw = _maybe_call_ai_main(
                sku_df=sku_df,
                inv_df=inv_scope,
                ship_df=ship_df,
                recv_df=recv_df,
                cfg=cfg,
                block_codes=req.block_codes,
                quality_names=req.quality_names,
                rotation_window_days=int(getattr(req, "rotation_window_days", 90) or 90),
                max_moves=int(req.max_moves),
            )

            # draft が得られなければ Greedy にフォールバック
            if not ai_moves_raw:
                logger.warning("AI main planner returned no moves; falling back to Greedy")
                _jsonl_log({"event": "ai.fallback", "trace_id": trace_id, "reason": "ai_main_empty_or_error"})
            else:
                # to_loc 補完など正規化
                ai_moves = _normalize_ai_moves(ai_moves_raw, inv_scope, trace_id=trace_id)
                # 容量ハードキャップ適用 + 監査情報付与
                accepted = _enforce_capacity_and_attach_audit(ai_moves[: int(req.max_moves)], sku_df, inv_scope, req.fill_rate)

                # lot_date 付与（既存と同じロジック）
                lot_date_map: dict[tuple[str, str, str], str] = {}
                try:
                    if (not inv_scope.empty) and ("lot_date" in inv_scope.columns):
                        tmp = inv_scope.loc[:, ["商品ID", "ロット", "ロケーション", "lot_date"]].copy()
                        def _to_iso(v):
                            import pandas as _pd
                            from datetime import date as _date, datetime as _dt
                            if v is None or (isinstance(v, float) and _pd.isna(v)):
                                return None
                            if isinstance(v, _pd.Timestamp):
                                return v.date().isoformat()
                            if isinstance(v, _dt):
                                return v.date().isoformat()
                            if isinstance(v, _date):
                                return v.isoformat()
                            return str(v)
                        tmp["lot_date"] = tmp["lot_date"].map(_to_iso)
                        tmp = tmp.dropna(subset=["lot_date"])
                        for _, r in tmp.iterrows():
                            sku = str(r["商品ID"]) if r.get("商品ID") is not None else ""
                            lot = str(r["ロット"]) if r.get("ロット") is not None else ""
                            loc = str(r["ロケーション"]) if r.get("ロケーション") is not None else ""
                            if sku and lot and loc:
                                lot_date_map[(sku, lot, loc)] = str(r["lot_date"])  # ISO
                except Exception:
                    pass

                for m in accepted:
                    try:
                        sku_str = str(m.get("sku_id") or "")
                        lot_val = str(m.get("lot") or "")
                        fr = str(m.get("from_loc") or "")
                        to = str(m.get("to_loc") or "")
                        m["lot_date"] = (
                            lot_date_map.get((sku_str, lot_val, fr))
                            or lot_date_map.get((sku_str, lot_val, to))
                            or None
                        )
                    except Exception:
                        continue

                # 監査サマリ
                violations = _audit_basic(accepted, sku_df, inv_scope, req.fill_rate)

                _jsonl_log({
                    "event": "ai.capacity",
                    "trace_id": trace_id,
                    "accepted": len(accepted),
                    "requested_max": int(req.max_moves),
                    "fill_rate": float(req.fill_rate),
                })
                _jsonl_log({
                    "event": "ai.audit",
                    "trace_id": trace_id,
                    "violations": violations,
                })
                _jsonl_log({
                    "event": "ai.success",
                    "trace_id": trace_id,
                    "moves": len(accepted),
                    "use_ai_main": True,
                })

                logger.info("relocation/start (ai_main): accepted=%d (requested max=%d)", len(accepted), int(req.max_moves))
                return {
                    "count": len(accepted),
                    "blocks": list(req.block_codes or []),
                    "quality_names": list(req.quality_names or []),
                    "max_moves": int(req.max_moves),
                    "fill_rate": float(req.fill_rate),
                    "use_ai": bool(req.use_ai),
                    "use_ai_main": True,
                    "chain_depth": int(getattr(req, "chain_depth", 0) or 0),
                    "eviction_budget": int(getattr(req, "eviction_budget", 0) or 0),
                    "touch_budget": int(getattr(req, "touch_budget", 0) or 0),
                    "ai_hints_skus": None,
                    "violations_summary": violations,
                    "moves": accepted,
                    "trace_id": trace_id,
                }
        # === /AIメイン経路 ================================================

        # AIヒントの構築（ブロック絞り込み後の在庫で作成）
        inv_scope = _inventory_scope(inv_df, req.block_codes)
        ai_hints: dict[str, list[int]] | None = None
        if bool(req.use_ai):
            ai_hints = _maybe_call_ai_planner(
                sku_df=sku_df,
                inv_df=inv_scope,
                block_codes=req.block_codes,
                quality_names=req.quality_names,
                max_candidates=int(req.ai_max_candidates),
            )
            if not ai_hints:
                ai_hints = _build_ai_col_hints(inv_scope, max_candidates=int(req.ai_max_candidates))

        # Greedy 最適化を実行（AIヒントを渡す）
        cfg = OptimizerConfig(
            max_moves=req.max_moves,
            fill_rate=req.fill_rate,
            chain_depth=getattr(req, "chain_depth", 0),
            eviction_budget=getattr(req, "eviction_budget", 0),
            touch_budget=getattr(req, "touch_budget", 0),
        )
        moves = plan_relocation(
            sku_master=sku_df,
            inventory=inv_df,
            cfg=cfg,
            block_filter=req.block_codes,
            ai_col_hints=ai_hints or {},
        )

        logger.debug("relocation/start: plan_relocation returned %d moves", len(moves))

        # --- Build lot_date lookup: (SKU, lot, location) -> ISO date string ---
        lot_date_map: dict[tuple[str, str, str], str] = {}
        try:
            scope_for_dates = _inventory_scope(inv_df, req.block_codes)
            if (not scope_for_dates.empty) and ("lot_date" in scope_for_dates.columns):
                tmp = scope_for_dates.loc[:, ["商品ID", "ロット", "ロケーション", "lot_date"]].copy()

                def _to_iso(v):
                    import pandas as _pd
                    from datetime import date as _date, datetime as _dt
                    if v is None or (isinstance(v, float) and _pd.isna(v)):
                        return None
                    if isinstance(v, _pd.Timestamp):
                        return v.date().isoformat()
                    if isinstance(v, _dt):
                        return v.date().isoformat()
                    if isinstance(v, _date):
                        return v.isoformat()
                    # already string-like
                    return str(v)

                tmp["lot_date"] = tmp["lot_date"].map(_to_iso)
                tmp = tmp.dropna(subset=["lot_date"])  # keep only rows with a parsed date
                for _, r in tmp.iterrows():
                    sku = str(r["商品ID"]) if r.get("商品ID") is not None else ""
                    lot = str(r["ロット"]) if r.get("ロット") is not None else ""
                    loc = str(r["ロケーション"]) if r.get("ロケーション") is not None else ""
                    if sku and lot and loc:
                        lot_date_map[(sku, lot, loc)] = str(r["lot_date"])  # ISO yyyy-mm-dd
        except Exception:
            # mapping is best-effort
            pass

        # NamedTuple → JSON 変換（qtyはケース単位：切り上げ）
        out = []
        for m in moves:
            try:
                lot_val = m.lot
                if lot_val is None or (isinstance(lot_val, float) and math.isnan(lot_val)):
                    lot_val = ""
                sku_str = str(m.sku_id) if m.sku_id is not None else ""
                from_loc_str = str(m.from_loc) if m.from_loc is not None else ""
                to_loc_str = str(m.to_loc) if m.to_loc is not None else ""

                # prefer the source location's lot_date; fallback to destination if needed
                lot_date_str = (
                    lot_date_map.get((sku_str, str(lot_val), from_loc_str))
                    or lot_date_map.get((sku_str, str(lot_val), to_loc_str))
                    or None
                )

                out.append({
                    "sku_id": sku_str,
                    "lot": str(lot_val),
                    "lot_date": lot_date_str,  # <- NEW: ISO yyyy-mm-dd or null
                    "qty": int(m.qty) if m.qty is not None else 0,
                    "from_loc": from_loc_str,
                    "to_loc": to_loc_str,
                })
            except Exception:
                logger.exception("relocation/start: failed to serialize a move row; skipping")
                continue

        # Enforce hard capacity cap and attach audit info
        out = _enforce_capacity_and_attach_audit(out, sku_df, _inventory_scope(inv_df, req.block_codes), req.fill_rate)

        logger.info("relocation/start: returning %d moves (ai_hints_skus=%d)", len(out), 0 if not ai_hints else len(ai_hints))
        return {
            "count": len(out),
            "blocks": list(req.block_codes or []),
            "quality_names": list(req.quality_names or []),
            "max_moves": int(req.max_moves),
            "fill_rate": float(req.fill_rate),
            "use_ai": bool(req.use_ai),
            "use_ai_main": False,
            "chain_depth": int(getattr(req, "chain_depth", 0) or 0),
            "eviction_budget": int(getattr(req, "eviction_budget", 0) or 0),
            "touch_budget": int(getattr(req, "touch_budget", 0) or 0),
            "ai_hints_skus": 0 if not ai_hints else len(ai_hints),
            "moves": out,
            "trace_id": locals().get("trace_id"),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("relocation/start failed")
        raise HTTPException(status_code=500, detail=str(e))