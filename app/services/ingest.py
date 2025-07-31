"""
app/services/ingest.py
----------------
Utility helpers that read the Excel files the front‑end uploads and persist the
data into the database.

The functions are intentionally *thin*: they turn an uploaded file into a
`pandas.DataFrame`, perform any lightweight enrichment (e.g. volume
calculation), and then delegate the actual *insert / update* work to the CRUD
helpers that live in `app/db/crud.py`.

These helpers are meant to be called from the FastAPI upload endpoints defined in the `app/routers/uploads.py` module that we’ll add in the next step.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, BinaryIO

__all__ = [
    "ingest_sku_master",
    "ingest_inbound_history",
    "ingest_outbound_history",
    "ingest_inventory_snapshot",
    "ingest_all",
    "detect_file_type",
    "INGEST_BY_TYPE",
    "process_master_file",
    "process_inventory_file",
    "ingest_file",
]

import pandas as pd
from sqlalchemy.orm import Session

import logging

# --------------------------------------------------------------------------- #
# Column canonicalisation helpers for Japanese‑labelled source files
# --------------------------------------------------------------------------- #
# The user's Excel files use warehouse‑specific Japanese column names.
# Map them to the canonical English field names that the rest of the pipeline
# expects.  Feel free to extend this mapping in the future as more files are
# on‑boarded.
COL_MAP: dict[str, str] = {
    # SKU master
    "商品ID": "SKU",
    "商品名": "商品名",
    "商品予備項目００３": "縦(mm)",
    "商品予備項目００４": "横(mm)",
    "商品予備項目００５": "高さ(mm)",
    "商品予備項目００６": "容積(L)",  # optional pre‑computed volume
    "入数": "ケース入数",
}


_JP_DIM_COLS = {"縦(mm)", "横(mm)", "高さ(mm)"}

# --------------------------------------------------------------------------- #
# Internal field‑name mappings                                                #
# --------------------------------------------------------------------------- #
# We keep the ingest layer decoupled from the DB schema by translating the
# canonical DataFrame column names into whatever the ORM layer expects.
# If the schema changes in the future, adjust the mapping here only.
SKU_FIELD_MAP: dict[str, str] = {
    "SKU": "sku_code",
    "商品名": "description",
    "縦(mm)": "length_mm",
    "横(mm)": "width_mm",
    "高さ(mm)": "height_mm",
    "ケース入数": "cases_per_outer",
    "volume_l": "volume_l",
}


def _canonicalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename user‑supplied Japanese columns to the internal canonical names.

    This keeps the ingest logic simple and backwards‑compatible.
    """
    df = df.rename(columns=COL_MAP)
    return df


from ..db import crud, models
from ..db import SessionLocal

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #

def _ensure_columns(df: pd.DataFrame, required: set[str], context: str) -> None:
    """Raise a helpful ValueError if the DataFrame is missing expected columns."""
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{context}: input is missing required column(s): {', '.join(missing)}"
        )


# --------------------------------------------------------------------------- #
# Session helper
# --------------------------------------------------------------------------- #


@contextmanager
def get_db(external_session: Session | None = None) -> Iterable[Session]:
    """
    Small helper that gives us either the caller‑supplied session *or* a fresh
    one that we manage locally.  This lets the same ingest function be reused
    from inside a larger transaction (e.g. in tests) or stand‑alone.
    """
    if external_session is not None:
        yield external_session
        return

    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #


def _read_excel(fp: str | Path | BinaryIO) -> pd.DataFrame:
    """
    Read an Excel file regardless of whether the caller passed a path or bytes.

    Notes
    -----
    * We rely on `engine="openpyxl"` because it supports `BytesIO` objects that
      FastAPI’s `UploadFile` gives us.
    """
    try:
        return pd.read_excel(fp, engine="openpyxl")  # type: ignore[arg-type]
    except ValueError as err:  # pragma: no cover
        # Usually raised when the file is not recognised as an Excel file.
        raise ValueError("Failed to read Excel – is the file actually a *.xlsx*?") from err


def _to_number(val):
    """
    Convert values that may contain full‑width (zenkaku) digits or commas
    into plain `float`.  Non‑numeric inputs become NaN.
    """
    if pd.isna(val):
        return float("nan")
    if isinstance(val, (int, float)):
        return float(val)
    # Replace full‑width digits ０‑９ with ascii 0‑9, drop commas
    trans = str.maketrans("０１２３４５６７８９．－", "0123456789.-")
    cleaned = str(val).translate(trans).replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return float("nan")


def _calc_volume_l(row: pd.Series) -> float:
    """
    Calculate carton‑level volume (litres) for a SKU master row.

    1 m³ = 1 000 L; 1 m³ = 1 000 000 000 mm³
    1 L  = 1 000 000 mm³
    """
    # If the source file already provides volume, prefer it.
    if pd.notna(row.get("容積(L)")):
        return _to_number(row["容積(L)"])

    # Fallback to dimensional calculation.
    mm3 = (
        _to_number(row["縦(mm)"])
        * _to_number(row["横(mm)"])
        * _to_number(row["高さ(mm)"])
    )
    cases_per_outer = _to_number(row["ケース入数"])
    if pd.isna(mm3) or pd.isna(cases_per_outer) or mm3 == 0:
        return float("nan")
    return (mm3 / 1_000_000) * cases_per_outer


# --------------------------------------------------------------------------- #
# Ingest functions
# --------------------------------------------------------------------------- #


def ingest_sku_master(
    file: str | Path | BinaryIO,
    db_session: Session | None = None,
) -> int:
    """
    Parse *SKU.xlsx* and upsert the records into the `sku` table via `crud.upsert_sku`.

    Returns
    -------
    int
        Number of rows processed.
    """
    df = _read_excel(file)

    # Canonicalise user column names → internal names
    df = _canonicalise_columns(df)
    # ---- Normalise SKU codes ------------------------------------------------
    # Excel often stores "numeric" IDs as floats (e.g. 20002000642.0).  We want
    # *string* primary keys without the trailing ".0".
    def _norm_sku(val):
        if pd.isna(val):
            return None
        if isinstance(val, (int, float)):
            # Cast to int first to drop the decimal part, then to str
            return str(int(val))
        return str(val).strip()

    df["SKU"] = df["SKU"].apply(_norm_sku)
    # ------------------------------------------------------------------
    # Drop rows where the primary key (SKU code) is missing / blank.
    # These rows would violate the NOT NULL constraint on `skus.sku_code`
    # and are usually just artefacts of empty lines at the tail of the
    # Excel sheet.
    before_len = len(df)
    df = df[df["SKU"].notna() & (df["SKU"].astype(str).str.strip() != "")]
    dropped = before_len - len(df)
    if dropped:
        logger.info("Skipped %s SKU rows with empty SKU code", dropped)

    required_cols = {"SKU", "商品名", "縦(mm)", "横(mm)", "高さ(mm)", "ケース入数"}
    _ensure_columns(df, required_cols, "SKU master")

    for col in ("縦(mm)", "横(mm)", "高さ(mm)", "ケース入数"):
        if col in df.columns:
            df[col] = df[col].apply(_to_number)

    if "volume_l" not in df.columns:
        df["volume_l"] = df.apply(_calc_volume_l, axis=1)
    logger.info("Computed/normalised volume (L) for %s SKU rows", len(df))

    with get_db(db_session) as session:
        for row in df.to_dict(orient="records"):
            # Skip rows that somehow still lack a SKU code after earlier filtering
            if not row.get("SKU"):
                continue
            # Ensure `volume_l` is always present – the DB column is NOT NULL
            if pd.isna(row.get("volume_l")) or row.get("volume_l") == "":
                row["volume_l"] = 0.0  # fallback to 0 L when calculation failed
            # Translate DataFrame keys -> model/CRUD keys, only non-null values, always include volume_l
            payload = {
                SKU_FIELD_MAP[k]: v
                for k, v in row.items()
                if k in SKU_FIELD_MAP and (
                    k == "volume_l"  # always include – ensured not‑NaN above
                    or (pd.notna(v) and v != "")
                )
            }
            # Skip insertion if the SKU code is missing after mapping
            if payload.get("sku_code"):
                crud.upsert_sku(session, payload)

    logger.info("%s rows loaded into %s", len(df), "sku")
    return len(df)


def ingest_inbound_history(
    file: str | Path | BinaryIO,
    db_session: Session | None = None,
) -> int:
    """
    Parse *入荷実績_YYYYMMDD.xlsx* and append the records to the `inbound` table.
    """
    df = _read_excel(file)

    # ------------------------------------------------------------------ #
    # 1.  Canonicalise column names coming from NetSuite export           #
    # ------------------------------------------------------------------ #
    # * item_internalid  -> SKU
    # * trandate         -> 受入日
    # * item_quantity    -> 数量
    df = df.rename(
        columns={
            "item_internalid": "SKU",
            "trandate": "受入日",
            "item_quantity": "数量",
        }
    )

    # ------------------------------------------------------------------ #
    # 2.  Keep only purchase‑order rows (“発注”); if the source does not  #
    #     contain such a hint column this step becomes a no‑op.          #
    # ------------------------------------------------------------------ #
    for hint_col in ("トランザクションタイプ", "type", "memo", "メモ"):
        if hint_col in df.columns:
            df = df[df[hint_col].astype(str).str.contains("発注", na=False)]
            break

    required_cols = {"受入日", "SKU", "数量"}
    _ensure_columns(df, required_cols, "Inbound history (発注のみ)")
    df["受入日"] = pd.to_datetime(df["受入日"], errors="coerce")

    with get_db(db_session) as session:
        for row in df.to_dict(orient="records"):
            crud.insert_inbound(session, row)

    logger.info("%s inbound rows (発注) loaded into %s", len(df), "inbound")
    return len(df)


def ingest_outbound_history(
    file: str | Path | BinaryIO,
    db_session: Session | None = None,
) -> int:
    """
    Parse *出荷実績.xlsx* and append the records to the `outbound` table.
    """
    df = _read_excel(file)

    # ------------------------------------------------------------------ #
    # 1.  Canonicalise NetSuite‑style column names                        #
    # ------------------------------------------------------------------ #
    # * item_internalid     -> SKU
    # * item_shipquantity   -> 数量
    # * internalid          -> 内部ID (contains the ship‑order id with date)
    df = df.rename(
        columns={
            "item_internalid": "SKU",
            "item_shipquantity": "数量",
            "internalid": "内部ID",
        }
    )

    # ------------------------------------------------------------------ #
    # 2.  Derive 出荷日 from the 8‑digit YYYYMMDD embedded in 内部ID       #
    #     Example: TOJP202412250012‑001  -> 2024‑12‑25                    #
    # ------------------------------------------------------------------ #
    df["出荷日"] = pd.to_datetime(
        df["内部ID"].astype(str).str.extract(r"[A-Za-z]+(\d{8})")[0],
        format="%Y%m%d",
        errors="coerce",
    )

    required_cols = {"出荷日", "SKU", "数量"}
    _ensure_columns(df, required_cols, "Outbound history")
    with get_db(db_session) as session:
        for row in df.to_dict(orient="records"):
            crud.insert_outbound(session, row)

    logger.info("%s outbound rows loaded into %s", len(df), "outbound")
    return len(df)


def ingest_inventory_snapshot(
    file: str | Path | BinaryIO,
    db_session: Session | None = None,
) -> int:
    """
    Parse the *在庫一覧.xlsx* (current‑inventory snapshot) and upsert/replace the
    records in the ``inventory`` table.

    The expected minimal columns are:
    * ``ロケーション`` – slot/location id
    * ``SKU``        – SKU code
    * ``ケース数``     – number of cases stored in that slot

    Returns
    -------
    int
        Number of rows processed.
    """
    df = _read_excel(file)
    required_cols = {"ロケーション", "SKU", "ケース数"}
    _ensure_columns(df, required_cols, "Inventory snapshot")

    with get_db(db_session) as session:
        for row in df.to_dict(orient="records"):
            # This assumes a `crud.upsert_inventory` helper exists.
            crud.upsert_inventory(session, row)

    logger.info("%s rows loaded into %s", len(df), "inventory")
    return len(df)


# --------------------------------------------------------------------------- #
# Convenience aggregate (optional)
# --------------------------------------------------------------------------- #


def ingest_all(
    sku_master: str | Path | BinaryIO,
    inbound: str | Path | BinaryIO,
    outbound: str | Path | BinaryIO,
    db_session: Session | None = None,
) -> tuple[int, int, int]:
    """
    Convenience wrapper that runs the three individual ingesters in one shot.
    Handy for tests or CLI usage.

    Returns a tuple with the count of rows loaded for (sku, inbound, outbound)
    respectively.
    """
    n_sku = ingest_sku_master(sku_master, db_session)
    n_in = ingest_inbound_history(inbound, db_session)
    n_out = ingest_outbound_history(outbound, db_session)
    return n_sku, n_in, n_out


# --------------------------------------------------------------------------- #
# File‑type detection & dispatcher
# --------------------------------------------------------------------------- #

def detect_file_type(filename: str) -> str:
    """
    Guess which kind of master file *filename* is.

    The function looks at simple keywords in the base filename and returns
    one of the strings: ``"sku"``, ``"inbound"``, or ``"outbound"``.

    It raises :class:`ValueError` if the filename does not match any of the
    expected patterns, so the caller can return a 400 error to the client.

    Examples
    --------
    >>> detect_file_type("SKU.xlsx")
    'sku'
    >>> detect_file_type("入荷実績_20250722.xlsx")
    'inbound'
    >>> detect_file_type("出荷実績.xlsx")
    'outbound'
    """
    name = Path(filename).name.lower()

    # English / romanised fall‑backs first
    if "sku" in name:
        return "sku"
    if "inbound" in name:
        return "inbound"
    if "outbound" in name:
        return "outbound"

    # Japanese originals
    if "入荷" in filename:
        return "inbound"
    if "出荷" in filename:
        return "outbound"

    if "inventory" in name:
        return "inventory"
    if "在庫" in filename:
        return "inventory"

    raise ValueError(f"Could not determine file type from: {filename!r}")


# Helper mapping so routers can dispatch generically
INGEST_BY_TYPE: dict[str, callable] = {
    "sku": ingest_sku_master,
    "inbound": ingest_inbound_history,
    "outbound": ingest_outbound_history,
    "inventory": ingest_inventory_snapshot,
}


# --------------------------------------------------------------------------- #
# Upload normalisation helper                                                 #
# --------------------------------------------------------------------------- #
def _normalise_upload(upload) -> "object":
    """
    Accept a variety of caller‑supplied objects and always return a lightweight
    proxy that exposes **both** ``filename`` and ``file`` attributes – exactly
    what the downstream ingest helpers expect.

    Supported inputs
    ----------------
    1.  FastAPI’s :class:`~fastapi.UploadFile`
    2.  A *tuple* ``(filename: str, file_like)``
    3.  A bare file‑like object that has a ``name`` attribute (e.g.
        :class:`tempfile.SpooledTemporaryFile`)

    Anything else raises :class:`TypeError`.
    """
    # Already an UploadFile‑like object?
    if hasattr(upload, "filename") and hasattr(upload, "file"):
        return upload  # type: ignore[return-value]

    # Tuple of (filename, file‑like)
    if (
        isinstance(upload, tuple)
        and len(upload) == 2
        and isinstance(upload[0], str)
        and hasattr(upload[1], "read")
    ):
        filename, fh = upload
        return type(
            "UploadProxy",
            (),
            {"filename": filename, "file": fh},
        )()

    # Bare file‑like object with a `.name`
    if hasattr(upload, "read") and hasattr(upload, "name"):
        return type(
            "UploadProxy",
            (),
            {"filename": Path(upload.name).name, "file": upload},
        )()

    raise TypeError(
        "upload must be an UploadFile‑like object, a (filename, file) tuple, "
        "or a file‑like object with a .name attribute"
    )

# --------------------------------------------------------------------------- #
# High‑level helper used by the FastAPI router
# --------------------------------------------------------------------------- #

def process_master_file(
    upload,  # expects a FastAPI UploadFile‑like object
    db_session: Session | None = None,
    db: Session | None = None,
) -> dict[str, int]:
    """
    Convenience wrapper for the upload endpoint.

    Parameters
    ----------
    upload :
        An object with ``filename`` and ``file`` attributes (e.g. FastAPI's
        :class:`~fastapi.UploadFile`).
    db_session :
        Optional SQLAlchemy session to use.  If *None*, a new managed session
        is created internally.

    Returns
    -------
    dict
        ``{"type": "<sku|inbound|outbound>", "rows": <int>}``
    """
    # Accept `db` as an alias for `db_session`
    if db_session is None and db is not None:
        db_session = db
    upload = _normalise_upload(upload)

    ingest_type = detect_file_type(upload.filename)
    ingest_fn = INGEST_BY_TYPE[ingest_type]

    # The ingest functions all expect a file‑like object (or path) as first arg
    rows = ingest_fn(upload.file, db_session)

    return {"type": ingest_type, "rows": rows}


def process_inventory_file(
    upload,  # expects a FastAPI UploadFile‑like object
    db_session: Session | None = None,
    db: Session | None = None,
) -> dict[str, int]:
    """
    Wrapper for handling inventory snapshot uploads.

    Returns ``{"type": "inventory", "rows": <int>}``.
    """
    # Accept `db` as an alias for `db_session`
    if db_session is None and db is not None:
        db_session = db
    upload = _normalise_upload(upload)

    rows = ingest_inventory_snapshot(upload.file, db_session)
    return {"type": "inventory", "rows": rows}


# --------------------------------------------------------------------------- #
# Backwards‑compatibility helper                                             #
# --------------------------------------------------------------------------- #
def ingest_file(
    upload,  # FastAPI UploadFile‑like
    db_session: Session | None = None,
    db: Session | None = None,
) -> dict[str, int]:
    """
    Legacy alias kept so older router code that still calls ``ingest.ingest_file``
    continues to work.

    It delegates to :func:`process_master_file` or
    :func:`process_inventory_file` depending on what :func:`detect_file_type`
    returns.

    Returns
    -------
    dict
        Same structure as the underlying helpers, e.g.
        ``{"type": "sku", "rows": 123}``.
    """
    # Accept `db` as an alias for `db_session`
    if db_session is None and db is not None:
        db_session = db
    upload = _normalise_upload(upload)

    ftype = detect_file_type(upload.filename)
    if ftype == "inventory":
        return process_inventory_file(upload, db_session=db_session)
    return process_master_file(upload, db_session=db_session)
