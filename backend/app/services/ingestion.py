from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
from fastapi import UploadFile
from sqlmodel import Session

# --------------------------------------------------------------------------- #
# Compatibility shim (same logic as in models/__init__, kept here so the file # 
# can be used standalone in notebooks/scripts)                                #
# --------------------------------------------------------------------------- #
from sqlalchemy import select as _sa_select
import sqlmodel as _sqlmodel_mod
from sqlmodel import Session as _SQLModelSession
from sqlalchemy.orm import Session as _SAOrmSession

if not hasattr(_sqlmodel_mod, "select"):
    _sqlmodel_mod.select = _sa_select  # type: ignore[attr-defined]
if not hasattr(_SQLModelSession, "select"):
    def _select(self, *entities, **kwargs):
        return _sa_select(*entities, **kwargs)
    _SQLModelSession.select = _select  # type: ignore[attr-defined]
if not hasattr(_SAOrmSession, "select"):
    _SAOrmSession.select = _SQLModelSession.select  # type: ignore[attr-defined]

# --------------------------------------------------------------------------- #
# Project imports                                                             #
# --------------------------------------------------------------------------- #
from app.utils.file_parser import read_dataframe
from app.models import Sku, Inventory, ShipTx, RecvTx
from app.routers.upload import (
    _generic_insert,
    _sku_mapper,
    _inventory_mapper,
    _ship_mapper,
    _recv_mapper,
)
from app.core.database import engine

# --------------------------------------------------------------------------- #
# Public dispatcher                                                           #
# --------------------------------------------------------------------------- #
def ingest(
    src,
    *args,
    kind: str | None = None,
    session: Session | None = None,
) -> dict:
    """
    Flexible convenience wrapper that dispatches to one of the specific
    ingest helpers (ingest_sku / ingest_inventory / ingest_ship_tx /
    ingest_recv_tx).

    See docstring in earlier conversation for accepted calling patterns.
    """
    # -------------------------------------------------------------- #
    # Allow "kind" first positional style → ingest("sku", df, ses)   #
    # -------------------------------------------------------------- #
    dispatcher_keys = {"sku", "inventory", "ship", "ship_tx", "recv", "recv_tx"}
    if isinstance(src, str) and src.lower() in dispatcher_keys and args:
        src, kind, *args = args[0], src.lower(), args[1:]

    # -------------------- extract *kind* & *session* --------------- #
    remaining = list(args)
    if kind is None:
        for i, a in enumerate(remaining):
            if isinstance(a, str):
                kind = remaining.pop(i)
                break
    if session is None:
        for i, a in enumerate(remaining):
            if isinstance(a, Session):
                session = remaining.pop(i)
                break
    if remaining:
        import warnings
        warnings.warn(
            f"Ignoring surplus positional arguments in ingest(): {remaining!r}",
            UserWarning,
            stacklevel=2,
        )

    # Infer kind from filename when not a DataFrame & not provided
    src_is_df = isinstance(src, pd.DataFrame)
    if kind is None and not src_is_df:
        fname = Path(src).name.lower()
        if "inventory" in fname:
            kind = "inventory"
        elif "ship" in fname and "tx" in fname:
            kind = "ship_tx"
        elif "recv" in fname and "tx" in fname:
            kind = "recv_tx"
        else:
            kind = "sku"

    if kind is None:
        raise ValueError("*kind* must be specified when *src* is a DataFrame")
    kind = kind.lower()

    # DataFrame sources are routed directly
    if src_is_df:
        return _ingest_df(src, kind, session=session)

    # Path / bytes / UploadFile go through file parsing
    dispatcher: dict[str, Callable[[object], dict]] = {
        "sku":       lambda s: ingest_sku(s, session=session),
        "inventory": lambda s: ingest_inventory(s, session=session),
        "ship":      lambda s: ingest_ship_tx(s, session=session),
        "ship_tx":   lambda s: ingest_ship_tx(s, session=session),
        "recv":      lambda s: ingest_recv_tx(s, session=session),
        "recv_tx":   lambda s: ingest_recv_tx(s, session=session),
    }
    try:
        return dispatcher[kind](src)
    except KeyError as exc:
        raise ValueError(f"Unknown ingest kind: {kind!r}") from exc

# --------------------------------------------------------------------------- #
# Internal helpers                                                            #
# --------------------------------------------------------------------------- #
def _ingest_df(df: pd.DataFrame, kind: str, *, session: Session | None = None) -> dict:
    """Route DataFrame directly to the corresponding ingest_* helper."""
    if session is None:
        with Session(engine) as ses:
            return _dispatch_df_to_helper(df, kind, ses)
    return _dispatch_df_to_helper(df, kind, session)

def _dispatch_df_to_helper(df: pd.DataFrame, kind: str, ses: Session) -> dict:
    kind = kind.lower()
    if kind == "sku":
        return ingest_sku(df, session=ses)
    if kind == "inventory":
        return ingest_inventory(df, session=ses)
    if kind in {"ship", "ship_tx"}:
        return ingest_ship_tx(df, session=ses)
    if kind in {"recv", "recv_tx"}:
        return ingest_recv_tx(df, session=ses)
    raise ValueError(f"Unknown ingest kind {kind!r}")

# --------------------------------------------------------------------------- #
# Kind‑specific helpers                                                       #
# --------------------------------------------------------------------------- #
def ingest_sku(src, *, session: Session | None = None) -> dict:
    df = _ensure_df(src)
    if session is None:
        with Session(engine) as ses:
            return _generic_insert(df, Sku, _sku_mapper, ses)
    return _generic_insert(df, Sku, _sku_mapper, session)

def ingest_inventory(src, *, session: Session | None = None) -> dict:
    df = _ensure_df(src)
    if session is None:
        with Session(engine) as ses:
            return _generic_insert(df, Inventory, _inventory_mapper, ses)
    return _generic_insert(df, Inventory, _inventory_mapper, session)

def ingest_ship_tx(src, *, session: Session | None = None) -> dict:
    df = _ensure_df(src)
    if session is None:
        with Session(engine) as ses:
            return _generic_insert(df, ShipTx, _ship_mapper, ses)
    return _generic_insert(df, ShipTx, _ship_mapper, session)

def ingest_recv_tx(src, *, session: Session | None = None) -> dict:
    df = _ensure_df(src)
    if session is None:
        with Session(engine) as ses:
            return _generic_insert(df, RecvTx, _recv_mapper, ses)
    return _generic_insert(df, RecvTx, _recv_mapper, session)

# --------------------------------------------------------------------------- #
# Utility                                                                     #
# --------------------------------------------------------------------------- #
def _ensure_df(src):
    """Convert src into a pandas DataFrame if necessary."""
    if isinstance(src, pd.DataFrame):
        return src
    if isinstance(src, (str, Path, UploadFile, bytes, bytearray)):
        return read_dataframe(src)
    raise TypeError(
        "ingest_* helpers accept pandas.DataFrame, path‑like, UploadFile or bytes, "
        f"got {type(src)}"
    )