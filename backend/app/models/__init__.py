"""
Aggregate export for all SQLModel table classes.

Having each model re‑exported here guarantees that
`import app.models` will register every table in
`SQLModel.metadata`, so Alembic can discover them
during `--autogenerate`.
"""

# --------------------------------------------------------------------------- #
# Compatibility shim – restore `select` helpers removed in SQLModel ≥ 0.0.16  #
# --------------------------------------------------------------------------- #
from sqlalchemy import select as _sa_select
import sqlmodel as _sqlmodel_mod
from sqlmodel import Session as _SQLModelSession
from sqlalchemy.orm import Session as _SAOrmSession

# Re‑export `sqlmodel.select` if missing (maps to SQLAlchemy's select)
if not hasattr(_sqlmodel_mod, "select"):
    _sqlmodel_mod.select = _sa_select  # type: ignore[attr-defined]

def _compat_select(self, *entities, **kwargs):  # noqa: D401
    """Mimic legacy `Session.select()` convenience wrapper."""
    return _sa_select(*entities, **kwargs)

# Add the convenience method to both SQLModel & SQLAlchemy Session classes
if not hasattr(_SQLModelSession, "select"):
    _SQLModelSession.select = _compat_select  # type: ignore[attr-defined]
if not hasattr(_SAOrmSession, "select"):
    _SAOrmSession.select = _compat_select  # type: ignore[attr-defined]


# --- SKU & master data -----------------------------------------------------
from .sku import Sku  # noqa: F401

# --- Inventory -------------------------------------------------------------
from .inventory import Inventory  # noqa: F401

# --- Flow (ship / receive) -------------------------------------------------
from .flow import ShipTx, RecvTx  # noqa: F401

# --- Location Master -------------------------------------------------------
from .location_master import LocationMaster  # noqa: F401

# --- Metrics ---------------------------------------------------------------
from .metrics import SkuMetric  # noqa: F401

__all__ = [
    "Sku",
    "Inventory",
    "ShipTx",
    "RecvTx",
    "LocationMaster",
    "SkuMetric",
]

# --------------------------------------------------------------------------- #
# Compatibility: Session.exec() -> .scalars() for single‑model selects        #
# --------------------------------------------------------------------------- #
from sqlalchemy.sql import Select as _SASelect
from sqlmodel import SQLModel as _SQLModelBase, Session as _SQLModelSession
from sqlalchemy.orm import Session as _SAOrigSession  # noqa: F401

# Preserve original exec so we can delegate
_orig_exec = _SQLModelSession.exec  # type: ignore[attr-defined]

def _compat_exec(self, statement, *args, **kwargs):  # noqa: D401
    """
    SQLModel ≤ 0.0.15 returned a *ScalarResult* when the given statement was
    a simple ``select(MyModel)`` of *one* SQLModel subclass.  Downstream
    code could then do::

        rows = session.exec(MyModel.select()).all()  # -> [MyModel, …]

    Newer SQLModel / SQLAlchemy versions instead yield a *Result* with
    *RowMapping* objects, breaking that expectation.

    We restore the legacy behaviour by detecting this specific pattern and
    calling :pyfunc:`Result.scalars()` transparently.  For every other type
    of statement we fall back to the original exec().
    """
    res = _orig_exec(self, statement, *args, **kwargs)  # delegate first

    # ------------------------------------------------------------------
    # Heuristic: if the statement selects **exactly one** entity we treat
    #            it as the classic “Model.select()” pattern and return the
    #            scalar view so that callers get model instances instead of
    #            RowMapping objects.
    # ------------------------------------------------------------------
    #
    # This is deliberately defensive – if anything goes wrong we fall back
    # to the unmodified result so we never *break* queries that rely on the
    # newer behaviour.
    #
    try:
        from sqlalchemy.sql import Select  # local import (SQLAlchemy present)
        # 1) Check that the compiled object *is* a Select
        # 2) Check that it requested exactly 1 column / entity
        if isinstance(statement, Select) and len(statement._raw_columns) == 1:  # noqa: WPS437
            # Returning `res.scalars()` recreates the legacy SQLModel ≤0.0.15
            # contract where `session.exec(select(Model)).all()` → [Model, …]
            return res.scalars()
    except Exception:  # pragma: no cover – any inspection failure → fallback
        pass

    return res  # non‑matching → unchanged

# Patch both SQLModel.Session & SQLAlchemy.Session if missing
if not hasattr(_SQLModelSession, "_orig_exec"):  # avoid double‑patching
    _SQLModelSession.exec = _compat_exec  # type: ignore[assignment]
    _SAOrigSession.exec = _compat_exec  # type: ignore[assignment]

# --------------------------------------------------------------------------- #
# Compatibility: SQLModel.<Model>.select()                                    #
# --------------------------------------------------------------------------- #
from sqlmodel import SQLModel as _SQLModelBase  # 既に import 済みなら再読み込みは無害

if not hasattr(_SQLModelBase, "select"):
    @classmethod
    def _model_select(cls, *args, **kwargs):  # noqa: D401
        """
        従来の ``Model.select()`` API を復活させる互換パッチ。
        現在の SQLModel では削除されているため、SQLAlchemy の
        ``select(cls, *args, **kwargs)`` を返す。
        """
        return _sa_select(cls, *args, **kwargs)

    _SQLModelBase.select = _model_select  # type: ignore[attr-defined]