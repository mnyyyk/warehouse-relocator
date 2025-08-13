"""
sitecustomize – runtime compatibility shim
Python は起動時に sitecustomize モジュールが存在すれば自動で import します。
ここで SQLModel ≤0.0.15 で削除された便利関数を復活させ、
古いテストコード／スニペットと共存させます。
"""

from sqlalchemy import select as _sa_select

import sqlmodel as _sqlmodel_mod
from sqlmodel import Session as _SQLModelSession
from sqlalchemy.orm import Session as _SAOrmSession

# sqlmodel.select の復活
if not hasattr(_sqlmodel_mod, "select"):
    _sqlmodel_mod.select = _sa_select  # type: ignore[attr-defined]

# Session.select(...) の復活
if not hasattr(_SQLModelSession, "select"):
    def _select(self, *entities, **kwargs):  # noqa: D401
        return _sa_select(*entities, **kwargs)
    _SQLModelSession.select = _select  # type: ignore[attr-defined]

if not hasattr(_SAOrmSession, "select"):
    _SAOrmSession.select = _SQLModelSession.select  # type: ignore[attr-defined]