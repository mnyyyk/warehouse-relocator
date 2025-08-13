"""
sitecustomize -- 早期ロードされる互換パッチ

* SQLModel ≥ 0.0.16 で削除された
    • sqlmodel.select
    • Session.select()
  を復活させて旧テストを通す。
"""
print("[DEBUG] sitecustomize imported")

from sqlalchemy import select as _sa_select
import sqlmodel as _sqlmodel_mod
from sqlmodel import Session as _SQLModelSession
from sqlalchemy.orm import Session as _SAOrmSession


# --- sqlmodel.select ------------------------------------------------------
if not hasattr(_sqlmodel_mod, "select"):
    _sqlmodel_mod.select = _sa_select  # type: ignore[attr-defined]

# --- Session.select() convenience ----------------------------------------
def _compat_select(self, *entities, **kwargs):  # noqa: D401
    return _sa_select(*entities, **kwargs)

for _cls in (_SQLModelSession, _SAOrmSession):
    if not hasattr(_cls, "select"):
        _cls.select = _compat_select  # type: ignore[attr-defined]