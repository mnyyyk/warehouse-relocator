

"""
app.services package
--------------------

This ``__init__`` re‑exports high‑level helpers so that other modules
can simply::

    from app.services import ingest, detect_file_type

rather than having to know the full module path.

Only *public* objects should be re‑exported here; keep internal helpers
private to their sub‑modules.
"""

# Re‑export the ingest module itself
from . import ingest as ingest  # noqa: F401

# Re‑export selected public callables for convenience
from .ingest import (          # noqa: F401
    detect_file_type,
    process_master_file,
    process_inventory_file,
)

__all__: list[str] = [
    "ingest",
    "detect_file_type",
    "process_master_file",
    "process_inventory_file",
]