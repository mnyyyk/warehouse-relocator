"""
Utility functions related to lot codes.

現在の倉庫ではロット文字列に「JP20241030」のような 2 桁アルファベット
+ 8 桁日付 (YYYYMMDD) が含まれている。
この日付を抽出して `datetime.date` として返す。

Examples
--------
>>> parse_lot_date("0001JP20241030004143")
datetime.date(2024, 10, 30)

>>> parse_lot_date("XX19991231ABC")
datetime.date(1999, 12, 31)

>>> parse_lot_date("LOT-NODATE")
None
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Optional

_LOT_DATE_PAT = re.compile(r"[A-Z]{2}(\d{8})")

def parse_lot_date(lot: str | None) -> Optional[dt.date]:
    """
    Extract YYYYMMDD from a lot string and return it as ``datetime.date``.

    Parameters
    ----------
    lot : str | None
        The original lot code.

    Returns
    -------
    datetime.date | None
        The parsed date, or ``None`` if pattern not found / invalid.
    """
    if not lot:
        return None

    match = _LOT_DATE_PAT.search(str(lot))
    if not match:
        return None

    try:
        return dt.datetime.strptime(match.group(1), "%Y%m%d").date()
    except ValueError:
        # Invalid date like 20241340
        return None
