"""
data_io.py
~~~~~~~~~~
Utility functions for loading and preprocessing raw Excel exports
from the warehouse system.

Functions
---------
load_snapshot(inv_path, sku_path, block_filter=None) -> DataFrame
    Merge inventory snapshot with SKU master, split location code,
    and calculate carton volume.

load_transactions(inbound_path, outbound_path, date_col='trandate')
    Load inbound / outbound history, parse the date column,
    and return two clean DataFrames.
"""
from __future__ import annotations

import pathlib
import re
from typing import List, Tuple

import pandas as pd

# -------------------------
# Internal helpers
# -------------------------

_LOCATION_RE = re.compile(r"(\d)(\d{3})(\d{2})")


def _parse_location_code(code: str) -> Tuple[int, int, int]:
    """
    Convert a 6‑digit location code such as '101419'
    into (level, column, depth) = (1, 14, 19).

    Parameters
    ----------
    code : str
        6‑digit numeric location code.

    Returns
    -------
    tuple[int, int, int]
    """
    match = _LOCATION_RE.fullmatch(str(code).strip())
    if match is None:
        raise ValueError(f"Invalid location code: {code}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


# -------------------------
# Public API
# -------------------------


def load_snapshot(
    inv_path: pathlib.Path,
    sku_path: pathlib.Path,
    block_filter: List[str] | None = None,
) -> pd.DataFrame:
    """
    Load the *在庫データ* and *SKU* Excel exports and enrich
    with location split & carton volume.

    Parameters
    ----------
    inv_path : pathlib.Path
        Path to `在庫データ.xlsx`.
    sku_path : pathlib.Path
        Path to `SKU.xlsx`.
    block_filter : list[str] | None, default None
        If given, keep only rows whose ブロック略称 is in this list
        (e.g. ``['B']``).  Pass ``None`` to keep all blocks.

    Returns
    -------
    pandas.DataFrame
        Inventory snapshot with at least the following columns::

            ブロック略称  ロケーション  商品ID  ロット  在庫数
            level  column  depth
            入り数  商品予備項目006  case_vol_m3
    """
    # --- load raw exports
    inv = pd.read_excel(inv_path)
    sku = pd.read_excel(sku_path)[["商品ID", "入り数", "商品予備項目006"]]

    # --- filter block (optional)
    if block_filter is not None:
        inv = inv[inv["ブロック略称"].isin(block_filter)].copy()

    # --- split location code
    loc_parts = (
        inv["ロケーション"]
        .astype(str)
        .apply(_parse_location_code)
        .apply(pd.Series)
        .rename(columns={0: "level", 1: "column", 2: "depth"})
    )
    inv = pd.concat([inv, loc_parts], axis=1)

    # --- merge SKU master & compute carton volume
    df = inv.merge(sku, on="商品ID", how="left")
    df["case_vol_m3"] = df["入り数"] * df["商品予備項目006"]

    return df.reset_index(drop=True)


def load_transactions(
    inbound_path: pathlib.Path,
    outbound_path: pathlib.Path,
    date_col: str = "trandate",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load inbound / outbound Excel exports and ensure the date column is parsed.

    Parameters
    ----------
    inbound_path : pathlib.Path
        Path to `入荷実績.xlsx`.
    outbound_path : pathlib.Path
        Path to `出荷実績.xlsx`.
    date_col : str, default 'trandate'
        Name of the column that stores the transaction date.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        (inbound_df, outbound_df) with `date_col` parsed as datetime and
        sorted ascending.
    """
    inbound = pd.read_excel(inbound_path)
    outbound = pd.read_excel(outbound_path)

    for df in (inbound, outbound):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df.sort_values(date_col, inplace=True)
        df.reset_index(drop=True, inplace=True)

    return inbound, outbound