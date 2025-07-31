"""
features.py
~~~~~~~~~~~
Functions that enrich the inventory snapshot DataFrame with
derived metrics required for optimisation.

Exported API
------------
add_metrics(df, outbound_df, today, freq_window_days=90) -> DataFrame
    Adds pick frequency (出庫頻度), lot age and parsed lot_date columns.

cluster_sku(df, n_clusters=10, random_state=0) -> DataFrame
    Performs K‑means clustering on carton size & 入り数 to group similar SKUs.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


# ----------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------
def add_metrics(
    df: pd.DataFrame,
    outbound_df: pd.DataFrame,
    today: pd.Timestamp,
    freq_window_days: int = 90,
) -> pd.DataFrame:
    """
    Enrich *inventory snapshot* with:
    1. pick_freq : 出庫頻度 (ケース数ではなく回数ベース)
    2. lot_date  : ロットから抽出した日付 (YYYYMMDD)
    3. age_days  : today からの経過日数
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # Sanity checks: ensure required columns exist before proceeding
    # ------------------------------------------------------------------
    inventory_required = ["商品ID", "ロット"]
    missing_inv = [c for c in inventory_required if c not in df.columns]
    if missing_inv:
        raise KeyError(
            f"add_metrics requires the inventory snapshot to include columns: {missing_inv}"
        )

    outbound_required = ["item_internalid", "item_shipquantity", "trandate"]
    missing_out = [c for c in outbound_required if c not in outbound_df.columns]
    if missing_out:
        raise KeyError(
            f"出荷実績シートに {missing_out} 列が見つかりません"
        )

    # --- outbound_df preprocessing ---
    outbound_df = outbound_df.copy()
    outbound_df.columns = outbound_df.columns.str.strip()

    if "trandate" not in outbound_df.columns:
        raise ValueError("出荷実績シートに 'trandate' 列が見つかりません")

    # セルに ndarray / list などが混在している場合は先頭要素を抽出してから日付型へ
    def _extract_scalar_date(v):
        if isinstance(v, (list, tuple, np.ndarray, pd.Series, pd.Index)):
            return v[0] if len(v) else None
        return v

    outbound_df["trandate"] = pd.to_datetime(
        outbound_df["trandate"].apply(_extract_scalar_date), errors="coerce"
    )
    # ① 出庫頻度
    window_start = today - pd.Timedelta(days=freq_window_days)
    recent = outbound_df[outbound_df["trandate"] >= window_start]

    hits = (
        recent.groupby("item_internalid")["item_shipquantity"]
        .count()
        .rename("pick_freq")
    )
    df = df.merge(hits, left_on="商品ID", right_index=True, how="left")
    df["pick_freq"] = df["pick_freq"].fillna(0)

    # ② ロット日付 & 経過日数
    lot_date_str = df["ロット"].astype(str).str.extract(r"(\d{8})")[0]
    df["lot_date"] = pd.to_datetime(lot_date_str, format="%Y%m%d", errors="coerce")
    df["age_days"] = (today - df["lot_date"]).dt.days
    df["age_days"] = df["age_days"].fillna(999)

    return df


def cluster_sku(
    df: pd.DataFrame,
    n_clusters: int = 10,
    random_state: Optional[int] = 0,
    skip_missing: bool = True,
) -> pd.DataFrame:
    """
    Cluster SKUs by carton volume & 入り数 to encourage adjacency
    during slotting optimisation.

    Parameters
    ----------
    df : DataFrame
        Inventory snapshot including carton volume and 入り数.
    n_clusters : int, default 10
        Number of K‑means clusters.
    random_state : int or None, default 0
        Random state passed to KMeans for reproducibility.
    skip_missing : bool, default True
        If True, rows that lack either `case_vol_m3` or `入り数` are **not**
        used for K‑means fitting.  These SKUs retain NaN in the `cluster`
        column so downstream logic can ignore them. If False, missing values
        are imputed with 0 and all rows are clustered.
    """
    df = df.copy()

    # --- sanity check -----------------------------------------------------
    required_cols = ["case_vol_m3", "入り数"]
    if any(col not in df.columns for col in required_cols):
        raise KeyError("cluster_sku requires `case_vol_m3` and `入り数` columns.")

    # ---------------------------------------------------------------------
    # Decide which rows participate in clustering
    # ---------------------------------------------------------------------
    if skip_missing:
        mask_valid = df[required_cols].notna().all(axis=1)
    else:
        df[required_cols] = df[required_cols].fillna(0)
        mask_valid = pd.Series(True, index=df.index)

    # Initialise cluster column with NaN so skipped SKUs remain unclustered
    df["cluster"] = np.nan

    # -----------------------------------------------------------------
    # Prevent NaNs from propagating into the optimisation model
    # -----------------------------------------------------------------
    # When `skip_missing` is True we leave rows with missing carton
    # volume or 入り数 in the DataFrame so that they can be excluded
    # downstream (e.g. by checking `cluster` is NaN).  However,
    # those NaNs must not appear inside linear‑programming coefficients.
    # Replace them with 0 so that any residual references become
    # harmless no‑ops in the solver.
    if skip_missing:
        df.loc[~mask_valid, ["case_vol_m3", "入り数"]] = df.loc[
            ~mask_valid, ["case_vol_m3", "入り数"]
        ].fillna(0)

    if mask_valid.any():  # Only run KMeans if there is something to fit
        X = df.loc[mask_valid, required_cols].to_numpy()
        km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
        df.loc[mask_valid, "cluster"] = km.fit_predict(X)

    return df