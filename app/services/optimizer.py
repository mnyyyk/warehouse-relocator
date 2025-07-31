"""
optimizer.py
~~~~~~~~~~~~
Slotting / relocation optimiser using Google OR‑Tools CP‑SAT.

API
---
solve(
    inv_df: DataFrame,
    max_moves: int = 100,
    w: tuple[float, float, float] = (1.0, 1.0, 1.0),
    time_limit: int = 60,
) -> DataFrame
    Returns a DataFrame indicating the target location for each inventory
    record and whether the item will be moved or kept in place.

Notes
-----
* The warehouse is represented as the set of **existing** locations that appear
  in the snapshot plus any extra "empty slots" you add manually before calling
  this function.
* Capacity per location is hard‑coded to 95 % of 1 000 × 1 000 × 1 300 mm
  (i.e. 1.235 m³). Adjust `SLOT_CAP_M3` if your rack size differs.
"""
from __future__ import annotations

from typing import List, Tuple
import math

import pandas as pd
from ortools.sat.python import cp_model

# 1 000 × 1 000 × 1 300 = 1.3 m³ → 95 %
SLOT_CAP_M3 = 1.3 * 0.95
# NOTE: All volumes are converted to litres (integers) before modelling to
# avoid OR‑Tools auto‑scaling warnings.

# Scale factor to convert cubic‑metres to litres so that CP‑SAT sees *integer*
# coefficients.  Using integers avoids the huge debug dump of FloatAffine(...)
# expressions that OR‑Tools prints when it has to auto‑scale floating values.
SCALE_L = 1000  # 1 m³  → 1000 L
SLOT_CAP_L = int(SLOT_CAP_M3 * SCALE_L)  # capacity in litres (integer)



def _build_loc_master(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a master list of unique locations found in the snapshot.

    Returns
    -------
    DataFrame
        columns = [loc_code, level, column, depth]
    """
    loc_cols = ["ロケーション", "level", "column", "depth"]
    loc_master = (
        df[loc_cols]
        .drop_duplicates()
        .rename(columns={"ロケーション": "loc_code"})
        .sort_values(["level", "column", "depth"])
        .reset_index(drop=True)
    )
    return loc_master


# Helper to build full grid of all possible locations (including empty slots)
def _build_full_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a full rack grid covering every (level, column, depth) combination
    observed in *df* so that empty slots are also included in the CP‑SAT model.

    Examples
    --------
    If the snapshot contains level=1‑3, column=1‑10, depth=1‑2, the resulting
    DataFrame will have 3 × 10 × 2 = 60 rows.

    Returns
    -------
    DataFrame
        columns = [loc_code, level, column, depth] sorted for stable ordering.
    """
    n_levels = int(df["level"].max())
    n_columns = int(df["column"].max())
    n_depths = int(df["depth"].max())

    grid = (
        pd.MultiIndex.from_product(
            [range(1, n_levels + 1), range(1, n_columns + 1), range(1, n_depths + 1)],
            names=["level", "column", "depth"],
        )
        .to_frame(index=False)
        .sort_values(["level", "column", "depth"])
        .reset_index(drop=True)
    )

    # Zero‑pad to mimic existing location code style: LL‑CCC‑DD
    grid["loc_code"] = (
        grid["level"].astype(str).str.zfill(2)
        + "-"
        + grid["column"].astype(str).str.zfill(3)
        + "-"
        + grid["depth"].astype(str).str.zfill(2)
    )
    return grid[["loc_code", "level", "column", "depth"]]


def solve(
    inv_df: pd.DataFrame,
    max_moves: int = 100,
    w: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    time_limit: int = 60,
) -> pd.DataFrame:
    """
    Optimise slotting of inventory items.

    Parameters
    ----------
    inv_df : DataFrame
        Inventory snapshot already enriched with `pick_freq`, `age_days`,
        `case_vol_m3`, `level`, `column`, `depth`, and `cluster`.
        It **must** include `ロケーション` (current slot) as well.
    max_moves : int, default 100
        Upper bound of items that can be relocated.
    w : tuple[float, float, float], default (1,1,1)
        Weights for (pick_dist, ageing, move_cost) in the objective.
    time_limit : int, default 60
        Solver time limit in seconds.

    Returns
    -------
    DataFrame
        Columns::

            idx          : original index in inv_df
            SKU, Lot ... : passthrough columns for convenience
            from_loc     : 現在ロケーション
            to_loc       : 配置後ロケーション
            will_move    : bool
    """
    # ------------------------------------------------------------------
    # 0) Prepare indices & helper tables
    # ------------------------------------------------------------------
    inv = inv_df.reset_index().copy()  # keep original idx
    # Convert volumes to *integers* (litres) so that all coefficients passed
    # to CP‑SAT are integers and we suppress the verbose FloatAffine logs.
    inv["vol_l"] = (inv["case_vol_m3"] * SCALE_L).round().astype(int)

    # ------------------------------------------------------------------
    # Auto‑split inventory lines larger than a single slot capacity
    #   • A single pallet/lot may physically span multiple rack slots.
    #   • We therefore break such records into several "pseudo‑lines"
    #     whose individual volume ≤ SLOT_CAP_L so that the CP‑SAT model
    #     can allocate them across multiple slots.
    # ------------------------------------------------------------------
    if (inv["vol_l"] > SLOT_CAP_L).any():
        expanded_rows: List[pd.Series] = []
        for _, row in inv.iterrows():
            vol = int(row["vol_l"])
            if vol <= SLOT_CAP_L:
                expanded_rows.append(row)
            else:
                n_parts = math.ceil(vol / SLOT_CAP_L)
                remaining = vol
                for part in range(n_parts):
                    part_row = row.copy()
                    part_row["vol_l"] = min(remaining, SLOT_CAP_L)
                    # Keep the original record index (row["index"]) so that
                    # the final merge back to the raw snapshot still works.
                    # Add a helper column so we can trace which split belongs
                    # to which original record if needed.
                    part_row["oversize_part"] = part + 1
                    expanded_rows.append(part_row)
                    remaining -= part_row["vol_l"]
        # Replace the working inventory table with the expanded one
        inv = pd.DataFrame(expanded_rows).reset_index(drop=True)

    # Use full theoretical grid so that empty slots are available to the optimiser
    loc_master = _build_full_grid(inv_df)
    # ------------------------------------------------------------------
    # Quick sanity‑check: is there enough total capacity?
    #   • Instead of counting only the locations that already exist in the
    #     snapshot, use the theoretical grid size given by the highest
    #     (level, column, depth) observed in the data.  This reflects the
    #     full rack capacity even if many slots are still empty.
    # ------------------------------------------------------------------
    n_levels: int = int(inv["level"].max())
    n_columns: int = int(inv["column"].max())
    n_depths: int = int(inv["depth"].max())

    total_slots = n_levels * n_columns * n_depths
    total_capacity_l = total_slots * SLOT_CAP_L
    total_volume_l = inv["vol_l"].sum()
    if total_volume_l > total_capacity_l:
        deficit_l = total_volume_l - total_capacity_l
        missing_slots = math.ceil(deficit_l / SLOT_CAP_L)
        raise RuntimeError(
            "Total inventory volume "
            f"({total_volume_l:,} L) exceeds total slot capacity "
            f"({total_capacity_l:,} L) by {deficit_l:,} L.\n"
            "👉  Add at least "
            f"{missing_slots} more empty slot{'s' if missing_slots > 1 else ''} "
            f"(≈ {missing_slots * SLOT_CAP_L:,} L capacity) or increase "
            "`SLOT_CAP_M3` before running the optimiser."
        )

    I = range(len(inv))
    L = range(len(loc_master))

    # Pre‑compute constants used in objective
    dist_l = {
        l: loc_master.loc[l, "column"] + loc_master.loc[l, "depth"] for l in L
    }
    level_l = {l: loc_master.loc[l, "level"] for l in L}

    # ------------------------------------------------------------------
    # 1) CP‑SAT model
    # ------------------------------------------------------------------
    model = cp_model.CpModel()

    x = {
        (i, l): model.NewBoolVar(f"x_{i}_{l}")
        for i in I
        for l in L
    }

    # 1‑a) Each inventory line assigned to exactly one slot
    for i in I:
        model.Add(sum(x[i, l] for l in L) == 1)

    # 1‑b) Location capacity (volume)
    for l in L:
        model.Add(
            sum(x[i, l] * inv.loc[i, "vol_l"] for i in I) <= SLOT_CAP_L
        )

    # 1‑c) Relocation upper bound
    move_terms = []
    for i in I:
        current = inv.loc[i, "ロケーション"]
        move_terms.append(
            sum(
                x[i, l]
                * (1 if loc_master.loc[l, "loc_code"] != current else 0)
                for l in L
            )
        )
    model.Add(sum(move_terms) <= max_moves)

    # ------------------------------------------------------------------
    # 2) Objective
    # ------------------------------------------------------------------
    obj_terms = []
    for i in I:
        pick = inv.loc[i, "pick_freq"]
        age  = inv.loc[i, "age_days"]

        current = inv.loc[i, "ロケーション"]

        for l in L:
            loc_code = loc_master.loc[l, "loc_code"]
            move = 1 if loc_code != current else 0

            term = x[i, l] * (
                w[0] * pick * dist_l[l] +
                w[1] * age * level_l[l] +
                w[2] * move
            )
            obj_terms.append(term)

    model.Minimize(sum(obj_terms))

    # ------------------------------------------------------------------
    # 3) Solve
    # ------------------------------------------------------------------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 8
    # Suppress verbose CP‑SAT internal logging
    solver.parameters.log_to_stdout = False
    solver.parameters.log_search_progress = False

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible solution found by OR‑Tools.")

    # ------------------------------------------------------------------
    # 4) Build result DataFrame
    # ------------------------------------------------------------------
    assignment = []
    for i in I:
        for l in L:
            if solver.BooleanValue(x[i, l]):
                assignment.append(
                    {
                        "idx": inv.loc[i, "index"],
                        "from_loc": inv.loc[i, "ロケーション"],
                        "to_loc": loc_master.loc[l, "loc_code"],
                        "will_move": inv.loc[i, "ロケーション"]
                        != loc_master.loc[l, "loc_code"],
                    }
                )
                break

    res_df = pd.DataFrame(assignment).merge(
        inv_df, left_on="idx", right_index=True, how="left"
    )

    return res_df