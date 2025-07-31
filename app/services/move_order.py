

"""
move_order.py
~~~~~~~~~~~~~
Utilities for determining a deadlock‑free relocation order based on
`from_loc` → `to_loc` assignments produced by the optimiser.

Key idea
--------
Treat each location as a node in a directed graph.
* Edge:  from current location  →  target location
* Acyclic graph  ->  topological sort gives a safe move sequence
* Cycles         ->  break by introducing a temporary buffer node (e.g. 'TMP')

Public API
----------
build_move_graph(df, from_col="from_loc", to_col="to_loc") -> networkx.DiGraph
    Build a graph of relocation edges.

plan_move_order(df, tmp_loc="TMP") -> DataFrame
    Add `move_order` column (1‑based) to the DataFrame so that performing
    moves in this order avoids blocking.
"""
from __future__ import annotations

from typing import List

import pandas as pd
import networkx as nx


# ----------------------------------------------------------------------
# Graph construction
# ----------------------------------------------------------------------
def build_move_graph(
    df: pd.DataFrame,
    from_col: str = "from_loc",
    to_col: str = "to_loc",
) -> nx.DiGraph:
    """
    Build a directed graph representing moves.

    Parameters
    ----------
    df : DataFrame
        Must contain `from_col` and `to_col`.
    from_col, to_col : str
        Column names for current / target locations.

    Returns
    -------
    networkx.DiGraph
        Nodes  : locations (str)
        Edges  : move from → to, with all original row indices aggregated
                 in edge attribute `rows`.
    """
    g = nx.DiGraph()

    for idx, row in df.iterrows():
        src, dst = row[from_col], row[to_col]
        if src == dst:  # no move => no edge
            continue

        if g.has_edge(src, dst):
            g[src][dst]["rows"].append(idx)
        else:
            g.add_edge(src, dst, rows=[idx])

    return g


# ----------------------------------------------------------------------
# Move ordering
# ----------------------------------------------------------------------
def _break_cycles_with_tmp(
    g: nx.DiGraph, tmp_loc: str = "TMP"
) -> nx.DiGraph:
    """
    Insert a temporary buffer node to break each cycle.

    Strategy: for each simple cycle, take its first edge (u,v)
    and replace it with two edges (u -> TMP) + (TMP -> v).

    The original row indices are carried over to the first edge
    (u -> TMP); the dummy edge (TMP -> v) has no associated rows.
    """
    g = g.copy()
    for cycle in list(nx.simple_cycles(g)):
        # pick first edge of the cycle
        u, v = cycle[0], cycle[1]
        rows = g[u][v]["rows"]
        g.remove_edge(u, v)

        # insert via TMP
        g.add_edge(u, tmp_loc, rows=rows)
        g.add_edge(tmp_loc, v, rows=[])
    return g


def plan_move_order(
    df: pd.DataFrame,
    tmp_loc: str = "TMP",
    from_col: str = "from_loc",
    to_col: str = "to_loc",
    order_col: str = "move_order",
) -> pd.DataFrame:
    """
    Determine a safe execution order.

    Parameters
    ----------
    df : DataFrame
        Optimiser output (must include from/to columns).
    tmp_loc : str, default 'TMP'
        Name of the virtual temporary buffer.
    order_col : str, default 'move_order'
        Name of the column storing the computed order (1‑based integer).

    Returns
    -------
    DataFrame
        Shallow copy of `df` with an extra `order_col`.
        Rows are sorted by this column for convenience.
    """
    g = build_move_graph(df, from_col, to_col)

    if not nx.is_directed_acyclic_graph(g):
        g = _break_cycles_with_tmp(g, tmp_loc)

    topo = list(nx.topological_sort(g))
    loc_rank = {loc: i for i, loc in enumerate(topo)}

    out = df.copy()
    out[order_col] = out[to_col].map(loc_rank).astype(int) + 1
    out.sort_values(order_col, inplace=True)

    # Re‑index for neatness
    out.reset_index(drop=True, inplace=True)
    return out