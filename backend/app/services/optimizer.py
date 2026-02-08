from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict, Set, Any
import asyncio
import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache

import pandas as pd
import numpy as np
import copy
import os
import secrets

# -------------------------------
# Constants / config
# -------------------------------
SHELF_WIDTH_M = 1.0
SHELF_DEPTH_M = 1.0
SHELF_HEIGHT_M = 1.3
DEFAULT_FILL_RATE = 0.90
UNKNOWN_LOT_KEY = 99_999_999


# プレースホルダロケ（集約・仮置きなど）
PLACEHOLDER_LOCS = {"00000000", "22222222"}

# --------------------------------
# Debug tracker for rejection breakdown (enforce_constraints)
# --------------------------------
_last_relocation_debug: Dict[str, Any] = {
    "trace_id": None,
    "planned": 0,
    "accepted": 0,
    "rejections": {
        "oversize": 0,
        "forbidden": 0,
        "capacity": 0,
        "fifo": 0,
        "pack_band": 0,
        "other": 0,
    },
    "examples": {
        "oversize": [],
        "forbidden": [],
        "capacity": [],
        "fifo": [],
        "pack_band": [],
        "other": [],
    },
}

def _dbg_reset(trace_id: Optional[str] = None) -> None:
    global _last_relocation_debug
    _last_relocation_debug = {
        "trace_id": trace_id,
        "planned": 0,
        "accepted": 0,
        "rejections": {
            "oversize": 0,
            "forbidden": 0,
            "capacity": 0,
            "fifo": 0,
            "pack_band": 0,
            "other": 0,
        },
        "examples": {
            "oversize": [],
            "forbidden": [],
            "capacity": [],
            "fifo": [],
            "pack_band": [],
            "other": [],
        },
    }

def _dbg_note_planned(n: int) -> None:
    _last_relocation_debug["planned"] = int(n)

def _dbg_note_accepted(n: int) -> None:
    _last_relocation_debug["accepted"] = int(n)

def _dbg_reject(reason: str, move_dict: Dict[str, Any], note: str = "") -> None:
    r = reason if reason in _last_relocation_debug["rejections"] else "other"
    _last_relocation_debug["rejections"][r] += 1
    ex = _last_relocation_debug["examples"][r]
    if len(ex) < 10:
        ex.append({
            "sku_id": str(move_dict.get("sku_id", "")),
            "lot": str(move_dict.get("lot", "")),
            "from": str(move_dict.get("from", move_dict.get("from_loc", ""))),
            "to_loc": str(move_dict.get("to_loc", "")),
            "to_col": move_dict.get("to_col"),
            "qty_cases": move_dict.get("qty_cases"),
            "note": str(note)[:160],
        })
    # mirror into drop store when a trace is active
    _record_drop(r, {
        "sku_id": move_dict.get("sku_id"),
        "lot": move_dict.get("lot"),
        "from_loc": move_dict.get("from", move_dict.get("from_loc")),
        "to_loc": move_dict.get("to_loc"),
        "to_col": move_dict.get("to_col"),
        "qty_cases": move_dict.get("qty_cases"),
    }, note=note)

def get_last_rejection_debug() -> Dict[str, Any]:
    """Return a deep copy of the last relocation rejection breakdown."""
    return copy.deepcopy(_last_relocation_debug)

# Alias (both names available)
def get_last_relocation_debug() -> Dict[str, Any]:
    return get_last_rejection_debug()

def get_last_summary_report() -> Optional[str]:
    """Return the last generated summary report."""
    return _LAST_SUMMARY_REPORT

def get_summary_report(trace_id: str) -> Optional[str]:
    """Return the summary report for a specific trace_id."""
    return _SUMMARY_REPORTS.get(trace_id)

# --------------------------------
# Current trace id binding & drop recording helper
# --------------------------------
_CURRENT_TRACE_ID: Optional[str] = None
_DROP_STORE: Dict[str, List[Dict[str, Any]]] = {}
_DROP_LIMIT_PER_TRACE = 20000  # safety cap per trace
_LAST_SUMMARY_REPORT: Optional[str] = None  # サマリーレポートを保存（後方互換性のため残す）
_SUMMARY_REPORTS: Dict[str, str] = {}  # trace_id -> サマリーレポートのマッピング
_SUMMARY_REPORTS_LIMIT = 50  # 最大保存件数

# 移動データキャッシュ（タイムアウト後もフロントエンドが取得できるように）
_MOVES_CACHE: Dict[str, List[Dict[str, Any]]] = {}  # trace_id -> moves
_MOVES_CACHE_LIMIT = 10  # 最大保存件数

# --------------------------------
# Global inventory indexes for fast lookup (shared across functions)
# These are populated by plan_relocation and used by eviction chain functions
# --------------------------------
_GLOBAL_INV_INDEXES: Dict[str, Any] = {
    "inv_lots_by_loc_sku": {},      # (location, sku) -> set of lot_keys
    "inv_cols_by_sku_lot": {},      # (sku, lot_key) -> set of columns
    "inv_lot_levels_by_sku_col": {},  # (sku, column) -> list of (lot_key, level)
    "inv_rows_by_loc": {},          # location -> list of row indices (for eviction chain)
}

def _reset_global_inv_indexes() -> None:
    """Reset global inventory indexes."""
    global _GLOBAL_INV_INDEXES
    _GLOBAL_INV_INDEXES = {
        "inv_lots_by_loc_sku": {},
        "inv_cols_by_sku_lot": {},
        "inv_lot_levels_by_sku_col": {},
        "inv_rows_by_loc": {},
    }

def _build_global_inv_indexes(inv: pd.DataFrame) -> None:
    """Build global inventory indexes for fast lookup in eviction chain functions."""
    global _GLOBAL_INV_INDEXES
    _reset_global_inv_indexes()
    
    if inv.empty or "商品ID" not in inv.columns or "lot_key" not in inv.columns:
        return
    
    inv_lots_by_loc_sku: Dict[Tuple[str, str], Set[int]] = {}
    inv_cols_by_sku_lot: Dict[Tuple[str, int], Set[int]] = {}
    inv_lot_levels_by_sku_col: Dict[Tuple[str, int], List[Tuple[int, int]]] = {}
    inv_rows_by_loc: Dict[str, List[int]] = {}
    
    for idx_row in inv.index:
        try:
            sku_v = str(inv.at[idx_row, "商品ID"])
            loc_v = str(inv.at[idx_row, "ロケーション"]) if "ロケーション" in inv.columns else ""
            lot_k = int(pd.to_numeric(inv.at[idx_row, "lot_key"], errors="coerce") or UNKNOWN_LOT_KEY)
            col_v = int(inv.at[idx_row, "col"]) if "col" in inv.columns and pd.notna(inv.at[idx_row, "col"]) else 0
            lv_v = int(inv.at[idx_row, "lv"]) if "lv" in inv.columns and pd.notna(inv.at[idx_row, "lv"]) else 0
            
            # Index 1: lots by (location, sku)
            key1 = (loc_v, sku_v)
            if key1 not in inv_lots_by_loc_sku:
                inv_lots_by_loc_sku[key1] = set()
            inv_lots_by_loc_sku[key1].add(lot_k)
            
            # Index 2: columns by (sku, lot_key)
            key2 = (sku_v, lot_k)
            if key2 not in inv_cols_by_sku_lot:
                inv_cols_by_sku_lot[key2] = set()
            inv_cols_by_sku_lot[key2].add(col_v)
            
            # Index 3: lot_key and level by (sku, column)
            key3 = (sku_v, col_v)
            if key3 not in inv_lot_levels_by_sku_col:
                inv_lot_levels_by_sku_col[key3] = []
            inv_lot_levels_by_sku_col[key3].append((lot_k, lv_v))
            
            # Index 4: row indices by location (for eviction chain)
            if loc_v not in inv_rows_by_loc:
                inv_rows_by_loc[loc_v] = []
            inv_rows_by_loc[loc_v].append(idx_row)
        except Exception:
            pass
    
    _GLOBAL_INV_INDEXES["inv_lots_by_loc_sku"] = inv_lots_by_loc_sku
    _GLOBAL_INV_INDEXES["inv_cols_by_sku_lot"] = inv_cols_by_sku_lot
    _GLOBAL_INV_INDEXES["inv_lot_levels_by_sku_col"] = inv_lot_levels_by_sku_col
    _GLOBAL_INV_INDEXES["inv_rows_by_loc"] = inv_rows_by_loc

def cache_moves(trace_id: str, moves: List[Dict[str, Any]]) -> None:
    """Cache moves for a given trace_id (for timeout recovery)."""
    global _MOVES_CACHE
    if len(_MOVES_CACHE) >= _MOVES_CACHE_LIMIT:
        # 古いものを削除
        oldest = next(iter(_MOVES_CACHE))
        del _MOVES_CACHE[oldest]
    _MOVES_CACHE[trace_id] = moves

def get_cached_moves(trace_id: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
    """Get cached moves for a given trace_id, or the latest if not specified."""
    if trace_id:
        return _MOVES_CACHE.get(trace_id)
    if _MOVES_CACHE:
        # 最新を返す
        return list(_MOVES_CACHE.values())[-1]
    return None

# --------------------------------
# Progress SSE (simple in-memory pub/sub per trace)
# --------------------------------
_TRACE_SUBS: Dict[str, List[asyncio.Queue[str]]] = {}
_TRACE_BUFFER: Dict[str, deque[str]] = {}
_TRACE_BUFFER_MAX = 200

def _publish_progress(trace_id: Optional[str], event: Dict[str, Any]) -> None:
    """Publish a progress event to subscribers and store in a small ring buffer."""
    try:
        if not trace_id:
            return
        tid = str(trace_id)
        payload = dict(event)
        payload.setdefault("ts", time.time())
        data = json.dumps(payload, ensure_ascii=False)
        
        # Debug log for summary_report events
        if payload.get("type") == "summary_report":
            print(f"[_publish_progress] Publishing summary_report to trace_id={tid}")
            print(f"[_publish_progress] Payload keys: {list(payload.keys())}")
            print(f"[_publish_progress] Report length: {len(payload.get('report', ''))}")
            print(f"[_publish_progress] Subscribers count: {len(_TRACE_SUBS.get(tid) or [])}")
        
        # buffer
        buf = _TRACE_BUFFER.setdefault(tid, deque(maxlen=_TRACE_BUFFER_MAX))
        buf.append(data)
        # fan-out
        subs = _TRACE_SUBS.get(tid) or []
        for q in list(subs):
            try:
                q.put_nowait(data)
            except Exception:
                try:
                    subs.remove(q)
                except Exception:
                    pass
    except Exception:
        pass

async def sse_events(trace_id: str):
    """Async generator for Server-Sent Events of a given trace id."""
    tid = str(trace_id)
    q: asyncio.Queue[str] = asyncio.Queue(maxsize=1000)
    subs = _TRACE_SUBS.setdefault(tid, [])
    subs.append(q)
    # send buffered history first
    try:
        for item in list(_TRACE_BUFFER.get(tid) or []):
            yield f"data: {item}\n\n"
    except Exception:
        pass
    try:
        while True:
            item = await q.get()
            yield f"data: {item}\n\n"
    except asyncio.CancelledError:
        pass
    finally:
        try:
            subs.remove(q)
        except Exception:
            pass

def _record_drop(reason: str, payload: Dict[str, Any], note: str = "") -> None:
    """
    Record a drop into the in-memory store if a trace id is bound.
    Never raises; diagnostics must not break planning.
    """
    try:
        tid = _CURRENT_TRACE_ID
        if tid:
            # defined later in this module; safe to call at runtime
            record_drop(tid, reason, **payload, note=note)
    except Exception:
        pass


# --------------------------------
# Helper: auto-bind trace id for drop logging
# --------------------------------
def _auto_bind_trace_if_needed(trace_id: Optional[str] = None) -> None:
    """
    Ensure _CURRENT_TRACE_ID is bound so that drop logs are recorded.
    - If trace_id is provided, bind to it (without clearing any existing buffer).
    - Else, if there is no current id, generate one and bind so we don't lose drops.
    """
    try:
        global _CURRENT_TRACE_ID
        if trace_id:
            bind_trace_id(str(trace_id))
            return
        if not _CURRENT_TRACE_ID:
            gen_id = secrets.token_hex(6)
            bind_trace_id(gen_id)
            try:
                print(f"[optimizer] trace bound (auto): {gen_id}")
            except Exception:
                pass
    except Exception:
        # diagnostics should never break main flow
        pass


# --------------------------------
# Missing trace API functions
# --------------------------------
def get_current_trace_id() -> Optional[str]:
    """Return the currently bound trace ID."""
    global _CURRENT_TRACE_ID
    return _CURRENT_TRACE_ID

def bind_trace_id(trace_id: str) -> None:
    """Bind a trace ID for drop logging."""
    global _CURRENT_TRACE_ID
    _CURRENT_TRACE_ID = trace_id

def start_drop_trace(trace_id: str) -> None:
    """Start drop trace logging with given trace ID."""
    global _CURRENT_TRACE_ID, _DROP_STORE
    _CURRENT_TRACE_ID = trace_id
    # initialize buffer for this trace (fresh run)
    try:
        _DROP_STORE[trace_id] = []
    except Exception:
        _DROP_STORE.clear()
        _DROP_STORE[trace_id] = []

def record_drop(trace_id: str, reason: str, **kwargs) -> None:
    """Record a drop event for given trace.
    Payload keys (best-effort): sku_id, lot, from_loc, to_loc, to_col, qty_cases, note.
    """
    try:
        buf = _DROP_STORE.setdefault(str(trace_id), [])
        row = {
            "reason": str(reason),
            "sku_id": None if kwargs.get("sku_id") is None else str(kwargs.get("sku_id")),
            "lot": None if kwargs.get("lot") is None else str(kwargs.get("lot")),
            "from_loc": None if kwargs.get("from_loc") is None else str(kwargs.get("from_loc")),
            "to_loc": None if kwargs.get("to_loc") is None else str(kwargs.get("to_loc")),
            "to_col": kwargs.get("to_col"),
            "qty_cases": kwargs.get("qty_cases"),
            "note": None if kwargs.get("note") is None else str(kwargs.get("note"))[:256],
        }
        buf.append(row)
        # trim if exceeding cap
        if len(buf) > _DROP_LIMIT_PER_TRACE:
            # drop oldest chunk (keep last N)
            del buf[: max(0, len(buf) - _DROP_LIMIT_PER_TRACE)]
    except Exception:
        # diagnostics should never break flow
        pass

def get_drop_summary(trace_id: str) -> List[Dict[str, Any]]:
    """Return reason-wise counts for a trace id."""
    try:
        buf = _DROP_STORE.get(str(trace_id)) or []
        cnt: Dict[str, int] = {}
        for r in buf:
            k = str(r.get("reason") or "other")
            cnt[k] = cnt.get(k, 0) + 1
        rows = [{"reason": k, "count": int(v)} for k, v in cnt.items()]
        # sort by count desc
        rows.sort(key=lambda x: (-int(x.get("count", 0)), str(x.get("reason", ""))))
        return rows
    except Exception:
        return []

def get_drop_details(trace_id: str, *, limit: int = 20, offset: int = 0) -> Tuple[List[Dict[str, Any]] | None, int]:
    """Return (rows, total) for a trace id with pagination."""
    try:
        buf = _DROP_STORE.get(str(trace_id)) or []
        total = len(buf)
        if limit is None or limit <= 0:
            limit = 20
        if offset is None or offset < 0:
            offset = 0
        end = min(total, offset + limit)
        rows = buf[offset:end]
        return rows, total
    except Exception:
        return None, 0


# -------------------------------
# Public dataclasses
# -------------------------------

@dataclass(frozen=True)
class Move:
    sku_id: str
    lot: str
    qty: int
    from_loc: str
    to_loc: str
    # Normalized date string derived from lot.
    # Format: 'YYYYMMDD' (if original lot is YYYYMM, we normalize to day '01')
    lot_date: Optional[str] = None
    # Reason for this relocation
    reason: Optional[str] = None
    # Chain group ID for linked moves (eviction chains, swaps)
    # Moves with the same chain_group_id should be executed together
    chain_group_id: Optional[str] = None
    # Execution order within a chain group (1, 2, 3...)
    # Lower numbers should be executed first
    execution_order: Optional[int] = None
    # Distance metric (optional, for reporting)
    distance: Optional[float] = None


# 入数が±帯域内か判定
def _lot_key_to_datestr8(lot_key: Optional[int]) -> Optional[str]:
    """Convert lot_key (YYYYMMDD as int) to 'YYYYMMDD' string. Unknown -> None."""
    try:
        k = int(lot_key)
    except Exception:
        return None
    # 99999999 is our sentinel for 'unknown'
    if k == UNKNOWN_LOT_KEY:
        return None
    if 20000101 <= k <= 20991231:
        return f"{k:08d}"
    return None


def _datestr8_to_lot_key(datestr8: Optional[str]) -> int:
    """Convert 'YYYYMMDD' string to lot_key (int). Invalid/None -> UNKNOWN_LOT_KEY."""
    if not datestr8:
        return UNKNOWN_LOT_KEY
    try:
        k = int(datestr8)
        if 20000101 <= k <= 20991231:
            return k
    except Exception:
        pass
    return UNKNOWN_LOT_KEY


def _validate_post_move_state(
    inv_before: pd.DataFrame,
    moves: List["Move"],
) -> Dict[str, Any]:
    """
    移動後の状態を検証し、問題点をレポートする。
    
    Returns:
        Dict with validation results:
        - fifo_violations: FIFO違反（古いロットが後から引き当てられる）
        - sku_dispersion_issues: SKU分散悪化（移動後にロケーション数が増えた）
        - validation_passed: 全体の検証結果
    """
    validation = {
        "fifo_violations": [],
        "sku_dispersion_issues": [],
        "sku_dispersion_improved": [],
        "validation_passed": True,
        "total_issues": 0,
    }
    
    if len(moves) == 0 or inv_before.empty:
        return validation
    
    # 必要な列があるかチェック
    required_cols = ["商品ID", "ロケーション", "ロット"]
    if not all(col in inv_before.columns for col in required_cols):
        return validation
    
    # lot_keyがない場合は追加
    if "lot_key" not in inv_before.columns:
        inv_before = inv_before.copy()
        inv_before["lot_key"] = inv_before["ロット"].apply(
            lambda x: _datestr8_to_lot_key(str(x)[:8]) if pd.notna(x) else UNKNOWN_LOT_KEY
        )
    
    # 移動後の状態をシミュレート
    inv_after = inv_before.copy()
    
    # 移動を適用
    move_map = {}  # (from_loc, sku_id, lot) -> to_loc
    for m in moves:
        move_map[(m.from_loc, m.sku_id, m.lot)] = m.to_loc
    
    # ロケーションを更新
    def apply_move(row):
        key = (str(row["ロケーション"]).zfill(8), str(row["商品ID"]), str(row.get("ロット", "")))
        if key in move_map:
            return move_map[key]
        return row["ロケーション"]
    
    inv_after["ロケーション"] = inv_after.apply(apply_move, axis=1)
    
    # ロケーション座標をパース
    def parse_loc(loc_str):
        loc_str = str(loc_str).zfill(8)
        if len(loc_str) == 8 and loc_str.isdigit():
            return int(loc_str[0:3]), int(loc_str[3:6]), int(loc_str[6:8])
        return 0, 0, 0
    
    inv_after["lv_after"], inv_after["col_after"], inv_after["dep_after"] = zip(
        *inv_after["ロケーション"].apply(parse_loc)
    )
    
    # === 1. FIFO違反チェック ===
    # 同一SKU内で、古いロットが後から引き当てられる位置にあるかチェック
    # 引き当て優先度: Lv小 → 列小 → 奥行小
    
    move_skus = set(m.sku_id for m in moves)
    
    for sku in move_skus:
        sku_rows = inv_after[inv_after["商品ID"].astype(str) == str(sku)].copy()
        if len(sku_rows) <= 1:
            continue
        
        # lot_keyでソート（古い順）
        sku_rows = sku_rows[sku_rows["lot_key"] != UNKNOWN_LOT_KEY]
        if len(sku_rows) <= 1:
            continue
        
        sku_rows = sku_rows.sort_values("lot_key")
        
        # 引き当て優先度キーを計算
        sku_rows["pick_priority"] = (
            sku_rows["lv_after"] * 1000000 + 
            sku_rows["col_after"] * 1000 + 
            sku_rows["dep_after"]
        )
        
        # 古いロットから順に、引き当て優先度をチェック
        prev_priority = -1
        prev_lot_key = -1
        for idx, row in sku_rows.iterrows():
            current_priority = row["pick_priority"]
            current_lot_key = row["lot_key"]
            
            if prev_priority > 0 and current_lot_key > prev_lot_key:
                # 新しいロットなのに、古いロットより優先度が高い（数値が小さい）
                if current_priority < prev_priority:
                    validation["fifo_violations"].append({
                        "sku_id": sku,
                        "newer_lot_key": current_lot_key,
                        "newer_loc": row["ロケーション"],
                        "newer_priority": current_priority,
                        "older_lot_key": prev_lot_key,
                        "issue": "新しいロットが古いロットより先に引き当てられる位置にある",
                    })
            
            prev_priority = current_priority
            prev_lot_key = current_lot_key
    
    # === 2. SKU分散チェック ===
    # 移動前後でロケーション数を比較
    
    for sku in move_skus:
        before_locs = set(
            inv_before[inv_before["商品ID"].astype(str) == str(sku)]["ロケーション"].astype(str)
        )
        after_locs = set(
            inv_after[inv_after["商品ID"].astype(str) == str(sku)]["ロケーション"].astype(str)
        )
        
        before_count = len(before_locs)
        after_count = len(after_locs)
        
        if after_count > before_count:
            validation["sku_dispersion_issues"].append({
                "sku_id": sku,
                "before_loc_count": before_count,
                "after_loc_count": after_count,
                "before_locs": sorted(list(before_locs))[:5],
                "after_locs": sorted(list(after_locs))[:5],
                "issue": f"SKUが分散 ({before_count}→{after_count}ロケ)",
            })
        elif after_count < before_count:
            validation["sku_dispersion_improved"].append({
                "sku_id": sku,
                "before_loc_count": before_count,
                "after_loc_count": after_count,
                "improvement": f"SKUが集約 ({before_count}→{after_count}ロケ)",
            })
    
    # === 総合判定 ===
    validation["total_issues"] = len(validation["fifo_violations"]) + len(validation["sku_dispersion_issues"])
    validation["validation_passed"] = validation["total_issues"] == 0
    
    return validation


def _generate_relocation_summary(
    inv_before: pd.DataFrame,
    moves: List["Move"],
    rejected_count: int = 0,
    pass_stats: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Generate comprehensive relocation summary report.
    
    Returns a dict with:
    - basic_stats: total moves, SKUs affected, locations involved
    - sku_consolidation: top SKUs before/after location counts
    - hard_rules: lot mixing check, allocated inventory exclusion
    - quality_breakdown: moves by quality category
    - recommendations: list of improvement suggestions
    - pass_stats: pass-wise move breakdown (Pass-0, Pass-1, Pass-2, Pass-3)
    """
    print(f"[_generate_relocation_summary] Starting: moves={len(moves)}, rejected={rejected_count}")
    
    summary = {
        "total_planned": len(moves) + rejected_count,
        "pass_stats": pass_stats or {},
        "total_rejected": rejected_count,
        "total_accepted": len(moves),
    }
    
    if len(moves) == 0:
        summary["message"] = "移動案なし"
        print(f"[_generate_relocation_summary] No moves, returning early")
        return summary
    
    # Extract move details
    move_skus = set(m.sku_id for m in moves)
    move_from_locs = set(m.from_loc for m in moves)
    move_to_locs = set(m.to_loc for m in moves)
    total_cases = sum(m.qty for m in moves)
    
    summary["affected_skus"] = len(move_skus)
    summary["from_locations"] = len(move_from_locs)
    summary["to_locations"] = len(move_to_locs)
    summary["total_cases"] = total_cases
    
    # SKU consolidation analysis (top 5)
    sku_before_locs = {}
    sku_after_locs = {}
    sku_before_locs_detail = {}  # 移動前のロケーション一覧
    sku_after_locs_detail = {}   # 移動後のロケーション一覧
    
    for sku in list(move_skus)[:5]:  # Top 5 for performance
        if "商品ID" in inv_before.columns and "ロケーション" in inv_before.columns:
            # Before: unique locations for this SKU in original inventory
            before_locs_set = set(inv_before[inv_before["商品ID"].astype(str) == str(sku)]["ロケーション"])
            sku_before_locs[sku] = len(before_locs_set)
            sku_before_locs_detail[sku] = sorted(list(before_locs_set))  # ソート済みリスト
            
            # After: original locations - from_locs + to_locs
            sku_moves = [m for m in moves if m.sku_id == sku]
            from_locs_set = set(m.from_loc for m in sku_moves)
            to_locs_set = set(m.to_loc for m in sku_moves)
            
            # After状態 = 移動前のロケ - 移動元ロケ + 移動先ロケ
            after_locs_set = (before_locs_set - from_locs_set) | to_locs_set
            sku_after_locs[sku] = len(after_locs_set)
            sku_after_locs_detail[sku] = sorted(list(after_locs_set))  # ソート済みリスト
    
    summary["sku_consolidation"] = {
        "before": sku_before_locs,
        "after": sku_after_locs,
        "before_detail": sku_before_locs_detail,
        "after_detail": sku_after_locs_detail,
    }
    
    # Lot mixing check (same loc + same SKU + different lots)
    lot_mixing_issues = []
    loc_sku_lots: Dict[Tuple[str, str], Set[str]] = {}
    
    for m in moves:
        key = (m.to_loc, m.sku_id)
        if key not in loc_sku_lots:
            loc_sku_lots[key] = set()
        if m.lot:
            loc_sku_lots[key].add(m.lot)
    
    for (loc, sku), lots in loc_sku_lots.items():
        if len(lots) > 1:
            lot_mixing_issues.append({
                "location": loc,
                "sku": sku,
                "lot_count": len(lots),
                "lots": list(lots)[:3],  # First 3 for brevity
            })
    
    summary["lot_mixing_issues"] = len(lot_mixing_issues)
    summary["lot_mixing_details"] = lot_mixing_issues[:5]  # First 5
    
    # Analyze move reasons (group by reason and count)
    reason_counts: Dict[str, int] = {}
    for m in moves:
        reason = m.reason if m.reason else "理由未設定"
        # Extract first action (before →)
        main_action = reason.split("→")[0].strip() if "→" in reason else reason
        reason_counts[main_action] = reason_counts.get(main_action, 0) + 1
    
    # Sort by count descending
    sorted_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
    summary["reason_breakdown"] = dict(sorted_reasons)
    
    # Quality analysis (if available)
    if "品質区分" in inv_before.columns:
        quality_counts = inv_before["品質区分"].value_counts().to_dict()
        summary["quality_breakdown"] = {
            "good_items": quality_counts.get(1, 0),  # 品質区分=1 is 良品
            "total_items": len(inv_before),
        }
    
    # === New evaluation metrics ===
    
    # 1. Lot moves to pick locations (level 1-2) - Pass-1 FIFO behavior
    # Count moves from storage/other levels to pick levels (取り口ロケ)
    old_lot_to_pick = 0
    pick_levels_set = {1, 2}  # 取り口レベル
    
    for m in moves:
        try:
            # Parse from_loc and to_loc to check level changes
            from_str = str(m.from_loc)
            to_str = str(m.to_loc)
            
            if len(from_str) == 8 and from_str.isdigit() and len(to_str) == 8 and to_str.isdigit():
                from_level = int(from_str[0:3])
                to_level = int(to_str[0:3])
                
                # Count moves TO pick levels (1-2) FROM non-pick levels (3+)
                # This indicates Pass-1 FIFO: moving older lots to pick locations
                if to_level in pick_levels_set and from_level not in pick_levels_set:
                    old_lot_to_pick += 1
        except Exception:
            pass
    
    summary["old_lot_to_pick_count"] = old_lot_to_pick
    
    # 2. High-velocity items moved to hotspot (center columns, low depth)
    hotspot_moves = 0
    high_velocity_threshold = None
    
    # Calculate velocity from inventory (出荷実績 or similar)
    # For now, use a heuristic: if SKU appears in many locations, assume high velocity
    # Better: use actual shipment data if available in inv_before
    sku_velocity = {}
    if "出荷数" in inv_before.columns or "出荷ケース数" in inv_before.columns:
        velocity_col = "出荷数" if "出荷数" in inv_before.columns else "出荷ケース数"
        sku_velocity = inv_before.groupby("商品ID")[velocity_col].sum().to_dict()
        if sku_velocity:
            velocities = sorted(sku_velocity.values(), reverse=True)
            # Top 20% is "high velocity"
            if len(velocities) > 5:
                high_velocity_threshold = velocities[len(velocities) // 5]
    
    for m in moves:
        try:
            to_str = str(m.to_loc)
            if len(to_str) == 8 and to_str.isdigit():
                to_level = int(to_str[0:3])
                to_col = int(to_str[3:6])
                to_dep = int(to_str[6:8])
                
                # Hotspot criteria:
                # - Level 1 (pick location)
                # - Column near center (山形の中心: columns 15-30 are typically center)
                # - Low depth (1-3 is front, easy to access)
                is_hotspot = (to_level == 1) and (15 <= to_col <= 30) and (to_dep <= 3)
                
                # Check if SKU is high velocity
                is_high_velocity = False
                if high_velocity_threshold and m.sku_id in sku_velocity:
                    is_high_velocity = sku_velocity[m.sku_id] >= high_velocity_threshold
                elif not high_velocity_threshold:
                    # Fallback: check if SKU appears in many locations (proxy for high velocity)
                    sku_loc_count = inv_before[inv_before["商品ID"].astype(str) == str(m.sku_id)]["ロケーション"].nunique()
                    is_high_velocity = sku_loc_count >= 5
                
                if is_hotspot and is_high_velocity:
                    hotspot_moves += 1
        except Exception:
            pass
    
    summary["hotspot_moves_count"] = hotspot_moves
    summary["hotspot_criteria"] = "レベル1, 列15-30(中心), 奥行1-3(手前)"
    
    # Recommendations
    recommendations = []
    
    move_rate = len(moves) / len(inv_before) * 100 if len(inv_before) > 0 else 0
    if move_rate < 2:
        recommendations.append("移動率が低い（<2%）- より積極的な最適化が可能かもしれません")
    
    if lot_mixing_issues:
        recommendations.append(f"⚠️ ロット混在が{len(lot_mixing_issues)}件検出されました - ハードルール違反の可能性")
    
    if len(move_skus) < 10 and len(moves) > 50:
        recommendations.append("少数のSKUに集中した移動 - 在庫集約が効果的に機能しています")
    
    if old_lot_to_pick > 0:
        recommendations.append(f"✅ 古いロット{old_lot_to_pick}件を取り口ロケ(Lv1)へ移動 - FIFO回転促進")
    
    if hotspot_moves > 0:
        recommendations.append(f"✅ 出荷頻度の高いSKU {hotspot_moves}件をホットスポットへ配置 - ピッキング効率向上")
    
    summary["recommendations"] = recommendations
    summary["move_rate_percent"] = round(move_rate, 2)
    
    # === 移動後検証レポート ===
    try:
        validation = _validate_post_move_state(inv_before, moves)
        summary["post_move_validation"] = validation
        
        if validation["fifo_violations"]:
            recommendations.append(
                f"⚠️ FIFO違反 {len(validation['fifo_violations'])}件検出 - "
                "新しいロットが古いロットより先に引き当てられる位置にあります"
            )
        if validation["sku_dispersion_issues"]:
            recommendations.append(
                f"⚠️ SKU分散悪化 {len(validation['sku_dispersion_issues'])}件 - "
                "移動後にロケーション数が増えたSKUがあります"
            )
        if validation["sku_dispersion_improved"]:
            recommendations.append(
                f"✅ SKU集約改善 {len(validation['sku_dispersion_improved'])}件 - "
                "移動後にロケーション数が減ったSKUがあります"
            )
        
        # 総合判定をサマリーに追加
        summary["validation_passed"] = validation["validation_passed"]
        summary["validation_issues_count"] = validation["total_issues"]
        
    except Exception as e:
        print(f"[_generate_relocation_summary] Validation failed: {e}")
        summary["post_move_validation"] = {"error": str(e)}
    
    print(f"[_generate_relocation_summary] Completed: keys={list(summary.keys())}")
    return summary


# -------------------------------
@dataclass
class OptimizerConfig:
    # 0 or None -> unlimited moves
    max_moves: Optional[int] = None
    fill_rate: float = DEFAULT_FILL_RATE
    # 入数の許容レンジ（±比率）
    pack_tolerance_ratio: float = 0.10
    # 既存マッピングの“ほどよい”維持のための支配的入数のしきい値
    preserve_pack_mapping_threshold: float = 0.70
    # 入数が許容レンジから外れるときのペナルティ（大きめ）
    pack_mismatch_penalty: float = 400.0
    # 許容内の微差に対する小さなペナルティ係数（diff%×この値）
    pack_within_tolerance_penalty: float = 12.0
    # 同一SKUを同じ列に集約できる場合のボーナス（ペナルティを減らす）
    same_sku_same_column_bonus: float = 20.0
    # 元と同じ列に置ける場合の小さなボーナス
    prefer_same_column_bonus: float = 5.0
    # SKUを新しい列に分散させる小さなペナルティ
    split_sku_new_column_penalty: float = 5.0
    # 同一SKU・同一ロットを同じ棚（列）に揃えるソフト嗜好（ボーナス）
    same_lot_same_column_bonus: float = 10.0
    # 取りやすさ（LV/COL/DEP を小さいほど優先）にかける重み（小さいほど他要素を優先）
    ease_weight: float = 0.0001
    # --- Column band preference (近・遠の帯嗜好) ---
    pack_low_max: int = 12                      # 少ない(=重い)とみなす入数上限
    pack_high_min: int = 50                     # 多い(=軽い)とみなす入数下限
    near_cols: tuple[int, ...] = tuple(range(35, 42))  # 35–41 を「近い帯」として優先
    far_cols:  tuple[int, ...] = tuple(range(1, 12))   #  1–11 を「遠い帯」として優先
    band_pref_weight: float = 20.0               # 帯嗜好の効き具合（負がボーナスになる）
    promo_quality_keywords: tuple[str, ...] = ("販促資材", "販促", "什器", "資材")
    # --- Pick/Storage levels and area zoning ---
    pick_levels: tuple[int, ...] = (1, 2)           # 取口=1-2段
    storage_levels: tuple[int, ...] = (3, 4)        # 保管=3-4段
    # --- Pass-0: Area (column band) rebalance control ---
    enable_pass0_area_rebalance: bool = True
    # 入数<=2 の緩和を Pass-0 でも尊重（True=帯外でも対象外とする）
    pass0_respect_smallpack_relax: bool = True
    # Pass-0 で候補とするターゲット段（優先順: 現在段→他段）
    pass0_target_levels: tuple[int, ...] = (1, 2, 3, 4)
    # Pass-0 の候補ロケーション数上限（パフォーマンス最適化）
    pass0_max_candidates_per_item: int = 50
    # --- Pass-1: Pick/Storage balance control ---
    # Pass-1 swap control
    enable_pass1_swap: bool = True
    pass1_swap_budget_per_group: int = 2  # 同一SKU×同一列のグループあたり最大スワップ回数
    # --- Pass-3: AI-driven optimization control ---
    enable_pass3_ai_optimize: bool = True  # AI最適化Passを有効化（デフォルト有効）

    # 入数帯ごとの列レンジ（エリア分け）
    # 1～11: 遠距離、12～23と35～41: 中距離、24～34: 近距離
    small_pack_cols: tuple[int, ...] = tuple(range(24, 35))   # 近距離（24～34列）
    medium_pack_cols: tuple[int, ...] = tuple(list(range(12, 24)) + list(range(35, 42)))  # 中距離（12～23 + 35～41列）
    large_pack_cols: tuple[int, ...] = tuple(range(1, 12))    # 遠距離（1～11列）

    # エリアをハード適用するか（ミックス枠で例外許容）
    strict_pack_area: bool = True

    # 出荷実績/HOT列ヒント関連（将来拡張用）
    ship_window_days: int = 180
    hot_columns_per_area: int = 4
    hot_hint_bonus: float = 25.0
    # 各列に許容する“ミックス列”枠（±10%を外れても許す枠数）
    mix_slots_per_col: int = 1
    # ミックス枠を消費する場合の微小なコスト
    mix_usage_penalty: float = 1.0
    # AI優先列配置のボーナス
    ai_preferred_column_bonus: float = 25.0
    # 絶対制約スイッチ類（AI案の最終ゲートなどで使用）
    hard_cap: bool = False            # 容量をハードに守る
    hard_fifo: bool = False           # 「古いロットが下」ルールをハードに守る
    strict_pack: Optional[str] = None # "A", "B", "A+B" を指定可（Noneは従来通り）
    exclude_oversize: bool = False    # 1ケースが棚容積上限を超えるSKUを除外
    # --- Time limit for main loop (メインループの時間制限) ---
    loop_time_limit: float = 1800.0   # seconds (30 minutes) - chain_depth=1で全行処理に必要
    # --- Eviction chain (bounded multi-step relocation) ---
    # chain_depth=1: 浅い連鎖のみ許可（パフォーマンスと効果のバランス）
    chain_depth: int = 1            # 0=disabled; 1=shallow; 2=deeper chains
    # 既定で控えめに有効化（性能と効果のバランス）
    eviction_budget: int = 50        # max number of eviction (auxiliary) moves
    touch_budget: int = 100          # max number of distinct locations we can touch
    buffer_slots: int = 0           # reserved empty-ish slots for temporary staging (0=disabled)
    # Allow same-level moves if (column,depth) strictly improve ease
    allow_same_level_ease_improve: bool = True
    # Diagnostics / pipeline
    trace_id: Optional[str] = None
    # If True, run a hard-gate pass here to both finalize moves and record drop reasons
    auto_enforce: bool = True
    # --- Depth preference (奥行きの優先度) ---
    # 'front'  = 従来通り「浅いほど良い」
    # 'center' = 列ごとの奥行き数に応じて“中心に近いほど良い”（山形）
    depth_preference: str = "front"
    # 山形評価の強さ（距離×この重みをスコアに加算）
    center_depth_weight: float = 1.0
    
    # --- Performance optimization (並列化・高速化) ---
    enable_parallel: bool = False      # 並列処理を無効化（安定性優先）
    parallel_workers: int = 4          # 並列ワーカー数（CPUコア数に応じて調整）
    parallel_batch_size: int = 100     # バッチサイズ
    enable_candidate_limit: bool = True # 候補ロケーション数の制限
    max_candidates_per_level: int = 50  # レベルごとの最大候補数（0=無制限）
    
    # --- SKU per-location limit (作業負荷軽減) ---
    # 1SKUあたりの移動元ロケーション数の上限（None=無制限）
    # 最も古いロットのロケーションを優先的に選択
    max_source_locs_per_sku: Optional[int] = 2
    
# -------------------------------
# Additional helpers for hard constraints and pack clustering
# -------------------------------

def _hard_fifo_violation_simple(inv: pd.DataFrame, sku: str, lot_key: int, target_level: int, tcol: int) -> bool:
    """(sku, col) に対して、target_level へ配置すると
    『古いロットほど低段（数値が小さい）／新しいロットほど高段（数値が大きい）』に
    違反するかを厳密判定する（同一SKU×同一列 内のみ）。
    """
    need_cols = {"lv", "lot_key", "商品ID", "col"}
    if not need_cols.issubset(inv.columns):
        return False
    same = inv[(inv["商品ID"].astype(str) == str(sku)) & (inv["col"] == int(tcol))]
    if same.empty:
        return False

    # 新しいロットは、本行より“高い段”（=数値が大きい）にいなければならない
    newer = same[same["lot_key"] > lot_key]
    if not newer.empty:
        min_newer_lv = int(newer["lv"].min())
        if min_newer_lv < target_level:
            return True

    # 古いロットは、本行より“低い段”（=数値が小さい）にいなければならない
    older = same[same["lot_key"] < lot_key]
    if not older.empty:
        max_older_lv = int(older["lv"].max())
        if max_older_lv > target_level:
            return True

    return False

def _compute_rep_pack_by_col_for_inv(inv: pd.DataFrame, pack_map: Optional[pd.Series], preserve_thr: float) -> dict[int, float]:
    """在庫から列ごとの代表入数(支配的入数またはメディアン)を計算して返す。"""
    rep_pack_by_col: dict[int, float] = {}
    if pack_map is None or inv.empty or "col" not in inv.columns:
        return rep_pack_by_col
    tmp = inv.copy()
    tmp["pack_est"] = tmp["商品ID"].astype(str).map(pack_map)
    tmp = tmp.dropna(subset=["pack_est"])
    if tmp.empty:
        return rep_pack_by_col
    # weight: prefer qty_cases_move; else ケース; else default 1.0 per row
    if "qty_cases_move" in tmp.columns:
        _w = pd.to_numeric(tmp["qty_cases_move"], errors="coerce")
    elif "ケース" in tmp.columns:
        _w = pd.to_numeric(tmp["ケース"], errors="coerce")
    else:
        _w = pd.Series(1.0, index=tmp.index)
    tmp["weight"] = _w.fillna(1.0).astype(float)
    by_col_pack = tmp.groupby(["col", "pack_est"], dropna=False)["weight"].sum().reset_index()
    total_by_col = by_col_pack.groupby("col")["weight"].sum()
    top_by_col = by_col_pack.sort_values(["col", "weight"], ascending=[True, False]).groupby("col").head(1)
    # 支配的入数
    for _, r in top_by_col.iterrows():
        c = int(r["col"]); top_w = float(r["weight"]); tot = float(total_by_col.get(c, 0.0) or 0.0)
        share = top_w / tot if tot > 0 else 0.0
        if share >= preserve_thr:
            rep_pack_by_col[c] = float(r["pack_est"])
    # 補完としてメディアン
    tmp_med = tmp.copy()
    tmp_med["pack_est"] = pd.to_numeric(tmp_med["pack_est"], errors="coerce")
    med_by_col = tmp_med.groupby("col", dropna=True)["pack_est"].median()
    for c, med in med_by_col.items():
        c = int(c)
        if c not in rep_pack_by_col and pd.notna(med):
            rep_pack_by_col[c] = float(med)
    return rep_pack_by_col


# ---------------------------------------------------------------------------
# Post-processing: resolve move dependencies (to_loc/from_loc conflicts)
# ---------------------------------------------------------------------------

def _resolve_move_dependencies(moves: List[Move]) -> List[Move]:
    """Resolve execution order dependencies between moves.

    If Move A wants to place items at location X, but Move B needs to
    remove items FROM location X first, then B must execute before A.
    This function:
      1. Detects to_loc/from_loc conflicts
      2. Groups dependent moves under the same chain_group_id
      3. Assigns correct execution_order (evacuate first, then place)
      4. Reorders the list so dependencies come before dependents

    Returns a new list of Move objects with updated chain_group_id and
    execution_order where dependencies exist.
    """
    if not moves:
        return moves

    from collections import defaultdict

    # Build index: to_loc -> list of move indices that want to place there
    to_loc_index: Dict[str, List[int]] = defaultdict(list)
    # Build index: from_loc -> list of move indices that evacuate from there
    from_loc_index: Dict[str, List[int]] = defaultdict(list)

    for i, m in enumerate(moves):
        to_loc_index[m.to_loc].append(i)
        from_loc_index[m.from_loc].append(i)

    # Find dependency pairs: move A wants to_loc X, move B from_loc X
    # B must execute before A
    # dep_graph[i] = set of indices that must execute BEFORE move i
    dep_graph: Dict[int, set] = defaultdict(set)
    # reverse: who depends on me
    reverse_deps: Dict[int, set] = defaultdict(set)

    for loc, placers in to_loc_index.items():
        evacuators = from_loc_index.get(loc, [])
        for placer_idx in placers:
            for evac_idx in evacuators:
                if placer_idx != evac_idx:
                    # evac must happen before placer
                    dep_graph[placer_idx].add(evac_idx)
                    reverse_deps[evac_idx].add(placer_idx)

    if not dep_graph:
        # No conflicts found
        return moves

    # Group connected components (transitive dependencies)
    visited = set()
    groups: List[set] = []

    def _dfs(node: int, group: set):
        if node in visited:
            return
        visited.add(node)
        group.add(node)
        for dep in dep_graph.get(node, set()):
            _dfs(dep, group)
        for dependent in reverse_deps.get(node, set()):
            _dfs(dependent, group)

    all_dep_nodes = set(dep_graph.keys()) | set(reverse_deps.keys())
    for node in all_dep_nodes:
        if node not in visited:
            group: set = set()
            _dfs(node, group)
            groups.append(group)

    # For each group, assign shared chain_group_id and topological order
    result = list(moves)  # copy
    dep_count = 0

    for group_indices in groups:
        if len(group_indices) < 2:
            continue

        group_chain_id = f"dep_{secrets.token_hex(6)}"

        # Topological sort within the group
        # In-degree based approach
        in_degree: Dict[int, int] = {idx: 0 for idx in group_indices}
        for idx in group_indices:
            for dep in dep_graph.get(idx, set()):
                if dep in group_indices:
                    in_degree[idx] = in_degree.get(idx, 0) + 1

        queue = [idx for idx in group_indices if in_degree[idx] == 0]
        topo_order = []
        while queue:
            # Sort queue for deterministic output
            queue.sort(key=lambda x: x)
            node = queue.pop(0)
            topo_order.append(node)
            for dependent in reverse_deps.get(node, set()):
                if dependent in group_indices:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        # Handle cycles (shouldn't happen, but safety)
        remaining = group_indices - set(topo_order)
        topo_order.extend(sorted(remaining))

        # Assign chain_group_id and execution_order
        for exec_order, idx in enumerate(topo_order, start=1):
            old_move = result[idx]
            result[idx] = Move(
                sku_id=old_move.sku_id,
                lot=old_move.lot,
                qty=old_move.qty,
                from_loc=old_move.from_loc,
                to_loc=old_move.to_loc,
                lot_date=old_move.lot_date,
                reason=old_move.reason,
                chain_group_id=group_chain_id,
                execution_order=exec_order,
                distance=old_move.distance,
            )
            dep_count += 1

    # Reorder: moves with dependencies should appear in execution order
    # Build final list: non-dependent moves keep original order,
    # dependent groups are inserted at the position of the first member
    dep_indices = set()
    for g in groups:
        if len(g) >= 2:
            dep_indices.update(g)

    # Build ordered groups map: first_index -> [ordered indices]
    group_first: Dict[int, List[int]] = {}
    for group_indices in groups:
        if len(group_indices) < 2:
            continue
        # Sort by execution_order
        sorted_group = sorted(group_indices, key=lambda i: result[i].execution_order or 0)
        first_idx = min(group_indices)
        group_first[first_idx] = sorted_group

    final_list: List[Move] = []
    inserted_groups: set = set()
    for i, m in enumerate(result):
        if i in dep_indices:
            # Check if this is the first member of a group
            if i in group_first and i not in inserted_groups:
                # Insert the entire group in order
                for idx in group_first[i]:
                    final_list.append(result[idx])
                inserted_groups.add(i)
            # Skip individual members (they're added as part of the group)
            continue
        else:
            final_list.append(m)

    if dep_count > 0:
        print(f"[optimizer] Dependency resolution: {dep_count} moves in {len(groups)} dependency groups")

    return final_list


def enforce_constraints(
    sku_master: pd.DataFrame,
    inventory: pd.DataFrame,
    moves: List[Move],
    *,
    cfg: OptimizerConfig | None = None,
    loc_master: Optional[pd.DataFrame] = None,
) -> List[Move]:
    """
    外部(例: AIドラフト)で生成された移動候補に対し、
    容量・FIFO・入数まとまり(A)・オーバーサイズ等の『絶対ゲート』を順次適用し、
    採択可能な移動のみを返す。
    - 容量は逐次加算で判定 (to_locごとの used + delta)
    - FIFOは (SKU, 列) 単位で『古いロットほど低段』の厳密順序をチェック
    - 入数まとまりは "A"（±帯）をハードに適用（"B"はスコア系なのでここではハード適用しない）
    - ロケーション・マスタが渡されれば、その capacity_m3 と can_receive を使用
    """
    cfg = cfg or OptimizerConfig()
    _auto_bind_trace_if_needed(getattr(cfg, "trace_id", None))
    tid = get_current_trace_id()
    # debug: initialize per-call tracker
    _dbg_reset(_CURRENT_TRACE_ID)
    _dbg_note_planned(len(moves))
    try:
        _publish_progress(tid, {
            "type": "enforce_start", "total": len(moves),
            "message": f"制約検証開始: {len(moves)}件の移動案をチェック"
        })
    except Exception:
        pass
    base_cap = _capacity_limit(getattr(cfg, "fill_rate", None))

    # --- Per-location capacities / allow list from location_master (optional)
    cap_by_loc: Dict[str, float] = {}
    can_receive_set: Set[str] = set()
    if loc_master is not None and not loc_master.empty:
        cap_by_loc, can_receive_set = _cap_map_from_master(loc_master, getattr(cfg, "fill_rate", DEFAULT_FILL_RATE))

    # --- SKU -> 1ケース容積(m³)
    sku_vol_map = _build_carton_volume_map(sku_master)

    # 入数マップ
    pack_map = None
    if "入数" in sku_master.columns:
        key = "sku_id" if "sku_id" in sku_master.columns else "商品ID"
        pack_map = sku_master.set_index(key)["入数"].astype(float)

    # 在庫の lot_key / 位置 を整備
    inv = inventory.copy()
    _require_cols(inv, ["ロケーション", "商品ID", "ロット"], "inventory")
    inv["lot_key"] = inv["ロット"].map(_parse_lot_date_key)
    # ベクトル化: apply不使用で高速化
    loc_str = inv["ロケーション"].astype(str).str.zfill(8)
    inv["lv"] = pd.to_numeric(loc_str.str[0:3], errors='coerce').fillna(0).astype(int)
    inv["col"] = pd.to_numeric(loc_str.str[3:6], errors='coerce').fillna(0).astype(int)
    inv["dep"] = pd.to_numeric(loc_str.str[6:8], errors='coerce').fillna(0).astype(int)
    # 山形（中心優先）用の列→(max_dep, center) マップ
    depths_by_col_calc: Dict[int, Tuple[int, float]] | None = None
    try:
        if getattr(cfg, "depth_preference", "front") == "center":
            depths_by_col_calc = {}
            if not inv.empty:
                max_dep_by_col = inv.groupby("col")["dep"].max()
                for c, m in max_dep_by_col.items():
                    c_int = int(c)
                    max_dep = int(m)
                    if max_dep <= 0:
                        continue
                    if max_dep % 2 == 1:
                        center = (max_dep + 1) / 2.0
                    else:
                        center = (max_dep / 2.0 + (max_dep / 2.0 + 1.0)) / 2.0
                    depths_by_col_calc[c_int] = (max_dep, float(center))
    except Exception:
        depths_by_col_calc = None

    # --- 列ごとの最大奥行きと中心値（山形用）
    depths_by_col_calc: Dict[int, Tuple[int, float]] | None = None
    try:
        if getattr(cfg, "depth_preference", "front") == "center":
            depths_by_col_calc = {}
            if not inv.empty:
                max_dep_by_col = inv.groupby("col")["dep"].max()
                for c, m in max_dep_by_col.items():
                    c_int = int(c)
                    max_dep = int(m)
                    if max_dep <= 0:
                        continue
                    if max_dep % 2 == 1:
                        center = (max_dep + 1) / 2.0
                    else:
                        center = (max_dep / 2.0 + (max_dep / 2.0 + 1.0)) / 2.0
                    depths_by_col_calc[c_int] = (max_dep, float(center))
    except Exception:
        depths_by_col_calc = None

    # --- 列ごとの最大奥行きと中心値（山形用）
    depths_by_col_calc: Dict[int, Tuple[int, float]] | None = None
    try:
        if getattr(cfg, "depth_preference", "front") == "center":
            depths_by_col_calc = {}
            if not inv.empty:
                max_dep_by_col = inv.groupby("col")["dep"].max()
                for c, m in max_dep_by_col.items():
                    c_int = int(c)
                    max_dep = int(m)
                    if max_dep <= 0:
                        continue
                    if max_dep % 2 == 1:
                        center = (max_dep + 1) / 2.0
                    else:
                        center = (max_dep / 2.0 + (max_dep / 2.0 + 1.0)) / 2.0
                    depths_by_col_calc[c_int] = (max_dep, float(center))
    except Exception:
        depths_by_col_calc = None

    # 代表入数(列)
    rep_pack_by_col = _compute_rep_pack_by_col_for_inv(inv, pack_map, getattr(cfg, "preserve_pack_mapping_threshold", 0.7))

    # ロケ別使用量（m³）
    inv_key = inv["商品ID"].astype(str)
    vol_each_series = inv_key.map(sku_vol_map).fillna(0.0)
    # Determine per-row case quantity for capacity use; fall back to 1.0 case when unknown
    if "qty_cases_move" in inv.columns:
        qty_cases_float = pd.to_numeric(inv["qty_cases_move"], errors="coerce")
    elif "ケース" in inv.columns:
        qty_cases_float = pd.to_numeric(inv["ケース"], errors="coerce")
    else:
        qty_cases_float = pd.Series(1.0, index=inv.index)  # default 1 case each if unknown
    qty_cases_float = qty_cases_float.fillna(1.0).astype(float)
    inv["volume_total"] = vol_each_series * qty_cases_float
    shelf_usage = inv.groupby("ロケーション")["volume_total"].sum().to_dict()

    # Ensure empty candidate locations exist in shelf_usage when loc_master provided
    if cap_by_loc:
        for loc in cap_by_loc.keys():
            shelf_usage.setdefault(loc, 0.0)

    # Oversize early exclusion uses the *largest* capacity if master is present; else base_cap
    max_cap = max(cap_by_loc.values()) if cap_by_loc else base_cap

    accepted: List[Move] = []
    sim_inv = inv[["商品ID", "lot_key", "lv", "col", "dep", "ロケーション"]].copy()

    processed = 0
    step = max(1, len(moves) // 20) if moves else 1
    for m in moves:
        sku = str(m.sku_id)
        lot_key = _parse_lot_date_key(m.lot)
        if lot_key == UNKNOWN_LOT_KEY:
            continue
        to_loc = str(m.to_loc)
        from_loc = str(m.from_loc)
        tlv, tcol, tdep = _parse_loc8(to_loc)
        add_each = float(sku_vol_map.get(sku, 0.0) or 0.0)
        add_vol = add_each * float(m.qty)
        _mctx = {
            "sku_id": sku,
            "lot": m.lot,
            "from": from_loc,
            "to_loc": to_loc,
            "to_col": tcol,
            "qty_cases": m.qty,
        }

        # 0) oversize (compare against global max capacity if using master)
        if cfg.exclude_oversize and add_each > max_cap:
            _dbg_reject("oversize", _mctx, note=f"each={add_each:.3f}m3 > max_cap={max_cap:.3f}")
            continue

        # 1) can_receive gate (when master provided)
        if can_receive_set and to_loc not in can_receive_set:
            _dbg_reject("forbidden", _mctx, note="destination cannot receive")
            continue

        # 2) capacity gate (per-location if provided)
        used_to = float(shelf_usage.get(to_loc, 0.0))
        limit_to = cap_by_loc.get(to_loc, base_cap) if cap_by_loc else base_cap
        if cfg.hard_cap and (used_to + add_vol > limit_to):
            _dbg_reject("capacity", _mctx, note=f"used={used_to:.3f}, limit={limit_to:.3f}, add={add_vol:.3f}")
            continue

        # 3) FIFO strict (same column)
        if cfg.hard_fifo:
            if _hard_fifo_violation_simple(sim_inv, sku, lot_key, tlv, tcol):
                _dbg_reject("fifo", _mctx, note=f"lot_key={lot_key}, target_lv={tlv}, col={tcol}")
                continue

        # 4) pack band (A)
        if cfg.strict_pack and ("A" in str(cfg.strict_pack)):
            if pack_map is not None:
                pack_est = float(pack_map.get(sku, float("nan")))
                rep = rep_pack_by_col.get(tcol)
                if rep and not _within_band(pack_est, rep, getattr(cfg, "pack_tolerance_ratio", 0.10)):
                    _dbg_reject("pack_band", _mctx, note=f"pack={pack_est}, rep={rep}, tol={getattr(cfg,'pack_tolerance_ratio',0.10)}")
                    continue

        lot_date_str = _lot_key_to_datestr8(lot_key)
        
        # Preserve original reason from the move plan
        # Only generate a default reason if none was provided
        if m.reason:
            move_reason = m.reason
        else:
            # Build default reason if none exists
            from_lv, from_col, from_dep = _parse_loc8(from_loc)
            
            reasons = []
            if tcol > from_col:
                reasons.append(f"入口に近づく(列{from_col}→{tcol})")
            elif tcol < from_col:
                reasons.append(f"適正エリアへ移動(列{from_col}→{tcol})")
            else:
                reasons.append(f"列{tcol}内で最適化")
            
            improvements = []
            if tcol > from_col:
                improvements.append("歩行距離短縮")
            improvements.append("エリア内在庫バランス改善")
            
            move_reason = " | ".join(reasons) + " → " + "、".join(improvements)
        
        accepted.append(Move(
            sku_id=sku, 
            lot=m.lot, 
            qty=m.qty, 
            from_loc=str(from_loc).zfill(8), 
            to_loc=str(to_loc).zfill(8), 
            lot_date=lot_date_str,
            reason=move_reason,
            chain_group_id=getattr(m, 'chain_group_id', None),
            execution_order=getattr(m, 'execution_order', None),
            distance=getattr(m, 'distance', None),
        ))
        shelf_usage[to_loc] = shelf_usage.get(to_loc, 0.0) + add_vol
        shelf_usage[from_loc] = max(0.0, shelf_usage.get(from_loc, 0.0) - add_vol)

        new_row = {"商品ID": sku, "lot_key": lot_key, "lv": tlv, "col": tcol, "dep": tdep, "ロケーション": to_loc}
        sim_inv = pd.concat([sim_inv, pd.DataFrame([new_row])], ignore_index=True)

        # periodic progress
        processed += 1
        if processed % step == 0:
            try:
                progress_pct = int(100 * processed / len(moves)) if len(moves) > 0 else 0
                _publish_progress(tid, {
                    "type": "enforce_progress", 
                    "processed": processed, "total": len(moves), "accepted": len(accepted),
                    "message": f"制約チェック中: {processed}/{len(moves)}件 ({progress_pct}%) - 承認{len(accepted)}件"
                })
            except Exception:
                pass

    # debug: finalize accepted count
    _dbg_note_accepted(len(accepted))
    try:
        rej = _last_relocation_debug.get("rejections", {})
        if rej:
            breakdown = ", ".join(f"{k}={int(v)}" for k, v in rej.items())
            print(f"[optimizer] constraints accepted={len(accepted)}/{_last_relocation_debug.get('planned', len(moves))}; rejects: {breakdown}")
    except Exception:
        pass
    try:
        rej = _last_relocation_debug.get("rejections", {})
        rej_msg = ", ".join(f"{k}:{int(v)}" for k, v in rej.items()) if rej else "なし"
        _publish_progress(tid, {
            "type": "enforce_done", 
            "accepted": len(accepted), 
            "planned": _last_relocation_debug.get("planned", len(moves)), 
            "rejections": rej,
            "message": f"制約チェック完了: {len(accepted)}/{_last_relocation_debug.get('planned', len(moves))}件承認 (却下: {rej_msg})"
        })
    except Exception:
        pass
    return accepted


# -------------------------------
# Internal helpers
# -------------------------------

def _capacity_limit(fill_rate: Optional[float]) -> float:
    """棚1マスの容積上限 (m^3)。既定は 1.0*1.0*1.3 * 0.90。
    `fill_rate` が None の場合は DEFAULT_FILL_RATE を用いる。
    """
    cap = SHELF_WIDTH_M * SHELF_DEPTH_M * SHELF_HEIGHT_M
    fr = DEFAULT_FILL_RATE if fill_rate is None else float(fill_rate)
    return cap * fr

# Build capacity map & destination allow-list from location_master (optional)
from typing import Tuple as _Tuple, Set as _Set

def _cap_map_from_master(loc_master: pd.DataFrame, fill_rate: float) -> _Tuple[Dict[str, float], _Set[str]]:
    """Return (cap_by_loc, can_receive_set) derived from location_master.

    - loc_master expected cols: level, column, depth, capacity_m3 (or bay_*_mm), can_receive(optional)
    - Location ID format used here: 'LLLCCCDD' (3-digit level, 3-digit column, 2-digit depth)
    - Returned capacities already include `fill_rate` multiplier.
    """
    cap_by_loc: Dict[str, float] = {}
    can_receive: _Set[str] = set()
    if loc_master is None or loc_master.empty:
        return cap_by_loc, can_receive

    lm = loc_master.copy()
    # Normalize column names if needed
    rename = {
        "列（段)": "level",
        "列（段）": "level",
        "連（列)": "column",
        "連（列）": "column",
        "段（連)": "depth",
        "段（連）": "depth",
    }
    for k, v in rename.items():
        if k in lm.columns and v not in lm.columns:
            lm = lm.rename(columns={k: v})

    for req in ("level", "column", "depth"):
        if req not in lm.columns:
            return {}, set()

    # Capacity column or derive from bay_*_mm
    cap_col = None
    for cand in ("capacity_m3", "容量m3", "capacity", "容積m3"):
        if cand in lm.columns:
            cap_col = cand
            break
    if cap_col is None and all(c in lm.columns for c in ("bay_width_mm", "bay_depth_mm", "bay_height_mm")):
        lm["__cap"] = (
            pd.to_numeric(lm["bay_width_mm"], errors="coerce").fillna(1000).astype(float)
            * pd.to_numeric(lm["bay_depth_mm"], errors="coerce").fillna(1000).astype(float)
            * pd.to_numeric(lm["bay_height_mm"], errors="coerce").fillna(1300).astype(float)
        ) / 1_000_000_000.0
        cap_col = "__cap"
    if cap_col is None:
        return {}, set()

    # can_receive column (optional)
    cr_col = None
    for cand in ("can_receive", "移動先可否", "受入可"):
        if cand in lm.columns:
            cr_col = cand
            break

    # explicit forbidden column (optional, overrides can_receive)
    forbid_col = None
    for cand in ("destination_forbidden", "forbidden", "移動先指定不可", "移動不可"):
        if cand in lm.columns:
            forbid_col = cand
            break

    # Build maps
    for _, r in lm.iterrows():
        try:
            lvl = int(r["level"]) ; col = int(r["column"]) ; dep = int(r["depth"])  # type: ignore
        except Exception:
            continue
        loc_id = f"{lvl:03d}{col:03d}{dep:02d}"
        cap_val = pd.to_numeric(r[cap_col], errors="coerce")
        if pd.isna(cap_val):
            continue
        cap = float(cap_val) * float(fill_rate)
        cap_by_loc[loc_id] = cap
        # Compute effective allow/forbid
        allowed = True
        if cr_col is not None:
            try:
                allowed = bool(r[cr_col])
            except Exception:
                allowed = bool(r.get(cr_col, True))
        if forbid_col is not None:
            try:
                if bool(r[forbid_col]):
                    allowed = False
            except Exception:
                pass
        if allowed:
            can_receive.add(loc_id)
    return cap_by_loc, can_receive


# Filtering helper for location master by block/quality
def _filter_loc_master_by_block_quality(lm: pd.DataFrame, block_filter: Optional[Iterable[str]], quality_filter: Optional[Iterable[str]]) -> pd.DataFrame:
    """
    If `lm` has block/quality columns, restrict rows to the specified filters.
    Recognized columns:
      - block: 'ブロック略称' or 'block_code'
      - quality: '品質区分名' or 'quality_name'
    """
    if lm is None or lm.empty:
        return lm
    df = lm.copy()
    # block filter
    if block_filter is not None:
        blk_col = "ブロック略称" if "ブロック略称" in df.columns else ("block_code" if "block_code" in df.columns else None)
        if blk_col:
            df = df[df[blk_col].astype(str).isin(list(block_filter))].copy()
    # quality filter
    if quality_filter is not None:
        q_col = "品質区分名" if "品質区分名" in df.columns else ("quality_name" if "quality_name" in df.columns else None)
        if q_col:
            qset = set(str(x) for x in quality_filter)
            df = df[df[q_col].astype(str).isin(qset)].copy()
    return df


def _require_cols(df: pd.DataFrame, cols: Iterable[str], name: str = "df") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} に必要な列がありません: {missing}")


def _parse_loc8(s: str) -> Tuple[int, int, int]:
    """8桁ロケーション `LLLCCCDD` を (level, column, depth) に分解。
    例: '00100101' -> (1, 1, 1), '00101407' -> (1, 14, 7)
    """
    s = str(s)
    if len(s) != 8 or not s.isdigit():
        return (0, 0, 0)
    lvl = int(s[0:3])   # 段 3桁
    col = int(s[3:6])   # 列 3桁
    dep = int(s[6:8])   # 奥行 2桁
    return (lvl, col, dep)


def _depth_center_distance(dep: int, col: int, depths_by_col: Optional[Dict[int, Tuple[int, float]]]) -> float:
    """中心からの距離（0=中心が最良）を返す。depths_by_col は {col: (max_dep, center)}。
    center は奇数本数なら中央(例: 5本→3)、偶数本数なら中央2点の平均(例: 6本→3.5)。
    対応する列情報が無い場合は 0 を返し、影響を無くす。
    """
    try:
        if depths_by_col is None:
            return 0.0
        info = depths_by_col.get(int(col))
        if not info:
            return 0.0
        _max_dep, center = info
        return abs(float(dep) - float(center))
    except Exception:
        return 0.0


def _location_key(loc: str) -> Tuple[int, int, int]:
    """入口に近い順（列→奥→段）でのソートキー。"""
    lv, col, dep = _parse_loc8(loc)
    return (col, dep, lv)

def _ease_key(lv: int, col: int, dep: int) -> int:
    """取りやすさキー。LV/DEPは従来どおり小さいほど良い、COLは**大きいほど良い**に反転（41が最強）。"""
    try:
        # 列は 1 → 41, 41 → 1 へ反転してスコア化（小さいほど良い）
        col_rev = 42 - int(col)
        return int(lv) * 10000 + int(col_rev) * 100 + int(dep)
    except Exception:
        return 99_999_999

# --- Band preference helper ---
def _band_preference(row: pd.Series, tcol: int, cfg: OptimizerConfig) -> float:
    """入り数と品質(販促系)に応じた列バンド嗜好のスコア（負がボーナス＝好ましい）。
    - 少ない(=重い)SKU → near_cols(既定:35–41) へ寄せる
    - 多い(=軽い)または販促資材 → far_cols(既定:1–11) へ寄せる
    """
    try:
        pack_val = row.get("pack_est")
        try:
            pack = float(pack_val) if pack_val is not None and not pd.isna(pack_val) else 0.0
        except Exception:
            pack = 0.0
        qname = str(row.get("quality_name") or row.get("品質区分名") or "").strip()

        W = float(getattr(cfg, "band_pref_weight", 20.0))
        # near/far は small/large エリア定義があればそれを優先
        near = set(getattr(cfg, "near_cols", getattr(cfg, "small_pack_cols", tuple(range(35, 42)))))
        far  = set(getattr(cfg, "far_cols",  getattr(cfg, "large_pack_cols", tuple(range(1, 12)))))
        low_max  = int(getattr(cfg, "pack_low_max", 12))
        high_min = int(getattr(cfg, "pack_high_min", 50))
        promo_kw = tuple(getattr(cfg, "promo_quality_keywords", ("販促資材", "販促", "什器", "資材")))

        delta = 0.0
        is_promo = any(k in qname for k in promo_kw)

        if pack >= high_min or is_promo:
            # 多い(=軽い) or 販促 → 1–11 へ寄せる
            if tcol in far:
                delta -= W
            if tcol in near:
                delta += W
        elif pack > 0 and pack <= low_max:
            # 少ない(=重い) → 35–41 へ寄せる
            if tcol in near:
                delta -= W
            if tcol in far:
                delta += W

        return float(delta)
    except Exception:
        return 0.0


# --- Helper functions for promo and area gating ---
def _is_promo(row: pd.Series, cfg: OptimizerConfig) -> bool:
    qname = str(row.get("quality_name") or row.get("品質区分名") or "").strip()
    promo_kw = tuple(getattr(cfg, "promo_quality_keywords", ("販促資材", "販促", "什器", "資材")))
    return any(k in qname for k in promo_kw)

def _allowed_cols_for_row(row: pd.Series, cfg: OptimizerConfig) -> Set[int]:
    """入数・販促からこの行に許容される列レンジ（セット）を返す。strict_pack_area 用。"""
    small = set(getattr(cfg, "small_pack_cols", tuple(range(35, 42))))
    medium = set(getattr(cfg, "medium_pack_cols", tuple(range(12, 35))))
    large = set(getattr(cfg, "large_pack_cols", tuple(range(1, 12))))
    low_max  = int(getattr(cfg, "pack_low_max", 12))
    high_min = int(getattr(cfg, "pack_high_min", 50))
    try:
        pack_val = row.get("pack_est")
        pack = float(pack_val) if pack_val is not None and not pd.isna(pack_val) else None
    except Exception:
        pack = None
    if _is_promo(row, cfg) or (pack is not None and pack >= high_min):
        return set(large)
    if pack is not None and pack <= low_max:
        return set(small)
    return set(medium)


def _column_of(loc: str) -> int:
    return _parse_loc8(str(loc))[1]


_lot_date_re = re.compile(r"(20\d{6})")  # 例: 20250114



def _parse_lot_date_key(lot: str) -> int:
    """ロットから日付キー(YYYYMMDD int)を抽出。
    想定パターン:
      - 英字の直後に続く数字が日付。YYYYMMDD(8桁) または YYYYMM(6桁)。
      - 末尾に連番等の余分な桁が付く場合は先頭8桁を利用。
      - 月まで(6桁)しかない場合は 01 日とする。
    合致しない場合は 99999999 を返す。
    """
    if lot is None:
        return UNKNOWN_LOT_KEY
    s = str(lot)

    # 英字の最後の位置を探し、その直後からの数字列を優先的に読み取る
    last_alpha_pos = -1
    for i, ch in enumerate(s):
        if ch.isalpha():
            last_alpha_pos = i
    digits = "".join(ch for ch in s[last_alpha_pos + 1 :] if ch.isdigit())

    # 英字の直後に十分な数字がなければ、全体から "20..." で始まる塊を拾う
    if len(digits) < 6:
        m = re.search(r"(20\d{6,})", s)  # 8桁以上の連番（先頭8桁を日付とみなす）
        if m:
            digits = m.group(1)

    # 8桁以上あれば先頭8桁を YYYYMMDD として採用
    if len(digits) >= 8:
        y, mo, d = digits[:4], digits[4:6], digits[6:8]
        try:
            y_i, mo_i, d_i = int(y), int(mo), int(d)
            # 範囲のざっくりバリデーション（不正ならフォールバック）
            if not (2000 <= y_i <= 2099 and 1 <= mo_i <= 12 and 1 <= d_i <= 31):
                raise ValueError
            return int(f"{y}{mo}{d}")
        except Exception:
            pass  # フォールバックへ

    # ちょうど6桁(YYYYMM)の場合は 01 日を補う
    if len(digits) == 6:
        y, mo = digits[:4], digits[4:6]
        try:
            y_i, mo_i = int(y), int(mo)
            if not (2000 <= y_i <= 2099 and 1 <= mo_i <= 12):
                raise ValueError
            return int(f"{y}{mo}01")
        except Exception:
            pass

    return UNKNOWN_LOT_KEY


# 入数が±帯域内か判定
def _within_band(pack: float, center: float, band_ratio: float) -> bool:
    """pack が center の ±band_ratio 内に収まるかを判定。center<=0 または変換失敗時は True 扱い。"""
    try:
        pack = float(pack)
        center = float(center)
        if center <= 0:
            return True
        return abs(pack - center) / center <= float(band_ratio)
    except Exception:
        return True


def _build_carton_volume_map(sku_master: pd.DataFrame) -> pd.Series:
    """SKU -> 1ケース容積(m^3) マップを構築。候補列の優先順に使用。"""
    key = "sku_id" if "sku_id" in sku_master.columns else "商品ID"
    vol_col = None
    for cand in ("carton_volume_m3", "volume_m3", "容積m3", "容積m^3"):
        if cand in sku_master.columns:
            vol_col = cand
            break
    if vol_col is None:
        return pd.Series(0.0, index=sku_master[key].astype(str))
    series = pd.to_numeric(sku_master[vol_col], errors="coerce").fillna(0.0).astype(float)
    series.index = sku_master[key].astype(str)
    return series


def _sort_index_for_pick(inv: pd.DataFrame) -> pd.Index:
    """古いロット優先 & 取りやすさ（高い段から）優先で並べ替えた index を返す。"""
    return inv.sort_values(
        by=["lot_key", "lv", "col", "dep"],
        ascending=[True, False, True, True],  # 古い→先、高い段→先
        kind="mergesort",
    ).index


def _violates_lot_level_rule(
    inv: pd.DataFrame,
    sku: str,
    lot_key: int,
    target_level: int,
    tcol: int,
    tdep: int,
    moving_index,
    inv_rows_by_loc_sku: Optional[Dict] = None,  # OPTIMIZATION: pre-built index
) -> bool:
    """同一列(=tcol)に限定して『古いロットほど低段／新しいロットほど高段』を**厳密**に維持する。
    条件:
      - 同列にある「新しいロット」の最小段 < target_level だと違反（新しいは必ず本行以上の段）
      - 同列にある「古いロット」の最大段 > target_level だと違反（古いは必ず本行以下の段）
    """
    need_cols = {"lv", "lot_key", "商品ID", "col"}
    if not need_cols.issubset(inv.columns):
        return False

    # OPTIMIZED path: use pre-built column-based index if available
    # We need inventory rows for this SKU in this column
    # Since inv_rows_by_loc_sku is by location, we need to check all locations in the column
    # This is still O(n) worst case, so use DataFrame for now but with .query() for speed
    
    same = inv.query("`商品ID` == @sku and col == @tcol")
    if same.empty:
        return False

    newer = same[same["lot_key"] > lot_key]
    if not newer.empty:
        min_newer_lv = int(newer["lv"].min())
        if min_newer_lv < target_level:
            return True

    older = same[same["lot_key"] < lot_key]
    if not older.empty:
        max_older_lv = int(older["lv"].max())
        if max_older_lv > target_level:
            return True

    return False


def _violates_lot_level_rule_fast(
    sku: str,
    lot_key: int,
    target_level: int,
    tcol: int,
    inv_lot_levels_by_sku_col: Dict[Tuple[str, int], List[Tuple[int, int]]],
) -> bool:
    """Fast version using pre-built index. Same logic as _violates_lot_level_rule.
    
    Rule: 同一列に限定して『古いロットほど低段／新しいロットほど高段』を厳密に維持
    - 新しいロット（lot_key > current）の最小段 < target_level → 違反
    - 古いロット（lot_key < current）の最大段 > target_level → 違反
    """
    key = (str(sku), int(tcol))
    lot_levels = inv_lot_levels_by_sku_col.get(key)
    if not lot_levels:
        return False
    
    # Find min level of newer lots and max level of older lots
    min_newer_level = None
    max_older_level = None
    
    for other_lot_key, other_level in lot_levels:
        if other_lot_key > lot_key:
            # Newer lot
            if min_newer_level is None or other_level < min_newer_level:
                min_newer_level = other_level
        elif other_lot_key < lot_key:
            # Older lot
            if max_older_level is None or other_level > max_older_level:
                max_older_level = other_level
    
    # Check violations
    if min_newer_level is not None and min_newer_level < target_level:
        return True
    if max_older_level is not None and max_older_level > target_level:
        return True
    
    return False


# ============================================================================
# Pass-FIFO: 列を跨いだFIFO最適化（同一SKU内で最古ロットを最優先引き当て位置へ）
# ============================================================================
def _pass_fifo_cross_column(
    inv: pd.DataFrame,
    shelf_usage: dict[str, float],
    cap_limit: float,
    cfg: OptimizerConfig,
    cap_by_loc: Optional[Dict[str, float]] = None,
    can_receive: Optional[Set[str]] = None,
    *,
    budget_left: Optional[int] = None,
) -> List[Move]:
    """同一SKUの全ロットを列を跨いで評価し、最古ロットを最優先引き当て位置に配置。
    
    引き当て順序の優先度: Lv小 → 列小 → 奥行小
    目的: 「古いロットが必ず先に引き当てられる」ようにロケーションを再配置
    """
    moves: List[Move] = []
    def _can_add(k: int) -> bool:
        return (budget_left is None) or (len(moves) + int(k) <= int(budget_left))
    
    if inv.empty:
        print("[Pass-FIFO] inv is empty")
        return moves
    
    # 有効な在庫のみ（容積あり、ロット日付あり、移動対象）
    # is_movable: バラのみ or 2ケース以下+バラ混在 は除外済み
    filter_cond = (inv["volume_each_case"] > 0) & (inv["lot_key"] != UNKNOWN_LOT_KEY)
    if "is_movable" in inv.columns:
        filter_cond = filter_cond & (inv["is_movable"] == True)
    subset = inv[filter_cond].copy()
    if subset.empty:
        print("[Pass-FIFO] subset is empty after filtering")
        return moves
    
    pick_levels = [int(x) for x in getattr(cfg, "pick_levels", (1, 2))]
    
    # 引き当て優先度キー（小さいほど優先）
    def _pick_priority_key(lv: int, col: int, dep: int) -> tuple:
        return (lv, col, dep)
    
    # SKU毎にグループ化して処理
    sku_groups = subset.groupby("商品ID")
    processed_skus = 0
    skus_with_cross_column_issue = 0
    
    for sku, sku_df in sku_groups:
        if len(sku_df) <= 1:
            continue
        
        # このSKUの全ロットを列を問わず収集
        lots_info = []
        for idx, row in sku_df.iterrows():
            lot_key = int(row["lot_key"])
            from_loc = str(row["ロケーション"])
            lv, col, dep = _parse_loc8(from_loc)
            qty_cases = int(row.get("qty_cases_move") or 0)
            vol_each = float(row.get("volume_each_case") or 0.0)
            if qty_cases <= 0 or vol_each <= 0:
                continue
            lots_info.append({
                "idx": idx,
                "row": row,
                "lot_key": lot_key,
                "from_loc": from_loc,
                "lv": lv,
                "col": col,
                "dep": dep,
                "qty_cases": qty_cases,
                "vol_each": vol_each,
                "need_vol": vol_each * qty_cases,
                "pick_priority": _pick_priority_key(lv, col, dep),
            })
        
        if len(lots_info) <= 1:
            continue
        
        # ロット日付順にソート（古い順）
        lots_sorted_by_date = sorted(lots_info, key=lambda x: x["lot_key"])
        # 引き当て優先度順にソート（現在位置ベース）
        lots_sorted_by_priority = sorted(lots_info, key=lambda x: x["pick_priority"])
        
        # 違反チェック: 古いロットが後から引き当てられる場合
        violations = []
        for i, current in enumerate(lots_sorted_by_priority):
            for j, other in enumerate(lots_sorted_by_priority[i+1:], i+1):
                if current["lot_key"] > other["lot_key"]:
                    # current(新しい)がother(古い)より先に引き当てられる = FIFO違反
                    violations.append((current, other))
        
        if not violations:
            continue
        
        skus_with_cross_column_issue += 1
        processed_skus += 1
        
        # 違反を解消するための移動を計画
        # 戦略: 最古ロットを、現在空いている中で最も引き当て優先度が高い位置へ移動
        # 重要: 古い順に優先位置を「予約」していき、新しいロットが古いロットより優先位置にならないようにする
        
        # 取口レベル(Lv1-2)の空きロケーションを収集
        available_pick_locs = []
        for loc in shelf_usage.keys():
            if loc in PLACEHOLDER_LOCS:
                continue
            if can_receive is not None and loc not in can_receive:
                continue
            lv, col, dep = _parse_loc8(loc)
            if lv not in pick_levels:
                continue
            used = float(shelf_usage.get(loc, 0.0))
            limit = cap_by_loc.get(loc, cap_limit) if cap_by_loc else cap_limit
            available_cap = limit - used
            if available_cap <= 0:
                continue
            # 完全空きロケか判定（used==0）
            is_empty = (used == 0.0)
            available_pick_locs.append({
                "loc": loc,
                "lv": lv,
                "col": col,
                "dep": dep,
                "available_cap": available_cap,
                "is_empty": is_empty,  # 空きロケフラグ
                "pick_priority": _pick_priority_key(lv, col, dep),
            })
        
        # 引き当て優先度順にソート（空きロケを優先、同優先度なら完全空きを先に）
        # ソートキー: (is_empty=Falseが後ろ, pick_priority)
        available_pick_locs.sort(key=lambda x: (not x["is_empty"], x["pick_priority"]))
        
        # 古いロットから順に最優先位置を割り当てる
        # next_priority_idx: 次に割り当てる優先位置のインデックス
        next_priority_idx = 0
        used_locs = set()  # この処理で使用したロケーション
        
        for lot_info in lots_sorted_by_date:
            if not _can_add(1):
                break
            
            current_priority = lot_info["pick_priority"]
            
            # このロットに割り当てるべき位置を探す
            # 条件: 現在位置より優先度が高く、まだ使われていない位置
            best_dest = None
            while next_priority_idx < len(available_pick_locs):
                dest = available_pick_locs[next_priority_idx]
                
                # 既に別のロットで使用済み、または容量不足ならスキップ
                if dest["loc"] in used_locs or dest["available_cap"] < lot_info["need_vol"]:
                    next_priority_idx += 1
                    continue
                
                # 現在位置と同じならスキップして次へ
                if dest["loc"] == lot_info["from_loc"]:
                    # この位置は「このロットのもの」としてマーク
                    used_locs.add(dest["loc"])
                    next_priority_idx += 1
                    break
                
                # この位置が現在位置より優先度が高いなら移動
                if dest["pick_priority"] < current_priority:
                    best_dest = dest
                    used_locs.add(dest["loc"])
                    next_priority_idx += 1
                    break
                else:
                    # 現在位置の方が優先度が高い = 移動不要
                    # ただし、次のロットのために現在位置は「使用済み」にしておく
                    next_priority_idx += 1
                    break
            
            if best_dest is None:
                continue
            
            # 移動を記録
            to_loc = best_dest["loc"]
            to_lv, to_col, to_dep = best_dest["lv"], best_dest["col"], best_dest["dep"]
            from_lv, from_col, from_dep = lot_info["lv"], lot_info["col"], lot_info["dep"]
            row = lot_info["row"]
            lot_key = lot_info["lot_key"]
            
            # ロット日付のフォーマット
            lot_date_str = _lot_key_to_datestr8(lot_key)
            formatted_date = ""
            if lot_date_str:
                formatted_date = f"{lot_date_str[0:4]}/{lot_date_str[4:6]}/{lot_date_str[6:8]}"
            
            # 移動理由を構築
            reason_parts = []
            reason_parts.append(f"FIFO是正(Lot:{formatted_date})")
            reason_parts.append(f"列{from_col}→{to_col}")
            if to_lv != from_lv:
                reason_parts.append(f"Lv{from_lv}→{to_lv}")
            reason_parts.append("最古ロット優先引当")
            
            moves.append(Move(
                sku_id=str(sku),
                lot=str(row.get("ロット") or ""),
                qty=lot_info["qty_cases"],
                from_loc=str(lot_info["from_loc"]).zfill(8),
                to_loc=str(to_loc).zfill(8),
                lot_date=lot_date_str,
                reason=" → ".join(reason_parts),
                chain_group_id=f"p1fifo_{secrets.token_hex(6)}",
                execution_order=1,
            ))
            
            # 棚使用量を更新
            shelf_usage[lot_info["from_loc"]] = max(0.0, shelf_usage.get(lot_info["from_loc"], 0.0) - lot_info["need_vol"])
            shelf_usage[to_loc] = shelf_usage.get(to_loc, 0.0) + lot_info["need_vol"]
            
            # 使用済みとしてマーク
            used_locs.add(to_loc)
            
            # invのロケーション情報も更新
            inv.at[lot_info["idx"], "lv"] = to_lv
            inv.at[lot_info["idx"], "col"] = to_col
            inv.at[lot_info["idx"], "dep"] = to_dep
            inv.at[lot_info["idx"], "ロケーション"] = to_loc
            
            # lot_infoの優先度も更新
            lot_info["lv"] = to_lv
            lot_info["col"] = to_col
            lot_info["dep"] = to_dep
            lot_info["from_loc"] = to_loc
            lot_info["pick_priority"] = _pick_priority_key(to_lv, to_col, to_dep)
    
    # 移動先が空きロケだった件数をカウント
    empty_loc_used = sum(1 for m in moves if shelf_usage.get(m.to_loc, 0.0) == 0.0 or m.to_loc in used_locs)
    print(f"[Pass-FIFO] Processed {processed_skus} SKUs with cross-column issues, generated {len(moves)} moves (空きロケ使用: {empty_loc_used}件)")
    return moves


#
# Pass-1: Pick/Storage balance (古いロットを取口=1-2段、新しいロットを保管=3-4段)
def _pass1_pick_storage_balance(
    inv: pd.DataFrame,
    shelf_usage: dict[str, float],
    cap_limit: float,
    cfg: OptimizerConfig,
    cap_by_loc: Optional[Dict[str, float]] = None,
    can_receive: Optional[Set[str]] = None,
    *,
    budget_left: Optional[int] = None,
) -> List[Move]:
    """同一SKU×同一列ごとに、古い順に並べて
    先頭 |pick_levels| 件を取口（例:1,2段）、残りを保管（例:3,4段）に**割当**し、
    乖離している行のみ同列内で最小移動して整列する。
    さらに、目標段に空きが無い場合は **同列内スワップ**（occupantを望ましい側へ退避→空いたマスに本行を移動）を行う。
    """
    moves: List[Move] = []
    def _can_add(k: int) -> bool:
        return (budget_left is None) or (len(moves) + int(k) <= int(budget_left))
    if inv.empty:
        print("[Pass-1] inv is empty, returning 0 moves")
        return moves

    # Debug: Check lot_key status
    total_rows = len(inv)
    has_volume = len(inv[inv["volume_each_case"] > 0])
    has_lot = len(inv[inv["lot_key"] != UNKNOWN_LOT_KEY])
    print(f"[Pass-1] total={total_rows}, has_volume={has_volume}, has_lot={has_lot}, unknown_lot_key={UNKNOWN_LOT_KEY}")
    
    # 有効な在庫のみ（容積あり、ロット日付あり、移動対象）
    # is_movable: バラのみ or 2ケース以下+バラ混在 は除外済み
    filter_cond = (inv["volume_each_case"] > 0) & (inv["lot_key"] != UNKNOWN_LOT_KEY)
    if "is_movable" in inv.columns:
        filter_cond = filter_cond & (inv["is_movable"] == True)
    subset = inv[filter_cond].copy()
    if subset.empty:
        print(f"[Pass-1] subset is empty after filtering (needs volume>0 AND lot_key!={UNKNOWN_LOT_KEY})")
        return moves
    print(f"[Pass-1] subset size after filter: {len(subset)}")

    pick_levels = [int(x) for x in getattr(cfg, "pick_levels", (1, 2))]
    storage_levels = [int(x) for x in getattr(cfg, "storage_levels", (3, 4))]
    enable_swap = bool(getattr(cfg, "enable_pass1_swap", True))
    swap_budget_default = int(getattr(cfg, "pass1_swap_budget_per_group", 2))

    def _side(lv: int) -> str:
        if lv in pick_levels: return "pick"
        if lv in storage_levels: return "storage"
        return "other"

    groups_processed = 0
    groups_with_moves = 0
    total_candidates = 0

    for (c, sku), g in subset.groupby(["col", "商品ID"]):
        groups_processed += 1
        try:
            c_int = int(c)
        except Exception:
            continue
        # グループ内スワップ回数の上限
        swap_budget_left = max(0, swap_budget_default)

        # 古い→新しい順に（同一レベル内は低段優先）
        g = g.sort_values(["lot_key", "lv"], ascending=[True, True])
        # 望ましい割当（index -> 目標レベル）
        desired_levels: Dict[int, int] = {}
        rows = list(g.iterrows())
        
        # Debug: Log group info
        if len(rows) > 1 and groups_processed <= 3:  # Log first 3 multi-row groups
            print(f"[Pass-1] Group: col={c}, sku={sku}, rows={len(rows)}")
            for rank, (idx, row) in enumerate(rows):
                print(f"  rank={rank}, lv={row['lv']}, lot_key={row['lot_key']}, loc={row['ロケーション']}")
        
        for rank, (idx, row) in enumerate(rows):
            if rank < len(pick_levels):
                desired_levels[idx] = pick_levels[rank]
            else:
                desired_levels[idx] = storage_levels[0] if storage_levels else int(row["lv"])

        # 乖離している行のみ移動
        group_moves = 0
        for idx, row in rows:
            cur_lv = int(row["lv"]) ; lot_key = int(row["lot_key"]) ; from_loc = str(row["ロケーション"])
            qty_cases = int(row.get("qty_cases_move") or 0)
            if qty_cases <= 0:
                continue
            need_vol = float(row.get("volume_each_case") or 0.0) * qty_cases
            target_lv = int(desired_levels.get(idx, cur_lv))

            # すでに目標段にいる場合はスキップ
            if cur_lv == target_lv:
                continue

            # --- 1) 同列・目標段の空き候補（容量OK)を探索
            cand_locs = [
                loc for loc in shelf_usage.keys()
                if loc not in PLACEHOLDER_LOCS
                and loc != from_loc
                and _parse_loc8(loc)[1] == c_int
                and _parse_loc8(loc)[0] == target_lv
                and (can_receive is None or loc in can_receive)
            ]
            # 目標段が埋まっている場合、同側の他段にフォールバック
            if not cand_locs and _side(target_lv) == "storage":
                cand_locs = [
                    loc for loc in shelf_usage.keys()
                    if loc not in PLACEHOLDER_LOCS
                    and loc != from_loc
                    and _parse_loc8(loc)[1] == c_int
                    and _parse_loc8(loc)[0] in storage_levels
                    and (can_receive is None or loc in can_receive)
                ]
            if not cand_locs and _side(target_lv) == "pick":
                cand_locs = [
                    loc for loc in shelf_usage.keys()
                    if loc not in PLACEHOLDER_LOCS
                    and loc != from_loc
                    and _parse_loc8(loc)[1] == c_int
                    and _parse_loc8(loc)[0] in pick_levels
                    and (can_receive is None or loc in can_receive)
                ]
            cand_locs.sort(key=_location_key)

            best_to = None
            best_score = math.inf
            empty_cand_count = sum(1 for loc in cand_locs if shelf_usage.get(loc, 0.0) == 0.0)
            for to_loc in cand_locs:
                tlv, tcol, tdep = _parse_loc8(to_loc)
                used = float(shelf_usage.get(to_loc, 0.0))
                limit = cap_by_loc.get(to_loc, cap_limit) if cap_by_loc else cap_limit
                if used + need_vol > limit:
                    continue
                if _violates_lot_level_rule(inv, str(row["商品ID"]), lot_key, tlv, tcol, tdep, idx):
                    continue
                # 空きロケ（used==0）を優先するスコア計算
                # 空きロケなら-1000でスコアを大幅に下げる
                is_empty_bonus = -1000.0 if (used == 0.0) else 0.0
                score = abs(tlv - cur_lv) + (tdep * 0.001) + is_empty_bonus
                if score < best_score:
                    best_score = score
                    best_to = to_loc
            
            # デバッグ: 空きロケ候補がある場合はログ出力
            if empty_cand_count > 0 and best_to and shelf_usage.get(best_to, 0.0) > 0:
                try:
                    print(f"[Pass-1 Debug] 空きロケ{empty_cand_count}件あったが選ばれず: col={c_int}, sku={row['商品ID']}, best_to={best_to} (used={shelf_usage.get(best_to, 0.0):.2f})")
                except:
                    pass

            # Initialize swap chain tracking
            _pending_swap_chain_id = None

            # --- 2) 空き候補が全く無い場合、同列内スワップを試行
            if best_to is None and enable_swap and swap_budget_left > 0:
                # 目標段(target_lv)にいる“占有者”のうち、望ましい段に居ない行を優先して退避
                occs = g[g["lv"].astype(int) == int(target_lv)]
                # スワップ候補選定：自分自身は除外
                occs = occs[occs.index != idx]
                # occupant の望ましい段
                def _desired_for(i: int, default_lv: int) -> int:
                    return int(desired_levels.get(i, default_lv))
                best_occ = None
                best_occ_dest = None
                best_occ_score = math.inf
                for o_idx, o_row in occs.iterrows():
                    o_from = str(o_row["ロケーション"]) ; o_cur_lv = int(o_row["lv"]) ; o_lot_key = int(o_row["lot_key"]) ; o_qty = int(o_row.get("qty_cases_move") or 0)
                    if o_qty <= 0:
                        continue
                    o_need_vol = float(o_row.get("volume_each_case") or 0.0) * o_qty
                    o_desired_lv = _desired_for(o_idx, o_cur_lv)
                    # occupant のターゲット側のレベル集合
                    if _side(o_desired_lv) == "pick":
                        o_side_lv = list(pick_levels)
                    elif _side(o_desired_lv) == "storage":
                        o_side_lv = list(storage_levels)
                    else:
                        # 未分類は現状側の反対（targetの反対）へ逃がす
                        o_side_lv = list(storage_levels if _side(target_lv) == "pick" else pick_levels)

                    # occupant の退避先候補（同列内・容量OK）
                    occ_cands = [
                        loc for loc in shelf_usage.keys()
                        if loc not in PLACEHOLDER_LOCS
                        and loc != o_from
                        and _parse_loc8(loc)[1] == c_int
                        and _parse_loc8(loc)[0] in o_side_lv
                        and (can_receive is None or loc in can_receive)
                    ]
                    occ_cands.sort(key=_location_key)

                    for o_to in occ_cands:
                        olv, ocol, odep = _parse_loc8(o_to)
                        used_o = float(shelf_usage.get(o_to, 0.0))
                        limit_o = cap_by_loc.get(o_to, cap_limit) if cap_by_loc else cap_limit
                        if used_o + o_need_vol > limit_o:
                            continue
                        # occupant の移動でも FIFO 破綻は不可
                        if _violates_lot_level_rule(inv, str(o_row["商品ID"]), o_lot_key, olv, ocol, odep, o_idx):
                            continue
                        o_score = abs(olv - o_cur_lv) + (odep * 0.001)
                        if o_score < best_occ_score:
                            best_occ_score = o_score
                            best_occ = (o_idx, o_row)
                            best_occ_dest = o_to

                # occupant の退避先が見つかったら、occupant→退避 → 本行→空席 の順で2手実行
                if best_occ is not None and best_occ_dest is not None:
                    o_idx, o_row = best_occ
                    o_from = str(o_row["ロケーション"]) ; o_cur_lv = int(o_row["lv"]) ; o_lot_key = int(o_row["lot_key"]) ; o_qty = int(o_row.get("qty_cases_move") or 0)
                    o_need_vol = float(o_row.get("volume_each_case") or 0.0) * o_qty

                    # 2-1) occupant move（本行と合わせて2手必要。足りなければスキップ）
                    if not _can_add(2):
                        best_to = None
                    else:
                        # Generate chain_group_id for linked moves
                        swap_chain_id = f"swap_{secrets.token_hex(6)}"
                        
                        tlv_o, tcol_o, tdep_o = _parse_loc8(best_occ_dest)
                        moves.append(
                            Move(
                                sku_id=str(o_row["商品ID"]),
                                lot=str(o_row.get("ロット") or ""),
                                qty=o_qty,
                                from_loc=str(o_from).zfill(8),
                                to_loc=str(best_occ_dest).zfill(8),
                                lot_date=_lot_key_to_datestr8(o_lot_key),
                                reason="スワップ準備退避 → 優先商品用スペース確保",
                                chain_group_id=swap_chain_id,
                                execution_order=1,  # First: evacuate occupant
                            )
                        )
                        shelf_usage[o_from] = max(0.0, shelf_usage.get(o_from, 0.0) - o_need_vol)
                        shelf_usage[best_occ_dest] = shelf_usage.get(best_occ_dest, 0.0) + o_need_vol
                        # inv を更新
                        inv.at[o_idx, "lv"] = tlv_o
                        inv.at[o_idx, "col"] = tcol_o
                        inv.at[o_idx, "dep"] = tdep_o
                        inv.at[o_idx, "ロケーション"] = best_occ_dest
                        # 2-2) 本行 move（occupant が空けたマスへ＝o_from）
                        best_to = o_from
                        # Remember chain_group_id for main move
                        _pending_swap_chain_id = swap_chain_id
                        swap_budget_left -= 1

            # スワップしても候補が無ければ、この行は Pass-2 へ委譲
            if best_to is None:
                continue

            # 移動実行（best_to は通常候補 or スワップ空席）
                        # 予算チェック：本行の1手を追加できるか
            if not _can_add(1):
                return moves
            tlv, tcol, tdep = _parse_loc8(best_to)
            lk = int(row.get("lot_key")) if pd.notna(row.get("lot_key")) else _parse_lot_date_key(str(row.get("ロット") or ""))
            
            # Determine reason based on move characteristics
            from_lv, from_col, from_dep = _parse_loc8(from_loc)
            
            # Calculate lot age
            lot_age_days = 0
            try:
                if lk != UNKNOWN_LOT_KEY:
                    from datetime import datetime
                    lot_date = datetime.strptime(_lot_key_to_datestr8(lk), "%Y%m%d")
                    lot_age_days = (datetime.now() - lot_date).days
            except Exception:
                pass
            
            # Build detailed reason - Pass-1 specific (FIFO enforcement)
            actions = []
            improvements = []
            
            # Indicate pick/storage side with lot context
            if target_lv in pick_levels and cur_lv not in pick_levels:
                # Moving from storage to pick = older lot to picking area
                lot_info = ""
                if lk != UNKNOWN_LOT_KEY:
                    lot_date_str = _lot_key_to_datestr8(lk)
                    if lot_date_str:
                        # Format: 20250424 -> 2025/04/24
                        formatted_date = f"{lot_date_str[0:4]}/{lot_date_str[4:6]}/{lot_date_str[6:8]}"
                        lot_info = f"(Lot:{formatted_date})"
                actions.append(f"古ロット{lot_info}→取口Lv{tlv}")
                improvements.append("先入先出徹底")
            elif target_lv in storage_levels and cur_lv not in storage_levels:
                # Moving from pick to storage = newer lot to storage area
                lot_info = ""
                if lk != UNKNOWN_LOT_KEY:
                    lot_date_str = _lot_key_to_datestr8(lk)
                    if lot_date_str:
                        formatted_date = f"{lot_date_str[0:4]}/{lot_date_str[4:6]}/{lot_date_str[6:8]}"
                        lot_info = f"(Lot:{formatted_date})"
                actions.append(f"新ロット{lot_info}→保管Lv{tlv}")
                improvements.append("在庫保管最適化")
            elif target_lv in pick_levels and cur_lv in pick_levels:
                # Within pick levels - lot reordering
                if tlv < from_lv:
                    actions.append(f"取口内整列(Lv{from_lv}→{tlv})")
                    improvements.append("FIFO順序最適化")
                else:
                    actions.append(f"取口内再配置(Lv{from_lv}→{tlv})")
            elif target_lv in storage_levels and cur_lv in storage_levels:
                # Within storage levels - lot reordering
                if tlv < from_lv:
                    actions.append(f"保管内整列(Lv{from_lv}→{tlv})")
                    improvements.append("FIFO順序最適化")
                else:
                    actions.append(f"保管内再配置(Lv{from_lv}→{tlv})")
            elif tlv < from_lv:
                actions.append(f"段下げ(Lv{from_lv}→{tlv})")
                improvements.append("作業効率UP")
            
            # Column movement
            if tcol != from_col:
                if abs(tcol - from_col) <= 2:
                    actions.append(f"同列内移動(列{from_col}→{tcol})")
                else:
                    actions.append(f"列移動(列{from_col}→{tcol})")
            
            # Depth optimization
            if tdep < from_dep:
                actions.append(f"手前化(奥{from_dep}→{tdep})")
                improvements.append("取り出し容易化")
            
            if not actions:
                actions.append("同一レベル内移動")
                improvements.append("在庫集約")
            
            move_reason = " & ".join(actions)
            if improvements:
                move_reason += " → " + "、".join(improvements)
            
            moves.append(
                Move(
                    sku_id=str(row["商品ID"]),
                    lot=str(row.get("ロット") or ""),
                    qty=qty_cases,
                    from_loc=str(from_loc).zfill(8),
                    to_loc=str(best_to).zfill(8),
                    lot_date=_lot_key_to_datestr8(lk),
                    reason=move_reason,
                    chain_group_id=_pending_swap_chain_id or f"p1main_{secrets.token_hex(6)}",
                    execution_order=2 if _pending_swap_chain_id else 1,
                )
            )
            group_moves += 1
            total_candidates += 1
            
            # 使用量更新
            shelf_usage[from_loc] = max(0.0, shelf_usage.get(from_loc, 0.0) - need_vol)
            shelf_usage[best_to] = shelf_usage.get(best_to, 0.0) + need_vol

            # inv を更新
            inv.at[idx, "lv"] = tlv
            inv.at[idx, "col"] = tcol
            inv.at[idx, "dep"] = tdep
            inv.at[idx, "ロケーション"] = best_to
        
        # End of group loop - update statistics
        if group_moves > 0:
            groups_with_moves += 1

    print(f"[Pass-1] Summary: groups={groups_processed}, with_moves={groups_with_moves}, total_moves={len(moves)}")
    if moves:
        print(f"[optimizer] pass1_pick_storage_balance (with swap) moves={len(moves)}")
    return moves

# -------------------------------
# Eviction chain helpers (bounded depth)
# -------------------------------

@dataclass
class _ChainBudget:
    depth_left: int
    evictions_left: int
    touch_left: int
    touched: Set[str]
    max_depth_used: int = 0

def _score_destination_for_row(
    row: pd.Series,
    to_loc: str,
    rep_pack_by_col: Dict[int, float],
    pack_tolerance_ratio: float,
    same_sku_same_column_bonus: float,
    prefer_same_column_bonus: float,
    split_sku_new_column_penalty: float,
    ai_col_hints: Optional[Dict[str, List[int]]] = None,
    *,
    ease_weight: float = 0.0001,
    depths_by_col: Optional[Dict[int, Tuple[int, float]]] = None,
    cfg: OptimizerConfig | None = None,
) -> float:
    sku_val = str(row["商品ID"])
    pack_val = row.get("pack_est")
    cur_col = int(row.get("col") or 0)
    tlv, tcol, tdep = _parse_loc8(str(to_loc))

    score = 0.0
    # AI column hints
    if ai_col_hints:
        try:
            pref_cols = ai_col_hints.get(sku_val) or ai_col_hints.get(str(sku_val))
            if pref_cols:
                pref_ints = [int(x) for x in pref_cols]
                if tcol in pref_ints:
                    rank = pref_ints.index(tcol)
                    score -= 25.0 / (rank + 1.0)
        except Exception:
            pass

    # pack clustering penalty
    rep = rep_pack_by_col.get(tcol)
    if pack_val is not None and pd.notna(pack_val) and rep and float(rep) > 0:
        diff_ratio = abs(float(pack_val) - float(rep)) / float(rep)
        if diff_ratio > pack_tolerance_ratio:
            excess = diff_ratio - pack_tolerance_ratio
            score += 400.0 * (excess * excess + 1.0)
            score += 1.0
        else:
            score += 12.0 * (diff_ratio * 100.0)

    # same sku same column bonus
    if tcol == cur_col:
        score -= prefer_same_column_bonus

    # spreading penalty (new column for this sku)
    try:
        row_sku_cols = set([int(x) for x in [row.get("col")]])
        if int(tcol) not in row_sku_cols:
            score += split_sku_new_column_penalty
    except Exception:
        pass

    # バンド嗜好（入り数/販促で 35–41 or 1–11 へ寄せる）
    try:
        score += _band_preference(row, tcol, cfg or OptimizerConfig())
    except Exception:
        pass

    # ★ 取りやすさ（LV/COL/DEPが小さいほど良い）をスコア化
    score += _ease_key(tlv, tcol, tdep) * float(ease_weight)

    # 奥行きの嗜好：front（従来） or center（山形）
    try:
        pref = (cfg.depth_preference if cfg else "front")
    except Exception:
        pref = "front"

    if pref == "center":
        # 中心からの距離に比例してコストを加える（距離が小さいほど良い＝スコアが小さい）
        dist = _depth_center_distance(tdep, tcol, depths_by_col)
        w = float(getattr(cfg, "center_depth_weight", 1.0) if cfg else 1.0)
        score += dist * w
        # 旧来の浅さタイブレークは弱めに保つ（列内で同距離のとき手前を少し優先）
        score += (tdep * 0.0005)
    else:
        # 従来の浅い奥行きを優先（タイブレーク）
        score += (tdep * 0.001)
    return score

def _find_best_destination_for_row(
    row: pd.Series,
    inv: pd.DataFrame,
    shelf_usage: Dict[str, float],
    cap_limit: float,
    avoid_locs: Set[str],
    rep_pack_by_col: Dict[int, float],
    pack_tolerance_ratio: float,
    ai_col_hints: Optional[Dict[str, List[int]]],
    *,
    can_receive: Optional[Set[str]] = None,
    hard_pack_A: bool = False,
    ease_weight: float = 0.0001,
    cfg: OptimizerConfig | None = None,
) -> Optional[str]:
    """Choose a plausible destination location for the whole row."""
    from_loc = str(row["ロケーション"])
    best_to = None
    best_score = math.inf
    # 列ごとの最大奥行き本数と中心を事前計算
    depths_by_col: Dict[int, Tuple[int, float]] | None = None
    try:
        if cfg and getattr(cfg, "depth_preference", "front") == "center":
            # inv から当該列の dep の最大を求める（loc_master があれば理想は master 起点だが、在庫実績から近似）
            depths_by_col = {}
            if not inv.empty and "col" in inv.columns and "dep" in inv.columns:
                max_dep_by_col = inv.groupby("col")["dep"].max()
                for c, m in max_dep_by_col.items():
                    c_int = int(c)
                    max_dep = int(m)
                    if max_dep <= 0:
                        continue
                    # 中央：奇数→(max+1)/2、偶数→(max/2 + (max/2+1))/2 = max/2 + 0.5
                    if max_dep % 2 == 1:
                        center = (max_dep + 1) / 2.0
                    else:
                        center = (max_dep / 2.0 + (max_dep / 2.0 + 1.0)) / 2.0
                    depths_by_col[c_int] = (max_dep, float(center))
    except Exception:
        depths_by_col = None

    for to_loc in shelf_usage.keys():
        if to_loc in PLACEHOLDER_LOCS or to_loc == from_loc or to_loc in avoid_locs:
            continue
        if can_receive is not None and to_loc not in can_receive:
            continue
        tlv, tcol, tdep = _parse_loc8(str(to_loc))
        if tlv <= 0:
            continue
        # If strict pack-A is enforced at gating, avoid selecting out-of-band destinations here
        if hard_pack_A:
            rep = rep_pack_by_col.get(tcol)
            pack_val = row.get("pack_est")
            try:
                if (pack_val is not None) and pd.notna(pack_val) and rep and float(rep) > 0:
                    diff_ratio = abs(float(pack_val) - float(rep)) / float(rep)
                    if diff_ratio > pack_tolerance_ratio:
                        continue
            except Exception:
                pass
        # FIFO strict within same column only: if moving within same column, ensure not violating
        lot_key = int(row.get("lot_key") or UNKNOWN_LOT_KEY)
        if lot_key != UNKNOWN_LOT_KEY and tcol == int(row.get("col") or 0):
            # OPTIMIZED: Use global index instead of DataFrame filtering
            inv_lot_levels = _GLOBAL_INV_INDEXES.get("inv_lot_levels_by_sku_col", {})
            if inv_lot_levels:
                if _violates_lot_level_rule_fast(str(row["商品ID"]), lot_key, tlv, tcol, inv_lot_levels):
                    continue
            else:
                # Fallback to slow version if index not available
                if _violates_lot_level_rule(inv, str(row["商品ID"]), lot_key, tlv, tcol, tdep, None):
                    continue
        # score and pick（空きロケを優先）
        s = _score_destination_for_row(
            row,
            str(to_loc),
            rep_pack_by_col,
            pack_tolerance_ratio,
            20.0,  # same_sku_same_column_bonus (default)
            5.0,   # prefer_same_column_bonus (default)
            5.0,   # split_sku_new_column_penalty (default)
            ai_col_hints,
            ease_weight=ease_weight,
            depths_by_col=depths_by_col,
            cfg=cfg,
        )
        # 空きロケ（used==0）の場合、スコアを大幅に下げて優先
        used = float(shelf_usage.get(to_loc, 0.0))
        if used == 0.0:
            s -= 1000.0  # 空きロケ優先ボーナス
        if s < best_score:
            best_score = s
            best_to = str(to_loc)
    return best_to


# ========================================
# Parallel Candidate Evaluation Helper
# ========================================

@dataclass
class _CandidateEvalResult:
    """Result of evaluating a single candidate location."""
    to_loc: str
    target_level: int
    tcol: int
    tdep: int
    score: float
    area_needs_mix: bool
    candidate_ev_chain: List[Move]
    failure_reason: Optional[str] = None  # "capacity", "fifo", "pack_band", "forbidden", etc.
    eviction_chain_group_id: Optional[str] = None  # chain_group_id for eviction chain moves


def _evaluate_candidate_location(
    to_loc: str,
    target_level: int,
    row: pd.Series,
    from_loc: str,
    lv: int,
    cur_key: int,
    lot_key: int,
    need_vol: float,
    sku_val: str,
    qty_cases: int,
    idx: int,
    inv: pd.DataFrame,
    shelf_usage: Dict[str, float],
    cap_limit: float,
    rep_pack_by_col: Dict[int, float],
    mix_slots_left: Dict[int, int],
    planned_lots_by_loc_sku: Dict[Tuple[str, str], Set[int]],
    sku_vol_map: pd.Series,
    ai_col_hints: Optional[Dict[str, List[int]]],
    cap_by_loc: Optional[Dict[str, float]],
    can_receive_set: Optional[Set[str]],
    hard_pack_A: bool,
    depths_by_col_calc: Optional[Dict[int, Tuple[int, float]]],
    cfg: OptimizerConfig,
) -> Optional[_CandidateEvalResult]:
    """
    Evaluate a single candidate location for a row move.
    Returns _CandidateEvalResult if candidate is valid, None if rejected.
    This function is thread-safe for parallel execution.
    """
    tlv, tcol, tdep = _parse_loc8(to_loc)
    
    # Same-level move: must strictly improve ease
    if target_level == lv:
        new_key = _ease_key(tlv, tcol, tdep)
        if new_key >= cur_key:
            return None
    
    area_needs_mix = False
    
    # --- Area gating by pack/promo (strict_pack_area + mix枠/小入数1-2緩和)
    if getattr(cfg, "strict_pack_area", True):
        allowed_cols = _allowed_cols_for_row(row, cfg)
        # 入数1-2は緩和（別途まとめ運用のため）
        relax_smallpack = False
        try:
            relax_smallpack = float(row.get("pack_est") or 0) <= 2.0
        except Exception:
            relax_smallpack = False
        if (tcol not in allowed_cols) and not relax_smallpack:
            if mix_slots_left.get(tcol, 0) <= 0:
                return _CandidateEvalResult(
                    to_loc=to_loc, target_level=target_level, tcol=tcol, tdep=tdep,
                    score=math.inf, area_needs_mix=False, candidate_ev_chain=[],
                    failure_reason="pack_band"
                )
            area_needs_mix = True
    
    # Strict pack-A: skip out-of-band destinations entirely
    if hard_pack_A and (row.get("pack_est") is not None) and not pd.isna(row.get("pack_est")):
        rep = rep_pack_by_col.get(tcol)
        try:
            if rep and float(rep) > 0:
                diff_ratio = abs(float(row.get("pack_est")) - float(rep)) / float(rep)
                if diff_ratio > cfg.pack_tolerance_ratio:
                    return _CandidateEvalResult(
                        to_loc=to_loc, target_level=target_level, tcol=tcol, tdep=tdep,
                        score=math.inf, area_needs_mix=False, candidate_ev_chain=[],
                        failure_reason="pack_band"
                    )
        except Exception:
            pass
    
    # FIFO lot-level rule
    if _violates_lot_level_rule(inv, sku_val, lot_key, target_level, tcol, tdep, idx):
        return _CandidateEvalResult(
            to_loc=to_loc, target_level=target_level, tcol=tcol, tdep=tdep,
            score=math.inf, area_needs_mix=False, candidate_ev_chain=[],
            failure_reason="fifo"
        )
    
    # Hard rule: 同一SKUで異なるロットは同一ロケーションに置かない
    try:
        exists_mixed = False
        conflicting_lots: Set[int] = set()
        
        # 1) Check existing inventory in target location
        if ("商品ID" in inv.columns) and ("lot_key" in inv.columns):
            same_loc = inv[(inv["商品ID"].astype(str) == str(sku_val)) & (inv["ロケーション"].astype(str) == str(to_loc))]
            if not same_loc.empty:
                if int(lot_key) == UNKNOWN_LOT_KEY:
                    exists_mixed = True
                else:
                    kseries = pd.to_numeric(same_loc.get("lot_key"), errors="coerce").fillna(UNKNOWN_LOT_KEY).astype(int)
                    existing_lots = set(kseries.unique())
                    conflicting_lots.update(existing_lots)
                    if existing_lots and int(lot_key) not in existing_lots:
                        exists_mixed = True
        
        # 2) Check already planned moves to this location with same SKU
        lookup_key = (str(to_loc), str(sku_val))
        if lookup_key in planned_lots_by_loc_sku:
            planned_lots = planned_lots_by_loc_sku[lookup_key]
            if int(lot_key) == UNKNOWN_LOT_KEY:
                if planned_lots:
                    exists_mixed = True
            elif planned_lots and int(lot_key) not in planned_lots:
                exists_mixed = True
                conflicting_lots.update(planned_lots)
        
        if exists_mixed:
            return _CandidateEvalResult(
                to_loc=to_loc, target_level=target_level, tcol=tcol, tdep=tdep,
                score=math.inf, area_needs_mix=False, candidate_ev_chain=[],
                failure_reason="forbidden"
            )
    except Exception:
        pass
    
    # Capacity check and eviction chain planning
    candidate_ev_chain: List[Move] = []
    eviction_chain_group_id: Optional[str] = None
    used = float(shelf_usage.get(to_loc, 0.0))
    limit = cap_by_loc.get(to_loc, cap_limit) if cap_by_loc else cap_limit
    if used + need_vol > limit:
        # Try bounded eviction chain if enabled
        if getattr(cfg, "chain_depth", 0) and getattr(cfg, "eviction_budget", 0) and getattr(cfg, "touch_budget", 0):
            budget = _ChainBudget(
                depth_left=int(getattr(cfg, "chain_depth", 0)),
                evictions_left=int(getattr(cfg, "eviction_budget", 0)),
                touch_left=int(getattr(cfg, "touch_budget", 0)),
                touched=set([from_loc, to_loc]),
            )
            # NOTE: _plan_eviction_chain mutates inv and shelf_usage, so NOT thread-safe
            # For parallel execution, we need to skip eviction chain or make it read-only
            # For now, return failure if eviction needed in parallel mode
            if getattr(cfg, "enable_parallel", False):
                return _CandidateEvalResult(
                    to_loc=to_loc, target_level=target_level, tcol=tcol, tdep=tdep,
                    score=math.inf, area_needs_mix=False, candidate_ev_chain=[],
                    failure_reason="capacity"
                )
            # Generate chain_group_id for eviction chain
            eviction_chain_group_id = f"evict_{secrets.token_hex(6)}"
            
            # In serial mode, we can plan eviction (but this code path won't be used in parallel)
            candidate_ev_chain_result = _plan_eviction_chain(
                need_vol=need_vol,
                target_loc=str(to_loc),
                inv=inv,
                shelf_usage=shelf_usage,
                cap_limit=cap_limit,
                sku_vol_map=sku_vol_map,
                rep_pack_by_col=rep_pack_by_col,
                pack_tolerance_ratio=getattr(cfg, "pack_tolerance_ratio", 0.10),
                budget=budget,
                ai_col_hints=ai_col_hints,
                cap_by_loc=cap_by_loc or None,
                can_receive=can_receive_set or None,
                hard_pack_A=hard_pack_A,
                ease_weight=getattr(cfg, "ease_weight", 0.0001),
                cfg=cfg,
                chain_group_id=eviction_chain_group_id,
                execution_order_start=1,
            )
            if candidate_ev_chain_result is None:
                return _CandidateEvalResult(
                    to_loc=to_loc, target_level=target_level, tcol=tcol, tdep=tdep,
                    score=math.inf, area_needs_mix=False, candidate_ev_chain=[],
                    failure_reason="capacity"
                )
            candidate_ev_chain = candidate_ev_chain_result
        else:
            return _CandidateEvalResult(
                to_loc=to_loc, target_level=target_level, tcol=tcol, tdep=tdep,
                score=math.inf, area_needs_mix=False, candidate_ev_chain=[],
                failure_reason="capacity"
            )
    
    # Score this destination choice
    score = _score_destination_for_row(
        row,
        str(to_loc),
        rep_pack_by_col,
        getattr(cfg, "pack_tolerance_ratio", 0.10),
        getattr(cfg, "same_sku_same_column_bonus", 20.0),
        getattr(cfg, "prefer_same_column_bonus", 5.0),
        getattr(cfg, "split_sku_new_column_penalty", 5.0),
        ai_col_hints,
        ease_weight=getattr(cfg, "ease_weight", 0.0001),
        depths_by_col=depths_by_col_calc,
        cfg=cfg,
    )
    
    # Soft preference: 同一SKU・同一ロットが既に存在する列ならボーナス
    try:
        same_lot_cols = set(
            inv.loc[
                (inv["商品ID"].astype(str) == str(sku_val)) & 
                (pd.to_numeric(inv.get("lot_key"), errors="coerce").fillna(UNKNOWN_LOT_KEY).astype(int) == int(lot_key)),
                "col",
            ].dropna().astype(int).unique().tolist()
        )
        if int(tcol) in same_lot_cols:
            score -= float(getattr(cfg, "same_lot_same_column_bonus", 10.0))
    except Exception:
        pass
    
    return _CandidateEvalResult(
        to_loc=to_loc,
        target_level=target_level,
        tcol=tcol,
        tdep=tdep,
        score=score,
        area_needs_mix=area_needs_mix,
        candidate_ev_chain=candidate_ev_chain,
        failure_reason=None,
        eviction_chain_group_id=eviction_chain_group_id,
    )


def _plan_eviction_chain(
    need_vol: float,
    target_loc: str,
    inv: pd.DataFrame,
    shelf_usage: Dict[str, float],
    cap_limit: float,
    sku_vol_map: pd.Series,
    rep_pack_by_col: Dict[int, float],
    pack_tolerance_ratio: float,
    budget: _ChainBudget,
    ai_col_hints: Optional[Dict[str, List[int]]] = None,
    *,
    cap_by_loc: Optional[Dict[str, float]] = None,
    can_receive: Optional[Set[str]] = None,
    hard_pack_A: bool = False,
    ease_weight: float = 0.0001,
    cfg: OptimizerConfig | None = None,
    chain_group_id: Optional[str] = None,
    execution_order_start: int = 1,
) -> Optional[List[Move]]:
    """Try to free `need_vol` capacity on `target_loc` by evicting whole rows.
    Returns a list of eviction moves in execution order if successful; otherwise None.
    This mutates `inv` and `shelf_usage` when it succeeds.
    """
    # Quick check
    used = float(shelf_usage.get(target_loc, 0.0))
    limit = cap_by_loc.get(target_loc, cap_limit) if cap_by_loc else cap_limit
    free = limit - used
    if free >= need_vol:
        return []
    if budget.depth_left <= 0 or budget.evictions_left <= 0 or budget.touch_left <= 0:
        return None

    # OPTIMIZED: Use global index to get rows at target location instead of DataFrame filtering
    inv_rows_by_loc = _GLOBAL_INV_INDEXES.get("inv_rows_by_loc", {})
    row_indices = inv_rows_by_loc.get(str(target_loc), [])
    if not row_indices:
        # Fallback to DataFrame filtering if index not available
        in_rows = inv[inv["ロケーション"].astype(str) == str(target_loc)].copy()
    else:
        # Fast path: use pre-built index
        in_rows = inv.loc[row_indices].copy()
    
    if in_rows.empty:
        return None
    def _row_sort_key(r: pd.Series) -> Tuple[float, int, int]:
        # larger pack diff first, newer lot first, higher level first (easier to move)
        rep = rep_pack_by_col.get(int(r.get("col") or 0))
        pack_val = r.get("pack_est")
        diff = 0.0
        try:
            if pack_val is not None and pd.notna(pack_val) and rep and float(rep) > 0:
                diff = abs(float(pack_val) - float(rep)) / float(rep)
        except Exception:
            diff = 0.0
        lotk = int(r.get("lot_key") or 0)
        lvl = int(r.get("lv") or 9)
        return (-(diff), -lotk, -lvl)

    cand_rows = list(in_rows.sort_values(by=["lot_key"]).itertuples(index=False))
    # Convert namedtuples back to Series-compatible by fetching with original DataFrame when used
    # We'll iterate with original DataFrame rows for safety
    in_rows_sorted = in_rows.sort_values(by=["lot_key", "lv"], ascending=[False, False])

    chain: List[Move] = []
    tried = 0
    max_rows = int(getattr(cfg, "max_chain_rows_per_target", 12))
    max_iterations = max(max_rows, 50)  # 安全上限
    iteration_count = 0
    for _, row in in_rows_sorted.iterrows():
        iteration_count += 1
        if iteration_count > max_iterations:
            print(f"[optimizer] _plan_eviction_chain: 最大反復回数({max_iterations})到達 - ループ中断")
            break
        tried += 1
        if max_rows > 0 and tried > max_rows:
            break
        vol_move = float(row.get("volume_total") or 0.0)
        if vol_move <= 0:
            continue
        dest = _find_best_destination_for_row(
            row,
            inv,
            shelf_usage,
            cap_limit,
            avoid_locs=budget.touched,
            rep_pack_by_col=rep_pack_by_col,
            pack_tolerance_ratio=pack_tolerance_ratio,
            ai_col_hints=ai_col_hints,
            can_receive=can_receive,
            hard_pack_A=hard_pack_A,
            ease_weight=ease_weight,
            cfg=cfg,
        )
        if not dest:
            continue

        # If destination lacks capacity, recursively free it first
        used_d = float(shelf_usage.get(dest, 0.0))
        limit_d = cap_by_loc.get(dest, cap_limit) if cap_by_loc else cap_limit
        need_d = max(0.0, used_d + vol_move - limit_d)
        submoves: List[Move] = []
        if need_d > 0:
            sub_budget = _ChainBudget(
                depth_left=budget.depth_left - 1,
                evictions_left=budget.evictions_left,
                touch_left=budget.touch_left,
                touched=set(budget.touched),
                max_depth_used=budget.max_depth_used,
            )
            sub = _plan_eviction_chain(
                need_vol=need_d,
                target_loc=str(dest),
                inv=inv,
                shelf_usage=shelf_usage,
                cap_limit=cap_limit,
                sku_vol_map=sku_vol_map,
                rep_pack_by_col=rep_pack_by_col,
                pack_tolerance_ratio=pack_tolerance_ratio,
                budget=sub_budget,
                ai_col_hints=ai_col_hints,
                cap_by_loc=cap_by_loc,
                can_receive=can_receive,
                hard_pack_A=hard_pack_A,
                ease_weight=ease_weight,
                cfg=cfg,
                chain_group_id=chain_group_id,
                execution_order_start=execution_order_start + len(chain),
            )
            if sub is None:
                continue
            submoves.extend(sub)
            budget.max_depth_used = max(budget.max_depth_used, sub_budget.max_depth_used)
            budget.evictions_left = sub_budget.evictions_left
            budget.touch_left = sub_budget.touch_left
            budget.touched = set(sub_budget.touched)

        # Now move this row to dest
        sku = str(row["商品ID"])
        lot_str = str(row.get("ロット") or "")
        lk = int(row.get("lot_key") or UNKNOWN_LOT_KEY)
        qty_cases = int(row.get("qty_cases_move") or 0)
        if qty_cases <= 0:
            continue
        
        # Calculate execution_order based on current chain length + submoves
        current_exec_order = execution_order_start + len(chain) + len(submoves)
        
        mv = Move(
            sku_id=sku,
            lot=lot_str,
            qty=qty_cases,
            from_loc=str(row["ロケーション"]).zfill(8),
            to_loc=str(dest).zfill(8),
            lot_date=_lot_key_to_datestr8(lk),
            reason=f"容量確保退避(連鎖深度{budget.max_depth_used}) → スペース創出・効率配置",
            chain_group_id=chain_group_id,
            execution_order=current_exec_order if chain_group_id else None,
        )
        # apply state updates
        shelf_usage[str(row["ロケーション"])] = max(0.0, shelf_usage.get(str(row["ロケーション"]), 0.0) - vol_move)
        shelf_usage[str(dest)] = shelf_usage.get(str(dest), 0.0) + vol_move

        # reflect into inv (update row's location/coordinates)
        tlv, tcol, tdep = _parse_loc8(str(dest))
        idxs = inv.index[(inv["ロケーション"].astype(str) == str(row["ロケーション"])) &
                         (inv["商品ID"].astype(str) == sku) &
                         (inv["ロット"].astype(str) == lot_str)]
        if len(idxs) > 0:
            ridx = idxs[0]
            inv.at[ridx, "ロケーション"] = str(dest)
            inv.at[ridx, "lv"] = tlv
            inv.at[ridx, "col"] = tcol
            inv.at[ridx, "dep"] = tdep

        chain.extend(submoves)
        chain.append(mv)

        budget.evictions_left -= 1
        if str(row["ロケーション"]) not in budget.touched:
            budget.touch_left -= 1
            budget.touched.add(str(row["ロケーション"]))
        if str(dest) not in budget.touched:
            budget.touch_left -= 1
            budget.touched.add(str(dest))
        budget.max_depth_used = max(budget.max_depth_used, 1)

        # check if we've freed enough on target
        used_now = float(shelf_usage.get(target_loc, 0.0))
        limit_now = cap_by_loc.get(target_loc, cap_limit) if cap_by_loc else cap_limit
        free_now = limit_now - used_now
        if free_now >= need_vol:
            return chain

        # else continue trying more rows in target_loc
    return None


# -------------------------------
# Pass-0: Area (column band) rebalancing
# -------------------------------

def _pass0_area_rebalance(
    inv: pd.DataFrame,
    shelf_usage: Dict[str, float],
    cap_limit: float,
    cfg: OptimizerConfig,
    rep_pack_by_col: Dict[int, float],
    pack_tolerance_ratio: float,
    sku_vol_map: pd.Series,
    ai_col_hints: Optional[Dict[str, List[int]]] = None,
    *,
    cap_by_loc: Optional[Dict[str, float]] = None,
    can_receive: Optional[Set[str]] = None,
    budget_left: Optional[int] = None,
) -> List[Move]:
    """ゾーニング（入数/販促に基づく列レンジ）に**明確に外れている**在庫を、
    まず『許容列セット』へ**列を跨いで**寄せる前処理。容量不足なら有界チェーンで空ける。
    - Pass-1（同列段整列）より**前**に呼ぶ。
    - 入数<=2 の緩和は `pass0_respect_smallpack_relax` で制御。
    """
    if not getattr(cfg, "enable_pass0_area_rebalance", True):
        return []
    if not getattr(cfg, "strict_pack_area", True):
        # エリアゲート運用でない場合は何もしない
        return []

    moves: List[Move] = []
    def _can_add(k: int) -> bool:
        return (budget_left is None) or (len(moves) + int(k) <= int(budget_left))
    if inv.empty:
        return moves

    # 対象抽出：容積>0・移動ケース>0・lot解析済み（厳密でなくてもOK）
    cand = inv[(inv["volume_each_case"] > 0) & (inv["qty_cases_move"] > 0)].copy()
    if cand.empty:
        return moves

    # 走査順：『帯外度』が高いものを先に（代表入数との差/ルール違反の明確さで近似）
    def _misalign_score(row: pd.Series) -> float:
        try:
            allowed = _allowed_cols_for_row(row, cfg)
            cur = int(row.get("col") or 0)
            in_allowed = cur in allowed
            pack_val = float(row.get("pack_est")) if pd.notna(row.get("pack_est")) else float("nan")
            rep = rep_pack_by_col.get(cur)
            diff = 0.0
            if pd.notna(pack_val) and rep and float(rep) > 0:
                diff = abs(float(pack_val) - float(rep)) / float(rep)
            base = 1.0 if not in_allowed else 0.0
            return base * (1.0 + diff)
        except Exception:
            return 0.0

    # ベクトル化: apply(axis=1)を避けて高速化
    try:
        cand["__misalign"] = cand.apply(_misalign_score, axis=1)
    except Exception:
        cand["__misalign"] = 0.0
    cand = cand.sort_values(["__misalign", "lot_key"], ascending=[False, True])

    # レベルの優先順：現在段→設定の pass0_target_levels 順
    # パフォーマンス最適化：レベル計算結果をキャッシュ
    _level_cache: Dict[int, List[int]] = {}
    def _level_order(cur_lv: int) -> List[int]:
        if cur_lv not in _level_cache:
            lv_seq = list(dict.fromkeys([int(cur_lv)] + list(getattr(cfg, "pass0_target_levels", (1,2,3,4)))))
            _level_cache[cur_lv] = [lv for lv in lv_seq if 1 <= int(lv) <= 4]
        return _level_cache[cur_lv]

    # 位置座標の事前計算とインデックス（SKU×列の探索を高速化）
    try:
        _coords_map: Dict[str, Tuple[int,int,int]] = {
            str(loc): _parse_loc8(str(loc)) for loc in shelf_usage.keys()
        }
        _locs_by_col: Dict[int, Set[str]] = {}
        _locs_by_level: Dict[int, Set[str]] = {}
        for loc, (lv, col, dep) in _coords_map.items():
            try:
                _locs_by_col.setdefault(int(col), set()).add(str(loc))
                _locs_by_level.setdefault(int(lv), set()).add(str(loc))
            except Exception:
                pass
    except Exception:
        _coords_map = {str(loc): _parse_loc8(str(loc)) for loc in shelf_usage.keys()}
        _locs_by_col, _locs_by_level = {}, {}

    # 1件ずつ帯外を是正
    max_pass0_iterations = len(cand) * 2  # 候補数の2倍を上限
    iteration_count = 0
    for idx, row in cand.iterrows():
        iteration_count += 1
        if iteration_count > max_pass0_iterations:
            print(f"[optimizer] pass0_area_rebalance: 最大反復回数({max_pass0_iterations})到達 - Pass-0完了")
            break
        try:
            cur_col = int(row.get("col") or 0)
            cur_lv  = int(row.get("lv") or 0)
            from_loc = str(row.get("ロケーション"))
            if not from_loc or from_loc in PLACEHOLDER_LOCS:
                continue
            # 緩和: 入数<=2 は対象外（運用で別まとめのため）
            if getattr(cfg, "pass0_respect_smallpack_relax", True):
                try:
                    pack_est = float(row.get("pack_est") or 0)
                    if pack_est <= 2.0:
                        continue
                except Exception:
                    pass

            allowed_cols = _allowed_cols_for_row(row, cfg)
            if cur_col in allowed_cols:
                continue  # 既に許容帯

            qty_cases = int(row.get("qty_cases_move") or 0)
            if qty_cases <= 0:
                continue
            add_each = float(row.get("volume_each_case") or 0.0)
            if add_each <= 0:
                continue
            need_vol = add_each * qty_cases

            # パフォーマンス最適化：適格ロケーションをキャッシュから合成
            allowed_cols_set = set(allowed_cols)
            allowed_levels_set = set(_level_order(cur_lv))
            # 列で広げ、段で絞る
            by_col: Set[str] = set()
            for c in allowed_cols_set:
                by_col |= _locs_by_col.get(int(c), set())
            by_lvl: Set[str] = set()
            for lv in allowed_levels_set:
                by_lvl |= _locs_by_level.get(int(lv), set())
            eligible_locations = (by_col & by_lvl)
            # 追加フィルタ（プレースホルダ・送元・受入可否）
            if from_loc:
                if can_receive is None:
                    eligible_locations = {loc for loc in eligible_locations if loc not in PLACEHOLDER_LOCS and loc != from_loc}
                else:
                    eligible_locations = {loc for loc in eligible_locations if loc not in PLACEHOLDER_LOCS and loc != from_loc and loc in can_receive}
            else:
                if can_receive is not None:
                    eligible_locations = {loc for loc in eligible_locations if loc in can_receive and loc not in PLACEHOLDER_LOCS}
                else:
                    eligible_locations = {loc for loc in eligible_locations if loc not in PLACEHOLDER_LOCS}

            best_choice: Optional[Tuple[str, int, int, int]] = None
            best_score = math.inf
            best_ev_chain: List[Move] = []
            best_ev_chain_group_id: Optional[str] = None

            # 候補ロケ:事前フィルタ済みのみをループ
            # パフォーマンス最適化: 候補が多すぎる場合は容量空き順でトップN件のみ評価
            eligible_count = len(eligible_locations)
            max_candidates_to_check = int(getattr(cfg, "pass0_max_candidates_per_item", 50))
            if eligible_count > max_candidates_to_check:
                # 容量の空きが多い順にソート
                sorted_locs = sorted(
                    eligible_locations,
                    key=lambda loc: (cap_by_loc.get(loc, cap_limit) if cap_by_loc else cap_limit) - shelf_usage.get(loc, 0.0),
                    reverse=True
                )
                eligible_locations = set(sorted_locs[:max_candidates_to_check])
                eligible_count = len(eligible_locations)
            
            checked_count = 0
            for to_loc in eligible_locations:
                checked_count += 1
                # 進捗ログ(10%ごと、または最初の10件)
                if eligible_count > 100 and checked_count % max(1, eligible_count // 10) == 0:
                    print(f"[optimizer] pass0 item {iteration_count}/{len(cand)}: 候補ロケ探索中 {checked_count}/{eligible_count}")
                elif checked_count <= 10:
                    pass  # 最初の10件は静か
                
                tlv, tcol, tdep = _coords_map.get(str(to_loc), _parse_loc8(str(to_loc)))

                # パフォーマンス最適化：容量チェックを早期実行
                used = float(shelf_usage.get(to_loc, 0.0))
                limit = cap_by_loc.get(to_loc, cap_limit) if cap_by_loc else cap_limit
                needs_eviction = used + need_vol > limit
                
                # チェーン許可がない場合は容量不足で除外
                if needs_eviction and not (getattr(cfg, "chain_depth", 0) and getattr(cfg, "eviction_budget", 0) and getattr(cfg, "touch_budget", 0)):
                    continue

                # FIFO：同一列内のみ厳密（別列へは制限なし）
                if int(tcol) == cur_col:
                    lot_key = int(row.get("lot_key") or UNKNOWN_LOT_KEY)
                    if lot_key != UNKNOWN_LOT_KEY:
                        if _violates_lot_level_rule(inv, str(row["商品ID"]), lot_key, tlv, tcol, tdep, idx):
                            continue

                candidate_ev_chain: List[Move] = []
                candidate_ev_chain_group_id: Optional[str] = None
                # 容量不足時のチェーン処理
                if needs_eviction:
                    # 有界チェーンで空ける
                    budget = _ChainBudget(
                        depth_left=int(getattr(cfg, "chain_depth", 0)),
                        evictions_left=int(getattr(cfg, "eviction_budget", 0)),
                        touch_left=int(getattr(cfg, "touch_budget", 0)),
                        touched=set([from_loc, to_loc]),
                    )
                    # Generate chain_group_id for eviction chain
                    candidate_ev_chain_group_id = f"evict_{secrets.token_hex(6)}"
                    
                    chain = _plan_eviction_chain(
                        need_vol=need_vol,
                        target_loc=str(to_loc),
                        inv=inv,
                        shelf_usage=shelf_usage,
                        cap_limit=cap_limit,
                        sku_vol_map=sku_vol_map,
                        rep_pack_by_col=rep_pack_by_col,
                        pack_tolerance_ratio=pack_tolerance_ratio,
                        budget=budget,
                        ai_col_hints=ai_col_hints,
                        cap_by_loc=cap_by_loc,
                        can_receive=can_receive,
                        hard_pack_A=False,
                        ease_weight=getattr(cfg, "ease_weight", 0.0001),
                        cfg=cfg,
                        chain_group_id=candidate_ev_chain_group_id,
                        execution_order_start=1,
                    )
                    if chain is None:
                        continue
                    candidate_ev_chain = chain

                # スコアリング（既存の評価関数を再利用）
                s = _score_destination_for_row(
                    row,
                    str(to_loc),
                    rep_pack_by_col,
                    pack_tolerance_ratio,
                    getattr(cfg, "same_sku_same_column_bonus", 20.0),
                    getattr(cfg, "prefer_same_column_bonus", 5.0),
                    getattr(cfg, "split_sku_new_column_penalty", 5.0),
                    ai_col_hints,
                    ease_weight=getattr(cfg, "ease_weight", 0.0001),
                    depths_by_col=None,  # pass-1 detailed loop uses center pref implicitly via ease/tie if needed
                    cfg=cfg,
                )
                # 帯適合には小ボーナス
                s -= 5.0

                if s < best_score:
                    best_score = s
                    best_choice = (str(to_loc), tlv, tcol, tdep)
                    best_ev_chain = candidate_ev_chain
                    best_ev_chain_group_id = candidate_ev_chain_group_id

            if best_choice is None:
                continue

            # 事前に必要な手数＋本体1手が予算内か
            if not _can_add(len(best_ev_chain) + 1):
                break

            # 事前に用意した eviction chain を先に反映
            if best_ev_chain:
                for mv in best_ev_chain:
                    moves.append(mv)
                    # state apply
                    add_each_o = float(sku_vol_map.get(str(mv.sku_id), 0.0) or 0.0)
                    add_vol_o = add_each_o * float(mv.qty)
                    shelf_usage[str(mv.from_loc)] = max(0.0, shelf_usage.get(str(mv.from_loc), 0.0) - add_vol_o)
                    shelf_usage[str(mv.to_loc)]   = shelf_usage.get(str(mv.to_loc), 0.0) + add_vol_o
                    # reflect in inv
                    tlv_o, tcol_o, tdep_o = _parse_loc8(str(mv.to_loc))
                    idxs = inv.index[(inv["ロケーション"].astype(str) == str(mv.from_loc)) &
                                     (inv["商品ID"].astype(str) == str(mv.sku_id)) &
                                     (inv["ロット"].astype(str) == str(mv.lot))]
                    if len(idxs) > 0:
                        ridx = idxs[0]
                        inv.at[ridx, "ロケーション"] = str(mv.to_loc)
                        inv.at[ridx, "lv"] = tlv_o
                        inv.at[ridx, "col"] = tcol_o
                        inv.at[ridx, "dep"] = tdep_o

            # 本体 move
            if not _can_add(1):
                break
            to_loc, tlv, tcol, tdep = best_choice
            lk = int(row.get("lot_key") or UNKNOWN_LOT_KEY)
            
            # Determine reason for Pass-0 moves (area rebalancing & pack size correction)
            from_lv, from_col, from_dep = _parse_loc8(from_loc)
            
            actions = []
            improvements = []
            
            # Check pack size zone correction
            pack_corrected = False
            try:
                pack_est = float(row.get("pack_est") or 0)
                rep_pack = rep_pack_by_col.get(from_col, 0)
                target_rep_pack = rep_pack_by_col.get(tcol, 0)
                if pack_est > 0 and target_rep_pack > 0 and rep_pack > 0:
                    # Check if moving to more appropriate pack size zone
                    current_diff = abs(pack_est - rep_pack) / max(pack_est, rep_pack)
                    target_diff = abs(pack_est - target_rep_pack) / max(pack_est, target_rep_pack)
                    if target_diff < current_diff:
                        actions.append(f"入数帯是正(列{from_col}→{tcol})")
                        improvements.append(f"適正入数帯へ({pack_est:.0f}入)")
                        pack_corrected = True
            except Exception:
                pass
            
            # Column movement (entrance proximity or area optimization)
            if tcol != from_col and not pack_corrected:
                if tcol > from_col:
                    actions.append(f"入口接近(列{from_col}→{tcol})")
                    improvements.append("動線短縮")
                else:
                    actions.append(f"エリア移動(列{from_col}→{tcol})")
                    improvements.append("エリア内バランス")
            
            # Level change
            if tlv != from_lv:
                if tlv < from_lv:
                    actions.append(f"段下げ(Lv{from_lv}→{tlv})")
                    improvements.append("下段活用")
                else:
                    actions.append(f"段上げ(Lv{from_lv}→{tlv})")
                    improvements.append("空間効率化")
            
            if not actions:
                actions.append("位置最適化")
                improvements.append("エリア内バランス改善")
            
            move_reason = " & ".join(actions)
            if improvements:
                move_reason += " → " + "、".join(improvements)
            
            # Calculate execution_order for main move (after eviction chain)
            main_exec_order = len(best_ev_chain) + 1 if best_ev_chain_group_id else None
            
            mv = Move(
                sku_id=str(row["商品ID"]),
                lot=str(row.get("ロット") or ""),
                qty=int(row.get("qty_cases_move") or 0),
                from_loc=str(from_loc).zfill(8),
                to_loc=str(to_loc).zfill(8),
                lot_date=_lot_key_to_datestr8(lk),
                reason=move_reason,
                chain_group_id=best_ev_chain_group_id or f"p0rebal_{secrets.token_hex(6)}",
                execution_order=main_exec_order or 1,
            )
            moves.append(mv)
            # apply state
            shelf_usage[from_loc] = max(0.0, shelf_usage.get(from_loc, 0.0) - need_vol)
            shelf_usage[str(to_loc)] = shelf_usage.get(str(to_loc), 0.0) + need_vol
            inv.at[idx, "ロケーション"] = str(to_loc)
            inv.at[idx, "lv"] = int(tlv)
            inv.at[idx, "col"] = int(tcol)
            inv.at[idx, "dep"] = int(tdep)
        except Exception:
            continue

    if moves:
        try:
            print(f"[optimizer] pass0_area_rebalance moves={len(moves)}")
        except Exception:
            pass
    return moves


# -------------------------------
# Core public APIs
# -------------------------------

def plan_relocation(
    sku_master: pd.DataFrame,
    inventory: pd.DataFrame,
    *,
    cfg: OptimizerConfig | None = None,
    block_filter: Iterable[str] | None = None,
    quality_filter: Iterable[str] | None = None,
    ai_col_hints: dict[str, list[int]] | None = None,
    loc_master: Optional[pd.DataFrame] = None,
) -> List[Move]:
    """
    Greedy 手法でリロケーション案を作成して返す。

    Parameters
    ----------
    sku_master : DataFrame
        SKU マスター。'carton_volume_m3' を含むか、式で導出可能であること。
        SKU キーは 'sku_id' または '商品ID' を使用。
    inventory : DataFrame
        現在庫。必要列: 'ブロック略称', 'ロケーション', '商品ID', 'ロット',
        数量は 'ケース' を優先、無ければ '在庫数(引当数を含む)' を '入数' でケース換算。
    block_filter : Iterable[str], optional
        対象ブロックを限定する場合に指定。
    quality_filter : Iterable[str], optional
        在庫を品質区分で限定する場合に指定（例: ["良品","販促物/什器"]）。
    ai_col_hints : dict[str, list[int]], optional
        AI が提案した「SKUごとの優先列リスト」。例: {"A123":[12,11,13], ...}
        スコアリング時に、該当列に置く場合は順位に応じてボーナス（減点）を与える。
    Returns
    -------
    list[Move]
        移動指示のリスト（順序 = 実行順）。
    """
    cfg = cfg or OptimizerConfig()
    # Bind or initialize trace id so all drops are recorded under the caller-provided id
    tid = getattr(cfg, "trace_id", None)
    if tid:
        try:
            start_drop_trace(str(tid))  # clear previous buffer and bind
        except Exception:
            bind_trace_id(str(tid))     # fallback: just bind
    else:
        # Ensure some id is bound so that drop logs are not lost
        if not _CURRENT_TRACE_ID:
            try:
                start_drop_trace(secrets.token_hex(6))
            except Exception:
                _auto_bind_trace_if_needed(None)
    cap_limit = _capacity_limit(getattr(cfg, "fill_rate", None))
    try:
        _publish_progress(get_current_trace_id(), {
            "type": "start", "phase": "plan", "cfg": {
                "max_moves": getattr(cfg, "max_moves", None),
                "fill_rate": getattr(cfg, "fill_rate", None),
                "chain_depth": getattr(cfg, "chain_depth", None),
                "eviction_budget": getattr(cfg, "eviction_budget", None),
                "touch_budget": getattr(cfg, "touch_budget", None),
            }
        })
    except Exception:
        pass

    cap_by_loc: Dict[str, float] = {}
    can_receive_set: Set[str] = set()
    if loc_master is not None and not loc_master.empty:
        # Narrow candidate destinations to the specified blocks/qualities if columns exist
        orig_slots = len(loc_master)
        lm_scoped = _filter_loc_master_by_block_quality(loc_master, block_filter, quality_filter)
        cap_by_loc, can_receive_set = _cap_map_from_master(lm_scoped, getattr(cfg, "fill_rate", DEFAULT_FILL_RATE))
        print(f"[optimizer] location_master provided: slots={len(cap_by_loc)} receivable={len(can_receive_set)} (filtered from {orig_slots})")
        try:
            _publish_progress(get_current_trace_id(), {
                "type": "info", "phase": "init", 
                "message": f"ロケーションマスター読込: {len(cap_by_loc)}ロケ (受入可能: {len(can_receive_set)})"
            })
        except Exception:
            pass
    else:
        print("[optimizer] location_master not provided; using global capacity")
        try:
            _publish_progress(get_current_trace_id(), {
                "type": "info", "phase": "init",
                "message": "ロケーションマスター未指定、グローバル容量を使用"
            })
        except Exception:
            pass

    # --- preflight: 必須列チェック & ログ
    print(f"[optimizer] sku_master cols={list(sku_master.columns)} rows={len(sku_master)}")
    if ("ブロック略称" not in inventory.columns) and ("block_code" not in inventory.columns):
        raise ValueError("inventory に ブロック略称 / block_code 列が必要です")
    _require_cols(inventory, ["ロケーション", "商品ID", "ロット"], "inventory")
    if quality_filter is not None:
        if ("quality_name" not in inventory.columns) and ("品質区分名" not in inventory.columns):
            raise ValueError("inventory に 品質区分名 / quality_name 列が必要です（quality_filter 指定あり）")

    inv = inventory.copy()

    # 列名の正規化
    if "ブロック略称" not in inv.columns and "block_code" in inv.columns:
        inv["ブロック略称"] = inv["block_code"]
    if "quality_name" not in inv.columns and "品質区分名" in inv.columns:
        inv["quality_name"] = inv["品質区分名"].astype(str)

    # フィルタ適用
    if block_filter is not None:
        inv = inv[inv["ブロック略称"].isin(list(block_filter))].copy()
    print(f"[optimizer] after block_filter={list(block_filter) if block_filter is not None else None} rows={len(inv)}")
    try:
        _publish_progress(get_current_trace_id(), {
            "type": "info", "phase": "filter",
            "message": f"ブロックフィルター適用後: {len(inv)}行"
        })
    except Exception:
        pass

    if quality_filter is not None:
        qset = set(str(x) for x in quality_filter)
        inv = inv[inv["quality_name"].astype(str).isin(qset)].copy()
    print(f"[optimizer] after quality_filter={list(quality_filter) if quality_filter is not None else None} rows={len(inv)}")
    try:
        _publish_progress(get_current_trace_id(), {
            "type": "info", "phase": "filter",
            "message": f"品質フィルター適用後: {len(inv)}行"
        })
    except Exception:
        pass

    # プレースホルダロケは移動元として扱わない
    inv = inv[~inv["ロケーション"].astype(str).isin(PLACEHOLDER_LOCS)].copy()

    # ロケマスタに存在するロケーションのみを移動対象とする
    if cap_by_loc:
        valid_locs_set = set(cap_by_loc.keys())
        before_count = len(inv)
        inv = inv[inv["ロケーション"].astype(str).str.zfill(8).isin(valid_locs_set)].copy()
        excluded_count = before_count - len(inv)
        print(f"[optimizer] ロケマスタ存在チェック: {excluded_count}件除外 → 残り{len(inv)}件")
        try:
            _publish_progress(get_current_trace_id(), {
                "type": "info", "phase": "filter",
                "message": f"ロケマスタ存在チェック: {excluded_count}件除外 → 残り{len(inv)}件"
            })
        except Exception:
            pass

    # 引当在庫は移動対象外: '引当数' が存在し、かつ >0 の行は除外
    try:
        if "引当数" in inv.columns:
            inv = inv[pd.to_numeric(inv["引当数"], errors="coerce").fillna(0) <= 0].copy()
    except Exception:
        # 列がない/変換不可などは無視（従来通り全件対象）
        pass

    # --- SKU 段ボール容積マップ（m³/ケース）
    sku_vol_map = _build_carton_volume_map(sku_master)
    print(f"[optimizer] carton_volume_map size={sku_vol_map.shape[0]}")
    try:
        _publish_progress(get_current_trace_id(), {
            "type": "info", "phase": "setup",
            "message": f"SKU容積マップ作成: {sku_vol_map.shape[0]}件"
        })
    except Exception:
        pass

    # --- 入数マップ（ケース換算に使用）
    pack_map = None
    if "入数" in sku_master.columns:
        key = "sku_id" if "sku_id" in sku_master.columns else "商品ID"
        pack_map = sku_master.set_index(key)["入数"].astype(float)

    # --- lot key & 現在のロケ座標（後続の段ルールで使用）
    inv["lot_key"] = inv["ロット"].map(_parse_lot_date_key)
    
    # Debug: Check location format before parsing
    sample_locs = inv["ロケーション"].head(5).tolist()
    print(f"[optimizer] Sample locations before parse: {sample_locs}")
    print(f"[optimizer] Location dtype: {inv['ロケーション'].dtype}")
    
    # Ensure locations are zero-padded 8-digit strings
    inv["ロケーション"] = inv["ロケーション"].astype(str).str.replace('.0', '').str.zfill(8)
    sample_locs_after = inv["ロケーション"].head(5).tolist()
    print(f"[optimizer] Sample locations after parse: {sample_locs_after}")
    
    # --- 複数SKU混在ロケーションを除外（移動元・移動先両方から除外）
    # ロケーションごとのユニークSKU数をカウント
    sku_count_per_loc = inv.groupby("ロケーション")["商品ID"].nunique()
    multi_sku_locs = set(sku_count_per_loc[sku_count_per_loc > 1].index)
    
    if multi_sku_locs:
        before_inv_count = len(inv)
        # 移動元から除外: 複数SKU混在ロケにある在庫は移動しない
        inv = inv[~inv["ロケーション"].isin(multi_sku_locs)].copy()
        excluded_inv_count = before_inv_count - len(inv)
        
        # 移動先から除外: can_receive_setから複数SKU混在ロケを削除
        if can_receive_set:
            before_recv_count = len(can_receive_set)
            can_receive_set = can_receive_set - multi_sku_locs
            excluded_recv_count = before_recv_count - len(can_receive_set)
        else:
            excluded_recv_count = 0
        
        print(f"[optimizer] 複数SKU混在ロケ除外: {len(multi_sku_locs)}ロケ → 移動元から{excluded_inv_count}件除外, 移動先から{excluded_recv_count}ロケ除外")
        try:
            _publish_progress(get_current_trace_id(), {
                "type": "info", "phase": "filter",
                "message": f"複数SKU混在ロケ除外: {len(multi_sku_locs)}ロケ → 移動元{excluded_inv_count}件, 移動先{excluded_recv_count}ロケ除外"
            })
        except Exception:
            pass
    else:
        print("[optimizer] 複数SKU混在ロケなし")

    # ベクトル化: apply不使用で高速化
    loc_str = inv["ロケーション"].astype(str).str.zfill(8)
    inv["lv"] = pd.to_numeric(loc_str.str[0:3], errors='coerce').fillna(0).astype(int)
    inv["col"] = pd.to_numeric(loc_str.str[3:6], errors='coerce').fillna(0).astype(int)
    inv["dep"] = pd.to_numeric(loc_str.str[6:8], errors='coerce').fillna(0).astype(int)
    # 山形（中心優先）用の列→(max_dep, center) マップ（在庫実績から推定）
    depths_by_col_calc: Dict[int, Tuple[int, float]] | None = None
    try:
        if getattr(cfg, "depth_preference", "front") == "center":
            depths_by_col_calc = {}
            if not inv.empty:
                max_dep_by_col = inv.groupby("col")["dep"].max()
                for c, m in max_dep_by_col.items():
                    c_int = int(c)
                    max_dep = int(m)
                    if max_dep <= 0:
                        continue
                    if max_dep % 2 == 1:
                        center = (max_dep + 1) / 2.0
                    else:
                        center = (max_dep / 2.0 + (max_dep / 2.0 + 1.0)) / 2.0
                    depths_by_col_calc[c_int] = (max_dep, float(center))
    except Exception:
        depths_by_col_calc = None

    # --- qty_cases（ベース: float）と、移動用の整数（ceil）
    if "cases" in inv.columns:
        inv["qty_cases_float"] = pd.to_numeric(inv["cases"], errors="coerce").fillna(0.0)
    elif "ケース" in inv.columns:
        inv["qty_cases_float"] = pd.to_numeric(inv["ケース"], errors="coerce").fillna(0.0)
    else:
        items = pd.to_numeric(inv.get("在庫数(引当数を含む)", 0), errors="coerce").fillna(0.0)
        if pack_map is not None:
            key_series = inv["商品ID"].astype(str)
            packs = key_series.map(pack_map).fillna(1.0)
            packs = packs.replace({0.0: 1.0})
            inv["qty_cases_float"] = items / packs
        else:
            inv["qty_cases_float"] = items

    # ベクトル化: np.ceilで高速化
    inv["qty_cases_move"] = np.ceil(inv["qty_cases_float"].astype(float)).astype(int)
    if not inv.empty:
        try:
            qmin = int(inv["qty_cases_move"].min())
            qmax = int(inv["qty_cases_move"].max())
            print(f"[optimizer] qty_cases_move min={qmin} max={qmax}")
        except Exception:
            pass

    # --- バラ数の計算と移動対象フラグ ---
    # 「バラのみ」または「10ケース以下+バラ混在」は移動対象から除外
    # qty: 元の総数量、pack_est: 入数
    if "qty" in inv.columns:
        inv["_orig_qty"] = pd.to_numeric(inv["qty"], errors="coerce").fillna(0).astype(int)
    elif "在庫数(引当数を含む)" in inv.columns:
        inv["_orig_qty"] = pd.to_numeric(inv["在庫数(引当数を含む)"], errors="coerce").fillna(0).astype(int)
    else:
        inv["_orig_qty"] = 0
    
    # 入数を取得（pack_mapがあれば使用、なければinvのpack_qty）
    if pack_map is not None:
        inv["_pack_qty"] = inv["商品ID"].astype(str).map(pack_map).fillna(1.0)
    elif "pack_qty" in inv.columns:
        inv["_pack_qty"] = pd.to_numeric(inv["pack_qty"], errors="coerce").fillna(1.0)
    else:
        inv["_pack_qty"] = 1.0
    inv["_pack_qty"] = inv["_pack_qty"].replace({0.0: 1.0})  # 0除算防止
    
    # 実際のケース数とバラ数を計算
    inv["_actual_cases"] = (inv["_orig_qty"] // inv["_pack_qty"]).astype(int)
    inv["_actual_bara"] = (inv["_orig_qty"] % inv["_pack_qty"]).astype(int)
    
    # 移動対象フラグ: バラのみ or 10ケース以下+バラ混在 は除外
    # is_movable = True: 移動対象（ケースのみ、または11ケース以上+バラ）
    # is_movable = False: 移動対象外（バラのみ、または1-10ケース+バラ）
    bara_only = (inv["_actual_cases"] == 0) & (inv["_actual_bara"] > 0)
    low_case_with_bara = (inv["_actual_cases"] > 0) & (inv["_actual_cases"] <= 10) & (inv["_actual_bara"] > 0)
    inv["is_movable"] = ~(bara_only | low_case_with_bara)
    
    # 除外件数をログ出力
    bara_only_count = bara_only.sum()
    low_case_bara_count = low_case_with_bara.sum()
    movable_count = inv["is_movable"].sum()
    print(f"[optimizer] バラ除外: バラのみ={bara_only_count}件, 10ケース以下+バラ={low_case_bara_count}件, 移動対象={movable_count}件")
    try:
        _publish_progress(get_current_trace_id(), {
            "type": "info", "phase": "filter",
            "message": f"バラ除外: バラのみ={bara_only_count}件, 10ケース以下+バラ={low_case_bara_count}件 → 移動対象={movable_count}件"
        })
    except Exception:
        pass

    # --- ケース容積を付与（安全側：移動用整数で容量評価）
    key_series = inv["商品ID"].astype(str)
    inv["volume_each_case"] = key_series.map(sku_vol_map).fillna(0.0)
    inv["volume_total"] = inv["qty_cases_move"] * inv["volume_each_case"]

    # オーバーサイズSKUの早期除外
    if getattr(cfg, "exclude_oversize", False):
        cap_threshold = (max(cap_by_loc.values()) if cap_by_loc else cap_limit)
        inv = inv[inv["volume_each_case"] <= cap_threshold].copy()

    # --- 推定入数（pack_est）: SKUマスターに入数があればそれを使う
    if pack_map is not None:
        inv["pack_est"] = inv["商品ID"].astype(str).map(pack_map)
    else:
        inv["pack_est"] = pd.NA

    # 列⇔SKU の存在マップ（同一SKUを同じ列にまとめるための判断に使用）
    # ※この時点の在庫状態に基づく（逐次更新は後続で最小限のみ反映）
    try:
        # ベクトル化: applyを使わずagg + 辞書内包表記で高速化
        col_to_skus = {col: set(grp["商品ID"].astype(str).unique()) for col, grp in inv.groupby("col")}
    except Exception:
        col_to_skus = {}
    try:
        sku_to_cols = {sku: set(grp["col"].astype(int).unique()) for sku, grp in inv.groupby("商品ID")}
    except Exception:
        sku_to_cols = {}

    # --- 列ごとの“支配的な入数”を計算し、元のマッピングをほどほどに維持
    dominant_pack_by_col: dict[int, float] = {}
    try:
        tmp = inv.dropna(subset=["pack_est"]).copy()
        if not tmp.empty:
            tmp["weight"] = pd.to_numeric(tmp["qty_cases_move"], errors="coerce").fillna(0).astype(float)
            by_col_pack = tmp.groupby(["col", "pack_est"], dropna=False)["weight"].sum().reset_index()
            total_by_col = by_col_pack.groupby("col")["weight"].sum()
            top_by_col = by_col_pack.sort_values(["col", "weight"], ascending=[True, False]).groupby("col").head(1)
            thr = getattr(cfg, "preserve_pack_mapping_threshold", 0.7)
            for _, r in top_by_col.iterrows():
                c = int(r["col"]) ; top_w = float(r["weight"]) ; tot = float(total_by_col.get(c, 0.0) or 0.0)
                share = top_w / tot if tot > 0 else 0.0
                if share >= thr:
                    dominant_pack_by_col[c] = float(r["pack_est"])
    except Exception:
        dominant_pack_by_col = {}
    print(f"[optimizer] dominant_pack_by_col keys={len(dominant_pack_by_col)} thr={getattr(cfg,'preserve_pack_mapping_threshold',0.7)}")

    # 列ごとの代表入数: 支配的入数があればそれを優先、無ければメディアン
    rep_pack_by_col: dict[int, float] = {}
    try:
        tmp2 = inv.dropna(subset=["pack_est"]).copy()
        if not tmp2.empty:
            tmp2["pack_est"] = pd.to_numeric(tmp2["pack_est"], errors="coerce")
            med_by_col = tmp2.groupby("col", dropna=True)["pack_est"].median()
            for c, med in med_by_col.items():
                if pd.notna(med):
                    rep_pack_by_col[int(c)] = float(med)
    except Exception:
        rep_pack_by_col = {}

    # 支配的入数がある列はそれを代表値として上書き
    for c, dom in dominant_pack_by_col.items():
        rep_pack_by_col[c] = float(dom)

    # 列ごとの“ミックス枠”初期化（±帯から外れても許容できる枠）
    try:
        unique_cols = sorted(int(x) for x in inv["col"].dropna().unique())
    except Exception:
        unique_cols = []
    mix_slots_left: dict[int, int] = {c: int(getattr(cfg, "mix_slots_per_col", 1)) for c in unique_cols}

    # 容積情報が無い行はスキップ
    inv = inv[inv["volume_each_case"] > 0].copy()

    # --- 棚毎の使用量 (m³)
    shelf_usage = inv.groupby("ロケーション")["volume_total"].sum().to_dict()
    # Ensure empty candidate locations exist in shelf_usage when loc_master provided
    if cap_by_loc:
        for loc in cap_by_loc.keys():
            shelf_usage.setdefault(loc, 0.0)
    print(f"[optimizer] shelf_usage locations={len(shelf_usage)} (cap={cap_limit})")
    moves: List[Move] = []

    # --- 実行順序: Pass-FIFO → Pass-1 → Pass-0 (列跨ぎFIFOを最優先、同列内整列を次に) ---
    import time
    def _log(msg):  # local log helper
        try:
            print(f"[optimizer] {msg}")
        except Exception:
            pass
    # Track Pass-wise statistics for summary report
    pass_stats = {"pass0": 0, "pass1": 0, "pass2": 0, "pass3": 0}
    
    # --- 200件制限の中で最適な組み合わせを実現 ---
    # 全Pass候補を収集してから、優先度でソートして上位200件を選択
    
    all_candidate_moves: List[Move] = []
    
    # --- Pass-FIFO: 列を跨いだFIFO最適化（最古ロットを最優先引き当て位置へ）【最優先】
    # Pass-1より先に実行することで、全ロットを考慮した列跨ぎFIFO是正を行う
    pf_t0 = time.perf_counter()
    pf_moves = _pass_fifo_cross_column(
        inv, shelf_usage, cap_limit, cfg,
        cap_by_loc=cap_by_loc or None,
        can_receive=can_receive_set or None,
        budget_left=None,  # 制限なしで候補収集
    )
    all_candidate_moves.extend(pf_moves)
    _log(f"Pass-FIFO cross-column: candidates={len(pf_moves)} time={time.perf_counter()-pf_t0:.2f}s")
    
    # --- Pass-1: 取口/保管バランス整列（古い→1-2段, 新しい→3-4段）【第2優先】
    # Pass-FIFOで移動しなかったロットに対して、同一列内での段整列を行う
    p1_t0 = time.perf_counter()
    p1_moves = _pass1_pick_storage_balance(
        inv, shelf_usage, cap_limit, cfg,
        cap_by_loc=cap_by_loc or None,
        can_receive=can_receive_set or None,
        budget_left=None,  # 制限なしで候補収集
    )
    all_candidate_moves.extend(p1_moves)
    _log(f"Pass-1 pick/storage: candidates={len(p1_moves)} time={time.perf_counter()-p1_t0:.2f}s")
    
    # --- Pass-0: エリア（列ゾーニング）の是正（列またぎ再配置）【第2優先】
    if getattr(cfg, "enable_pass0_area_rebalance", True) and getattr(cfg, "strict_pack_area", True):
        p0_t0 = time.perf_counter()
        p0_moves = _pass0_area_rebalance(
            inv=inv,
            shelf_usage=shelf_usage,
            cap_limit=cap_limit,
            cfg=cfg,
            rep_pack_by_col=rep_pack_by_col,
            pack_tolerance_ratio=getattr(cfg, "pack_tolerance_ratio", 0.10),
            sku_vol_map=sku_vol_map,
            ai_col_hints=ai_col_hints,
            cap_by_loc=cap_by_loc or None,
            can_receive=can_receive_set or None,
            budget_left=None,  # 制限なしで候補収集
        )
        all_candidate_moves.extend(p0_moves)
        _log(f"Pass-0 area_rebalance: candidates={len(p0_moves)} time={time.perf_counter()-p0_t0:.2f}s")
    
    print(f"[optimizer] Total candidates collected: {len(all_candidate_moves)}")
    
    # --- 優先度ソート: Pass-1 > Pass-0 (reasonキーワードで判定) ---
    def _get_pass_priority(m: Move) -> int:
        reason = str(m.reason or "")
        if "古ロット" in reason or "先入先出" in reason or "取口配置" in reason:
            return 1  # Pass-1: FIFO最優先
        elif "FIFO是正" in reason or "最古ロット優先" in reason:
            return 1  # Pass-FIFO: 列跨ぎFIFOも最優先
        elif "入数帯是正" in reason or "エリア" in reason or "列移動" in reason or "入口に近づく" in reason:
            return 2  # Pass-0: エリア
        elif "圧縮" in reason or "集約" in reason:
            return 3  # Pass-2
        elif "AI" in reason:
            return 4  # Pass-3
        return 5
    
    def _calc_ease_gain(m: Move) -> float:
        # 移動効果: 距離改善 + レベル改善
        # location座標から距離を計算
        dist_gain = 0.0
        level_gain = 0.0
        try:
            from_lv, from_col, from_dep = _parse_loc8(str(m.from_loc))
            to_lv, to_col, to_dep = _parse_loc8(str(m.to_loc))
            
            # 距離改善: 列と奥行の差分
            col_dist = abs(to_col - from_col)
            dep_dist = abs(to_dep - from_dep)
            dist_gain = col_dist + dep_dist
            
            # レベル改善: 下段化はプラス
            if to_lv < from_lv:
                level_gain = (from_lv - to_lv) * 2.0
        except Exception:
            pass
        return dist_gain + level_gain
    
    print(f"[optimizer] About to sort {len(all_candidate_moves)} candidates...")
    # 優先度順にソート (Pass優先 → 効果大きい順)
    # ただし chain_group_id がある移動はグループ単位でまとめて並べる
    try:
        moves_with_meta = []
        for i, m in enumerate(all_candidate_moves):
            if i % 100 == 0:
                print(f"[optimizer] Processing candidate {i}/{len(all_candidate_moves)}...")
            try:
                priority = _get_pass_priority(m)
                gain = _calc_ease_gain(m)
                moves_with_meta.append((m, priority, gain))
            except Exception as e:
                print(f"[optimizer] ERROR processing candidate {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
        print(f"[optimizer] Created moves_with_meta with {len(moves_with_meta)} entries, now sorting...")
        
        # === chain_group_id 対応ソート ===
        # 1. chain_group_id でグループ化
        from collections import defaultdict
        chain_groups: Dict[str, List[Tuple[Move, int, float]]] = defaultdict(list)
        standalone_moves: List[Tuple[Move, int, float]] = []
        
        for m, priority, gain in moves_with_meta:
            chain_id = getattr(m, 'chain_group_id', None)
            if chain_id:
                chain_groups[chain_id].append((m, priority, gain))
            else:
                standalone_moves.append((m, priority, gain))
        
        # 2. 各グループ内を execution_order でソート
        for chain_id in chain_groups:
            chain_groups[chain_id].sort(
                key=lambda x: (getattr(x[0], 'execution_order', 999) or 999)
            )
        
        # 3. グループの代表優先度・効果を計算（グループ内の最高優先度 & 合計効果）
        group_meta: List[Tuple[str, int, float, List[Tuple[Move, int, float]]]] = []
        for chain_id, group_moves in chain_groups.items():
            group_priority = min(m[1] for m in group_moves)  # 最高優先度（最小値）
            group_gain = sum(m[2] for m in group_moves)  # 合計効果
            group_meta.append((chain_id, group_priority, group_gain, group_moves))
        
        # 4. グループを優先度→効果順でソート
        group_meta.sort(key=lambda x: (x[1], -x[2]))  # priority ASC, gain DESC
        
        # 5. スタンドアロン移動もソート
        standalone_moves.sort(key=lambda x: (x[1], -x[2]))  # priority ASC, gain DESC
        
        # 6. グループとスタンドアロンを統合（優先度→効果順でマージ）
        sorted_moves_with_meta: List[Tuple[Move, int, float]] = []
        group_idx = 0
        standalone_idx = 0
        
        while group_idx < len(group_meta) or standalone_idx < len(standalone_moves):
            # 次のグループと次のスタンドアロンを比較
            if group_idx >= len(group_meta):
                # グループが尽きた -> スタンドアロンを追加
                sorted_moves_with_meta.append(standalone_moves[standalone_idx])
                standalone_idx += 1
            elif standalone_idx >= len(standalone_moves):
                # スタンドアロンが尽きた -> グループを追加
                for m_tuple in group_meta[group_idx][3]:
                    sorted_moves_with_meta.append(m_tuple)
                group_idx += 1
            else:
                # 両方ある -> 優先度と効果で比較
                g_priority, g_gain = group_meta[group_idx][1], group_meta[group_idx][2]
                s_priority, s_gain = standalone_moves[standalone_idx][1], standalone_moves[standalone_idx][2]
                
                if (g_priority, -g_gain) <= (s_priority, -s_gain):
                    # グループの方が優先
                    for m_tuple in group_meta[group_idx][3]:
                        sorted_moves_with_meta.append(m_tuple)
                    group_idx += 1
                else:
                    # スタンドアロンの方が優先
                    sorted_moves_with_meta.append(standalone_moves[standalone_idx])
                    standalone_idx += 1
        
        moves_with_meta = sorted_moves_with_meta
        print(f"[optimizer] Sorting complete (chain_groups={len(chain_groups)}, standalone={len(standalone_moves)})")
    except Exception as e:
        print(f"[optimizer] FATAL ERROR in sorting: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # max_moves制限とSKU移動元ロケーション数制限を適用
    max_moves = getattr(cfg, "max_moves", None)
    max_source_locs_per_sku = getattr(cfg, "max_source_locs_per_sku", None)
    print(f"[optimizer] max_moves={max_moves}, max_source_locs_per_sku={max_source_locs_per_sku}, total_candidates={len(moves_with_meta)}")
    
    # SKUごとの移動元ロケーション数を制限しながら選択
    moves = []
    sku_source_locs: Dict[str, Set[str]] = {}  # SKU -> 移動元ロケーションのセット
    skipped_by_limit = 0
    
    for m, priority, gain in moves_with_meta:
        # max_moves制限チェック
        if max_moves is not None and len(moves) >= max_moves:
            break
        
        # SKU移動元ロケーション数制限チェック
        if max_source_locs_per_sku is not None:
            sku_id = str(m.sku_id)
            from_loc = str(m.from_loc)
            
            if sku_id not in sku_source_locs:
                sku_source_locs[sku_id] = set()
            
            # このSKUの移動元ロケーション数が上限に達している場合
            if from_loc not in sku_source_locs[sku_id]:
                if len(sku_source_locs[sku_id]) >= max_source_locs_per_sku:
                    # スキップ（別のロケーションからの移動は許可しない）
                    skipped_by_limit += 1
                    continue
                # 新しい移動元ロケーションを追加
                sku_source_locs[sku_id].add(from_loc)
        
        moves.append(m)
    
    if skipped_by_limit > 0:
        _log(f"SKU移動元ロケーション数制限により {skipped_by_limit} 件をスキップ")
        print(f"[optimizer] Skipped {skipped_by_limit} moves due to max_source_locs_per_sku={max_source_locs_per_sku}")
    
    if max_moves is not None and len(moves_with_meta) > len(moves):
        _log(f"Trimmed to {len(moves)} moves (from {len(moves_with_meta)} candidates)")
    
    print(f"[optimizer] After selection: len(moves)={len(moves)}")
    
    # Pass統計を更新
    pass_stats["pass1"] = sum(1 for m in moves if _get_pass_priority(m) == 1)
    pass_stats["pass0"] = sum(1 for m in moves if _get_pass_priority(m) == 2)
    
    _log(f"Final move selection: Pass-1={pass_stats['pass1']}, Pass-0={pass_stats['pass0']}, total={len(moves)}")
    
    # Debug: Log first few moves
    if moves:
        print(f"[optimizer] First 5 selected moves:")
        for i, m in enumerate(moves[:5]):
            priority = _get_pass_priority(m)
            print(f"  [{i}] priority={priority}, sku={m.sku_id}, from={m.from_loc}, to={m.to_loc}, reason={m.reason[:80] if m.reason else 'None'}")
    else:
        print(f"[optimizer] WARNING: No moves selected after sorting/filtering!")
    
    try:
        _publish_progress(get_current_trace_id(), {
            "type": "phase", "name": "pass_selection", 
            "moves": len(moves), "total_moves": len(moves),
            "pass1": pass_stats["pass1"], "pass0": pass_stats["pass0"],
            "message": f"最適な{len(moves)}件を選択 (Pass-1: {pass_stats['pass1']}件, Pass-0: {pass_stats['pass0']}件)"
        })
    except Exception:
        pass
    
    moved = len(moves)
    # Pass-0/Pass-1で全体が更新されたので並べ替えし直し
    order_idx = _sort_index_for_pick(inv)

    # --- strict pack-A gating flag ---
    hard_pack_A = bool(getattr(cfg, "strict_pack", None) and ("A" in str(getattr(cfg, "strict_pack"))))

    # Track planned moves to same location for lot-mixing checks
    # Initialize with Pass-1 and Pass-0 moves
    main_loop_t0 = time.perf_counter()
    planned_lots_by_loc_sku: Dict[Tuple[str, str], Set[int]] = {}
    for move in moves:
        # Extract lot_key from move's lot_date (YYYYMMDD format)
        try:
            lot_key = _datestr8_to_lot_key(move.lot_date) if move.lot_date else UNKNOWN_LOT_KEY
        except Exception:
            lot_key = UNKNOWN_LOT_KEY
        lookup_key = (str(move.to_loc), str(move.sku_id))
        if lookup_key not in planned_lots_by_loc_sku:
            planned_lots_by_loc_sku[lookup_key] = set()
        planned_lots_by_loc_sku[lookup_key].add(int(lot_key))

    # Performance optimization: pre-group locations by level
    locs_by_level: Dict[int, List[str]] = {}
    for loc in shelf_usage.keys():
        if loc not in PLACEHOLDER_LOCS:
            lv = _parse_loc8(loc)[0]
            if lv not in locs_by_level:
                locs_by_level[lv] = []
            locs_by_level[lv].append(loc)
    # Pre-sort each level's locations
    for lv in locs_by_level:
        locs_by_level[lv].sort(key=_location_key)
    _log(f"Pre-cached {len(locs_by_level)} levels with locations")

    # ====== PERFORMANCE OPTIMIZATION: Pre-build inventory indexes ======
    # Build global indexes first (for eviction chain functions)
    _build_global_inv_indexes(inv)
    
    # This avoids repeated DataFrame filtering in the main loop (major bottleneck)
    _log("Building inventory indexes for fast lookup...")
    _idx_build_start = time.perf_counter()
    
    # Index 1: lots by (location, sku) -> set of lot_keys
    inv_lots_by_loc_sku: Dict[Tuple[str, str], Set[int]] = {}
    # Index 2: columns where each (sku, lot_key) exists
    inv_cols_by_sku_lot: Dict[Tuple[str, int], Set[int]] = {}
    # Index 3: inventory rows at each location for a SKU (for lot-level rule check)
    inv_rows_by_loc_sku: Dict[Tuple[str, str], List[Tuple[int, int, int]]] = {}  # list of (lot_key, level, idx)
    # Index 4: lot_key -> level mapping by (sku, column) for FIFO rule check
    inv_lot_levels_by_sku_col: Dict[Tuple[str, int], List[Tuple[int, int]]] = {}  # list of (lot_key, level)
    
    if "商品ID" in inv.columns and "lot_key" in inv.columns and "ロケーション" in inv.columns:
        for idx_row in inv.index:
            try:
                sku_v = str(inv.at[idx_row, "商品ID"])
                loc_v = str(inv.at[idx_row, "ロケーション"])
                lot_k = int(pd.to_numeric(inv.at[idx_row, "lot_key"], errors="coerce") or UNKNOWN_LOT_KEY)
                col_v = int(inv.at[idx_row, "col"]) if pd.notna(inv.at[idx_row, "col"]) else 0
                lv_v = int(inv.at[idx_row, "lv"]) if pd.notna(inv.at[idx_row, "lv"]) else 0
                
                # Index 1
                key1 = (loc_v, sku_v)
                if key1 not in inv_lots_by_loc_sku:
                    inv_lots_by_loc_sku[key1] = set()
                inv_lots_by_loc_sku[key1].add(lot_k)
                
                # Index 2
                key2 = (sku_v, lot_k)
                if key2 not in inv_cols_by_sku_lot:
                    inv_cols_by_sku_lot[key2] = set()
                inv_cols_by_sku_lot[key2].add(col_v)
                
                # Index 3
                if key1 not in inv_rows_by_loc_sku:
                    inv_rows_by_loc_sku[key1] = []
                inv_rows_by_loc_sku[key1].append((lot_k, lv_v, idx_row))
                
                # Index 4: For FIFO rule - lot_key and level by (sku, column)
                key4 = (sku_v, col_v)
                if key4 not in inv_lot_levels_by_sku_col:
                    inv_lot_levels_by_sku_col[key4] = []
                inv_lot_levels_by_sku_col[key4].append((lot_k, lv_v))
            except Exception:
                pass
    
    _log(f"Inventory indexes built in {time.perf_counter() - _idx_build_start:.2f}s: {len(inv_lots_by_loc_sku)} loc-sku pairs, {len(inv_cols_by_sku_lot)} sku-lot pairs, {len(inv_lot_levels_by_sku_col)} sku-col pairs")
    # ====== END PERFORMANCE OPTIMIZATION ======

    processed_rows = 0
    total_rows = len(order_idx)
    progress_mod = max(1, total_rows // 20)  # Show progress every 5% of total, at least every row
    cancellation_check_mod = max(1, total_rows // 10)  # Check cancellation every 10%
    
    # Time limit for main loop processing (default 5 minutes = 300 seconds)
    import time as _time_module
    _loop_start_time = _time_module.time()
    _loop_time_limit = float(getattr(cfg, "loop_time_limit", 300))  # seconds
    _time_check_mod = max(1, total_rows // 50)  # Check time every 2%
    
    def _log(msg):
        try:
            print(f"[optimizer] {msg}")
        except Exception:
            pass
    
    # Import cancellation check function
    try:
        from app.services.relocation_tasks import is_task_cancelled
    except ImportError:
        is_task_cancelled = lambda x: False  # Fallback if not available
    
    for idx in order_idx:
        # move cap（UIの最大など）を厳守
        if getattr(cfg, "max_moves", None) is not None and len(moves) >= int(getattr(cfg, "max_moves")):
            break
        
        # Check if task was cancelled (superseded by newer request)
        if processed_rows % cancellation_check_mod == 0 and processed_rows > 0:
            current_trace = get_current_trace_id()
            if current_trace and is_task_cancelled(current_trace):
                _log(f"Task {current_trace} cancelled, stopping optimization early")
                break
        
        # Time limit check - stop if we've exceeded the time budget
        if processed_rows % _time_check_mod == 0 and processed_rows > 0:
            elapsed = _time_module.time() - _loop_start_time
            if elapsed > _loop_time_limit:
                _log(f"Time limit reached ({elapsed:.1f}s > {_loop_time_limit}s), stopping with {len(moves)} moves after {processed_rows}/{total_rows} rows")
                _publish_progress(get_current_trace_id(), {
                    "type": "info", "phase": "plan",
                    "message": f"処理時間制限に達しました（{int(elapsed)}秒）。{len(moves)}件の移動案で終了します。"
                })
                break
        
        row = inv.loc[idx]
        processed_rows += 1
        if processed_rows % progress_mod == 0:
            try:
                rem = (int(getattr(cfg, "max_moves")) - len(moves)) if getattr(cfg, "max_moves", None) is not None else None
            except Exception:
                rem = None
            elapsed = _time_module.time() - _loop_start_time
            _log(f"main: processed {processed_rows}/{total_rows}; moves={len(moves)} remaining={rem} elapsed={elapsed:.1f}s")
            try:
                progress_pct = int(100 * processed_rows / total_rows) if total_rows > 0 else 0
                msg = f"在庫処理中: {processed_rows}/{total_rows}行 ({progress_pct}%) - 移動案{len(moves)}件"
                if rem is not None:
                    msg += f" (残り{rem}件まで)"
                _publish_progress(get_current_trace_id(), {
                    "type": "progress", "phase": "plan", 
                    "processed": processed_rows, "total": total_rows, "moves": len(moves),
                    "message": msg
                })
            except Exception:
                pass
        from_loc = str(row["ロケーション"])  # 8桁
        lv, col, dep = row.get("lv"), row.get("col"), row.get("dep")
        if pd.isna(lv) or pd.isna(col) or pd.isna(dep):
            continue
        lv, col, dep = int(lv), int(col), int(dep)
        cur_key = _ease_key(lv, col, dep)
        if lv == 1:
            _record_drop("already_optimal", {
                "sku_id": str(row["商品ID"]),
                "lot": str(row.get("ロット") or ""),
                "from_loc": from_loc,
                "to_loc": "",
                "qty_cases": int(row.get("qty_cases_move") or 0),
            }, note="already at lowest level")
            continue  # 既に最下段

        qty_cases = int(row["qty_cases_move"]) or 0
        if qty_cases <= 0:
            continue
        need_vol = float(row["volume_each_case"]) * qty_cases
        sku_val = str(row["商品ID"])
        lot_key = int(row.get("lot_key") or UNKNOWN_LOT_KEY)
        if int(lot_key) == UNKNOWN_LOT_KEY:
            _record_drop("unknown_lot", {
                "sku_id": sku_val,
                "lot": str(row.get("ロット") or ""),
                "from_loc": from_loc,
                "to_loc": "",
                "qty_cases": qty_cases,
            }, note="cannot parse lot date")
            continue

        # Initialize failure counters for this row
        row_fail = {"capacity": 0, "fifo": 0, "pack_band": 0, "forbidden": 0, "other": 0}

        # Allow same-level improvements
        levels = list(range(1, lv))
        if getattr(cfg, "allow_same_level_ease_improve", True):
            levels.append(lv)
        best_choice: Optional[Tuple[str, int, int, int, bool]] = None
        best_score: float = math.inf
        best_ev_chain: List[Move] = []
        best_ev_chain_group_id: Optional[str] = None
        cur_col = int(col)
        pack_val = row.get("pack_est")
        
        # --- Parallel or serial candidate evaluation ---
        use_parallel = getattr(cfg, "enable_parallel", False) and not getattr(cfg, "chain_depth", 0)
        
        if use_parallel:
            # Parallel evaluation: collect all candidates first, then evaluate in parallel
            all_candidates = []
            for target_level in levels:
                cand_locs = locs_by_level.get(target_level, [])
                if can_receive_set:
                    cand_locs = [loc for loc in cand_locs if loc != from_loc and loc in can_receive_set]
                else:
                    cand_locs = [loc for loc in cand_locs if loc != from_loc]
                for to_loc in cand_locs:
                    all_candidates.append((to_loc, target_level))
            
            # Evaluate candidates in parallel
            if all_candidates:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                max_workers = min(getattr(cfg, "parallel_workers", 4), len(all_candidates))
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {}
                    for to_loc, target_level in all_candidates:
                        future = executor.submit(
                            _evaluate_candidate_location,
                            to_loc, target_level, row, from_loc, lv, cur_key, lot_key,
                            need_vol, sku_val, qty_cases, idx, inv, shelf_usage, cap_limit,
                            rep_pack_by_col, mix_slots_left, planned_lots_by_loc_sku,
                            sku_vol_map, ai_col_hints, cap_by_loc, can_receive_set,
                            hard_pack_A, depths_by_col_calc, cfg
                        )
                        futures[future] = (to_loc, target_level)
                    
                    # Collect results
                    for future in as_completed(futures):
                        result = future.result()
                        if result is None:
                            continue
                        if result.failure_reason:
                            row_fail[result.failure_reason] = row_fail.get(result.failure_reason, 0) + 1
                            continue
                        if result.score < best_score:
                            best_score = result.score
                            best_choice = (result.to_loc, result.target_level, result.tcol, result.tdep, result.area_needs_mix)
                            best_ev_chain = result.candidate_ev_chain
        else:
            # Serial evaluation (original code path)
            for target_level in levels:
                # Use pre-cached locations by level (performance optimization)
                cand_locs = locs_by_level.get(target_level, [])
                # Filter by can_receive_set and exclude from_loc
                if can_receive_set:
                    cand_locs = [loc for loc in cand_locs if loc != from_loc and loc in can_receive_set]
                else:
                    cand_locs = [loc for loc in cand_locs if loc != from_loc]
                # Already sorted by _location_key during pre-caching
                for to_loc in cand_locs:
                    tlv, tcol, tdep = _parse_loc8(to_loc)
                    # 同段移動は (col,dep) が厳密に良くなる場合のみ
                    if target_level == lv:
                        new_key = _ease_key(tlv, tcol, tdep)
                        if new_key >= cur_key:
                            continue
                    # --- Area gating by pack/promo (strict_pack_area + mix枠/小入数1-2緩和)
                    area_needs_mix = False
                    if getattr(cfg, "strict_pack_area", True):
                        allowed_cols = _allowed_cols_for_row(row, cfg)
                        # 入数1-2は緩和（別途まとめ運用のため）
                        relax_smallpack = False
                        try:
                            relax_smallpack = float(row.get("pack_est") or 0) <= 2.0
                        except Exception:
                            relax_smallpack = False
                        if (tcol not in allowed_cols) and not relax_smallpack:
                            if mix_slots_left.get(tcol, 0) <= 0:
                                row_fail["pack_band"] += 1
                                continue
                            area_needs_mix = True
                    # Strict pack-A: skip out-of-band destinations entirely
                    if hard_pack_A and (row.get("pack_est") is not None) and not pd.isna(row.get("pack_est")):
                        rep = rep_pack_by_col.get(tcol)
                        try:
                            if rep and float(rep) > 0:
                                diff_ratio = abs(float(row.get("pack_est")) - float(rep)) / float(rep)
                                if diff_ratio > cfg.pack_tolerance_ratio:
                                    row_fail["pack_band"] += 1
                                    continue
                        except Exception:
                            pass
                    # OPTIMIZED: Use fast dict-based FIFO check instead of DataFrame filtering
                    if _violates_lot_level_rule_fast(sku_val, lot_key, target_level, tcol, inv_lot_levels_by_sku_col):
                        row_fail["fifo"] += 1
                        continue

                    # Hard rule: 同一SKUで異なるロットは同一ロケーションに置かない
                    # Check both existing inventory AND already planned moves to this location
                    # OPTIMIZED: Use pre-built index instead of DataFrame filtering
                    try:
                        exists_mixed = False
                        conflicting_lots: Set[int] = set()
                        
                        # 1) Check existing inventory in target location (FAST: dict lookup)
                        loc_sku_key = (str(to_loc), str(sku_val))
                        existing_lots = inv_lots_by_loc_sku.get(loc_sku_key, set())
                        if existing_lots:
                            if int(lot_key) == UNKNOWN_LOT_KEY:
                                exists_mixed = True
                            elif int(lot_key) not in existing_lots:
                                exists_mixed = True
                                conflicting_lots.update(existing_lots)
                        
                        # 2) Check already planned moves to this location with same SKU
                        if loc_sku_key in planned_lots_by_loc_sku:
                            planned_lots = planned_lots_by_loc_sku[loc_sku_key]
                            if int(lot_key) == UNKNOWN_LOT_KEY:
                                if planned_lots:
                                    exists_mixed = True
                            elif planned_lots and int(lot_key) not in planned_lots:
                                exists_mixed = True
                                conflicting_lots.update(planned_lots)
                        
                        if exists_mixed:
                            row_fail["forbidden"] += 1
                            continue
                    except Exception as e:
                        # 診断は最適化を壊さない
                        _log(f"Warning: lot-mixing check failed: {e}")
                        pass
                    candidate_ev_chain: List[Move] = []
                    candidate_ev_chain_group_id: Optional[str] = None
                    used = float(shelf_usage.get(to_loc, 0.0))
                    limit = cap_by_loc.get(to_loc, cap_limit) if cap_by_loc else cap_limit
                    if used + need_vol > limit:
                        # Try bounded eviction chain if enabled
                        if getattr(cfg, "chain_depth", 0) and getattr(cfg, "eviction_budget", 0) and getattr(cfg, "touch_budget", 0):
                            budget = _ChainBudget(
                                depth_left=int(getattr(cfg, "chain_depth", 0)),
                                evictions_left=int(getattr(cfg, "eviction_budget", 0)),
                                touch_left=int(getattr(cfg, "touch_budget", 0)),
                                touched=set([from_loc, to_loc]),
                            )
                            # Generate chain_group_id for eviction chain
                            candidate_ev_chain_group_id = f"evict_{secrets.token_hex(6)}"
                            
                            candidate_ev_chain = _plan_eviction_chain(
                                need_vol=need_vol,
                                target_loc=str(to_loc),
                                inv=inv,
                                shelf_usage=shelf_usage,
                                cap_limit=cap_limit,
                                sku_vol_map=sku_vol_map,
                                rep_pack_by_col=rep_pack_by_col,
                                pack_tolerance_ratio=getattr(cfg, "pack_tolerance_ratio", 0.10),
                                budget=budget,
                                ai_col_hints=ai_col_hints,
                                cap_by_loc=cap_by_loc or None,
                                can_receive=can_receive_set or None,
                                hard_pack_A=hard_pack_A,
                                ease_weight=getattr(cfg, "ease_weight", 0.0001),
                                cfg=cfg,
                                chain_group_id=candidate_ev_chain_group_id,
                                execution_order_start=1,
                            )
                            if candidate_ev_chain is None:
                                row_fail["capacity"] += 1
                                continue
                            # Tentatively accept this candidate with the eviction chain
                            # We will append `candidate_ev_chain` moves just before the main move (after scoring passes)
                        else:
                            row_fail["capacity"] += 1
                            continue

                    # Score this destination choice (unified scoring API)
                    score = _score_destination_for_row(
                        row,
                        str(to_loc),
                        rep_pack_by_col,
                        getattr(cfg, "pack_tolerance_ratio", 0.10),
                        getattr(cfg, "same_sku_same_column_bonus", 20.0),
                        getattr(cfg, "prefer_same_column_bonus", 5.0),
                        getattr(cfg, "split_sku_new_column_penalty", 5.0),
                        ai_col_hints,
                        ease_weight=getattr(cfg, "ease_weight", 0.0001),
                        depths_by_col=depths_by_col_calc,
                        cfg=cfg,
                    )
                    # Soft preference: 同一SKU・同一ロットが既に存在する列ならボーナス
                    # OPTIMIZED: Use pre-built index instead of DataFrame filtering
                    try:
                        same_lot_cols = inv_cols_by_sku_lot.get((str(sku_val), int(lot_key)), set())
                        if int(tcol) in same_lot_cols:
                            score -= float(getattr(cfg, "same_lot_same_column_bonus", 10.0))
                    except Exception:
                        pass
                    
                    if score < best_score:
                        best_score = score
                        best_choice = (to_loc, target_level, tcol, tdep, area_needs_mix)
                        best_ev_chain = candidate_ev_chain
                        best_ev_chain_group_id = candidate_ev_chain_group_id
        
        # If we found a best choice, create the move
        if best_choice is not None:
            to_loc, target_level, tcol, tdep, needs_mix = best_choice
            
            # If this move needs a mix slot, consume it
            if needs_mix and mix_slots_left.get(tcol, 0) > 0:
                mix_slots_left[tcol] -= 1
            
            # Determine reason for Pass-2 moves (compression & consolidation)
            from_lv, from_col, from_dep = _parse_loc8(from_loc)
            
            actions = []
            improvements = []
            
            # Level movement (prioritize lower levels)
            if target_level < from_lv:
                level_diff = from_lv - target_level
                if level_diff >= 2:
                    actions.append(f"大幅段下げ(Lv{from_lv}→{target_level})")
                    improvements.append("下段活用・圧縮")
                else:
                    actions.append(f"段下げ(Lv{from_lv}→{target_level})")
                    improvements.append("下段活用")
            elif target_level == from_lv:
                actions.append(f"同段内移動(Lv{target_level})")
                improvements.append("集約・圧縮")
            
            # Column movement (entrance proximity and SKU consolidation)
            if tcol != from_col:
                # Check if moving toward entrance (higher column numbers)
                if tcol > from_col:
                    actions.append(f"入口側移動(列{from_col}→{tcol})")
                    improvements.append("動線短縮")
                else:
                    # Check if this is SKU consolidation
                    try:
                        same_sku_cols = set(
                            inv.loc[inv["商品ID"].astype(str) == str(sku_val), "col"]
                            .dropna().astype(int).unique().tolist()
                        )
                        if tcol in same_sku_cols:
                            actions.append(f"SKU集約(列{from_col}→{tcol})")
                            improvements.append("集約効果")
                        else:
                            actions.append(f"列移動(列{from_col}→{tcol})")
                            improvements.append("配置最適化")
                    except Exception:
                        actions.append(f"列移動(列{from_col}→{tcol})")
                        improvements.append("配置最適化")
            
            # Depth optimization (bring items forward)
            if tdep < from_dep:
                actions.append(f"手前化(奥{from_dep}→{tdep})")
                improvements.append("取出し容易化")
            
            # Mix slot usage
            if needs_mix:
                actions.append("混載枠活用")
                improvements.append("柔軟配置")
            
            # Eviction chain (space creation)
            if best_ev_chain:
                chain_len = len(best_ev_chain)
                actions.append(f"スペース確保({chain_len}手連鎖)")
                improvements.append("容量最適化")
            
            # Default if no specific actions identified
            if not actions:
                actions.append("配置最適化")
                improvements.append("圧縮・集約")
            
            move_reason = " & ".join(actions) + " → " + "、".join(improvements)
            
            # Create the main move
            # Set chain_group_id/execution_order if eviction chain was used
            main_chain_id = best_ev_chain_group_id if best_ev_chain else None
            main_exec_order = len(best_ev_chain) + 1 if main_chain_id else None
            
            move = Move(
                sku_id=sku_val,
                lot=str(row.get("ロット") or ""),
                qty=int(qty_cases),
                from_loc=str(from_loc).zfill(8),
                to_loc=str(to_loc).zfill(8),
                lot_date=_lot_key_to_datestr8(lot_key),
                reason=move_reason,
                chain_group_id=main_chain_id or f"p2consol_{secrets.token_hex(6)}",
                execution_order=main_exec_order or 1,
            )
            
            # Track this move for future lot-mixing checks
            lookup_key = (str(to_loc), str(sku_val))
            if lookup_key not in planned_lots_by_loc_sku:
                planned_lots_by_loc_sku[lookup_key] = set()
            planned_lots_by_loc_sku[lookup_key].add(int(lot_key))
            
            # Add eviction chain moves first
            moves.extend(best_ev_chain)
            # Then add the main move
            moves.append(move)
            
            # Update shelf usage
            shelf_usage[from_loc] = max(0, shelf_usage.get(from_loc, 0) - need_vol)
            shelf_usage[to_loc] = shelf_usage.get(to_loc, 0) + need_vol
    
    # Record Pass-2 statistics (main loop moves after Pass-0 and Pass-1)
    main_loop_moves = len(moves) - pass_stats["pass0"] - pass_stats["pass1"]
    pass_stats["pass2"] = main_loop_moves
    
    # Main loop (Pass-2) timing
    _log(f"Pass-2 completed: {main_loop_moves} moves in {time.perf_counter()-main_loop_t0:.2f}s, total={len(moves)}")
    try:
        _publish_progress(get_current_trace_id(), {
            "type": "phase", "name": "pass2", "moves": main_loop_moves, "total_moves": len(moves),
            "message": f"Pass-2 圧縮・集約: {main_loop_moves}件追加 (計{len(moves)}件)"
        })
    except Exception:
        pass
    
    # --- Pass-3: AI-driven optimization (additional refinement pass) ---
    if getattr(cfg, "enable_pass3_ai_optimize", False):
        p3_t0 = time.perf_counter()
        p3_start_count = len(moves)
        remaining = None if getattr(cfg, "max_moves", None) is None else int(getattr(cfg, "max_moves")) - len(moves)
        
        if remaining is None or remaining > 0:
            _log(f"Starting Pass-3 AI optimization, budget remaining: {remaining}")
            try:
                _publish_progress(get_current_trace_id(), {
                    "type": "phase", "name": "pass3_start", "moves": 0, "total_moves": len(moves),
                    "message": f"Pass-3 AI最適化を開始..."
                })
            except Exception:
                pass
            
            # Pass-3: Focus on AI hints and fine-tuning
            # Re-scan inventory for items that can benefit from AI column hints
            p3_order_idx = _sort_index_for_pick(inv)
            p3_processed = 0
            
            for idx in p3_order_idx:
                if getattr(cfg, "max_moves", None) is not None and len(moves) >= int(getattr(cfg, "max_moves")):
                    break
                    
                row = inv.loc[idx]
                p3_processed += 1
                
                from_loc = str(row["ロケーション"])
                lv, col, dep = row.get("lv"), row.get("col"), row.get("dep")
                if pd.isna(lv) or pd.isna(col) or pd.isna(dep):
                    continue
                lv, col, dep = int(lv), int(col), int(dep)
                
                # Pass-3 only considers items that can be further optimized
                # Skip if already at level 1
                if lv == 1:
                    continue
                
                # Skip if no AI hints available for this SKU
                sku_val = str(row["商品ID"])
                if not ai_col_hints or sku_val not in ai_col_hints:
                    continue
                
                qty_cases = int(row["qty_cases_move"]) or 0
                if qty_cases <= 0:
                    continue
                    
                need_vol = float(row["volume_each_case"]) * qty_cases
                lot_key = int(row.get("lot_key") or UNKNOWN_LOT_KEY)
                
                # Get AI recommended columns for this SKU
                ai_recommended_cols = ai_col_hints.get(sku_val, [])
                if not ai_recommended_cols:
                    continue
                
                # Try to move to AI-recommended columns at lower levels
                best_choice = None
                best_score = math.inf
                best_ev_chain = []
                
                for target_level in [1, 2]:  # Pass-3 focuses on pick levels
                    if target_level >= lv:
                        continue
                        
                    cand_locs = locs_by_level.get(target_level, [])
                    if can_receive_set:
                        cand_locs = [loc for loc in cand_locs if loc != from_loc and loc in can_receive_set]
                    else:
                        cand_locs = [loc for loc in cand_locs if loc != from_loc]
                    
                    for to_loc in cand_locs:
                        tlv, tcol, tdep = _parse_loc8(str(to_loc))
                        
                        # Prioritize AI-recommended columns
                        if int(tcol) not in ai_recommended_cols:
                            continue
                        
                        # Check capacity
                        used = float(shelf_usage.get(to_loc, 0.0))
                        limit = cap_by_loc.get(to_loc, cap_limit) if cap_by_loc else cap_limit
                        
                        if used + need_vol <= limit:
                            # Score based on AI hint priority (first hint = best)
                            col_priority = ai_recommended_cols.index(int(tcol)) if int(tcol) in ai_recommended_cols else 999
                            score = col_priority * 100 + tlv * 10 + tdep
                            
                            if score < best_score:
                                best_score = score
                                best_choice = (to_loc, target_level, tcol, tdep)
                                best_ev_chain = []
                
                if best_choice is not None:
                    to_loc, target_level, tcol, tdep = best_choice
                    
                    # Create Pass-3 move with AI-specific reason
                    from_lv, from_col, from_dep = _parse_loc8(from_loc)
                    
                    actions = []
                    improvements = []
                    
                    if target_level < from_lv:
                        actions.append(f"AI推奨段配置(Lv{from_lv}→{target_level})")
                        improvements.append("出荷実績最適化")
                    
                    if tcol in ai_recommended_cols:
                        hint_rank = ai_recommended_cols.index(tcol) + 1
                        actions.append(f"AI推奨列配置(列{from_col}→{tcol}/優先度{hint_rank})")
                        improvements.append("動線AI最適化")
                    
                    if not actions:
                        actions.append("AI総合最適化")
                        improvements.append("配置効率UP")
                    
                    move_reason = " & ".join(actions) + " → " + "、".join(improvements)
                    
                    move = Move(
                        sku_id=sku_val,
                        lot=str(row.get("ロット") or ""),
                        qty=int(qty_cases),
                        from_loc=str(from_loc).zfill(8),
                        to_loc=str(to_loc).zfill(8),
                        lot_date=_lot_key_to_datestr8(lot_key),
                        reason=move_reason,
                        chain_group_id=f"p3ai_{secrets.token_hex(6)}",
                        execution_order=1,
                    )
                    
                    moves.append(move)
                    
                    # Update state
                    shelf_usage[from_loc] = max(0, shelf_usage.get(from_loc, 0) - need_vol)
                    shelf_usage[to_loc] = shelf_usage.get(to_loc, 0) + need_vol
                    inv.at[idx, "ロケーション"] = str(to_loc)
                    inv.at[idx, "lv"] = int(target_level)
                    inv.at[idx, "col"] = int(tcol)
                    inv.at[idx, "dep"] = int(tdep)
        
        p3_moves = len(moves) - p3_start_count
        pass_stats["pass3"] = p3_moves
        _log(f"Pass-3 AI optimization: moves={p3_moves} time={time.perf_counter()-p3_t0:.2f}s total={len(moves)}")
        try:
            _publish_progress(get_current_trace_id(), {
                "type": "phase", "name": "pass3", "moves": p3_moves, "total_moves": len(moves),
                "message": f"Pass-3 AI最適化: {p3_moves}件追加 (計{len(moves)}件)"
            })
        except Exception:
            pass
    
    # --- Safety cap already applied during candidate selection ---
    # moves list is already sorted by Pass priority and limited to max_moves
    
    try:
        print(f"[optimizer] planned moves={len(moves)} (limit={getattr(cfg,'max_moves',None)})")
        _publish_progress(get_current_trace_id(), {
            "type": "planned", "count": len(moves),
            "message": f"移動案作成完了: {len(moves)}件（Pass優先度順ソート済み）"
        })
    except Exception:
        pass

    # Auto-apply hard constraints and record debug counters
    if getattr(cfg, "auto_enforce", True):
        print(f"[optimizer] Auto-enforce is enabled, starting constraint enforcement")
        try:
            lm_df = locals().get("lm_scoped", loc_master)
        except Exception as e:
            print(f"[optimizer] Failed to get lm_scoped: {e}")
            lm_df = loc_master
        try:
            _publish_progress(get_current_trace_id(), {
                "type": "info", "phase": "enforce",
                "message": f"制約チェック開始: {len(moves)}件を検証中..."
            })
        except Exception as e:
            print(f"[optimizer] Failed to publish enforce start progress: {e}")
            pass
        
        print(f"[optimizer] Calling enforce_constraints with {len(moves)} moves")
        accepted = enforce_constraints(
            sku_master=sku_master,
            inventory=inv,
            moves=moves,
            cfg=cfg,
            loc_master=lm_df,
        )
        print(f"[optimizer] enforce_constraints completed: {len(accepted)} accepted out of {len(moves)}")
        
        # Resolve move dependencies (to_loc/from_loc conflicts)
        try:
            accepted = _resolve_move_dependencies(accepted)
            print(f"[optimizer] Dependency resolution completed: {len(accepted)} moves")
        except Exception as e:
            print(f"[optimizer] WARNING: Dependency resolution failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Generate comprehensive summary report
        rejected_count = len(moves) - len(accepted)
        try:
            print(f"[optimizer] Generating summary report: accepted={len(accepted)}, rejected={rejected_count}")
            summary_report = _generate_relocation_summary(
                inv_before=inv,
                moves=accepted,
                rejected_count=rejected_count,
                pass_stats=pass_stats,
            )
            print(f"[optimizer] Summary report generated: {list(summary_report.keys())}")
            
            # Format report as readable text
            report_lines = []
            report_lines.append("=" * 70)
            report_lines.append("📊 リロケーション結果の総合評価")
            report_lines.append("=" * 70)
            report_lines.append("")
            report_lines.append("【実施結果】")
            report_lines.append(f"  計画移動数: {summary_report.get('total_planned', 0):,} 件")
            report_lines.append(f"  承認移動数: {summary_report.get('total_accepted', 0):,} 件")
            report_lines.append(f"  却下移動数: {summary_report.get('total_rejected', 0):,} 件")
            report_lines.append(f"  移動率: {summary_report.get('move_rate_percent', 0):.1f}%")
            report_lines.append(f"  影響SKU数: {summary_report.get('affected_skus', 0):,} 種類")
            report_lines.append(f"  総ケース数: {summary_report.get('total_cases', 0):,} ケース")
            report_lines.append("")
            
            # === New evaluation metrics ===
            report_lines.append("【最適化効果】")
            
            # Pass-wise breakdown (実行順: Pass-1 → Pass-0 → Pass-2 → Pass-3)
            pass_stats_from_summary = summary_report.get('pass_stats', {})
            if pass_stats_from_summary and any(pass_stats_from_summary.values()):
                report_lines.append("  ▶ Pass毎の改善 (実行順):")
                if pass_stats_from_summary.get("pass1", 0) > 0:
                    report_lines.append(f"    • Pass-1 (取口/保管整列) 【最優先】: {pass_stats_from_summary['pass1']:,}件")
                    report_lines.append(f"      → 同一SKU内で古いロットを取り口へ、新しいロットを保管段へ")
                if pass_stats_from_summary.get("pass0", 0) > 0:
                    report_lines.append(f"    • Pass-0 (エリア再配置) 【第2優先】: {pass_stats_from_summary['pass0']:,}件")
                    report_lines.append(f"      → 入数帯の是正、適切な列への移動")
                if pass_stats_from_summary.get("pass2", 0) > 0:
                    report_lines.append(f"    • Pass-2 (圧縮/集約) 【第3優先】: {pass_stats_from_summary['pass2']:,}件")
                    report_lines.append(f"      → SKU集約、下段優先配置、動線最適化")
                if pass_stats_from_summary.get("pass3", 0) > 0:
                    report_lines.append(f"    • Pass-3 (AI最適化) 【第4優先】: {pass_stats_from_summary['pass3']:,}件")
                    report_lines.append(f"      → 出荷実績AIヒント、高度な配置最適化")
                report_lines.append("")
            
            old_lot_count = summary_report.get('old_lot_to_pick_count', 0)
            hotspot_count = summary_report.get('hotspot_moves_count', 0)
            
            if old_lot_count > 0:
                report_lines.append(f"  📦 古いロット→取り口ロケ: {old_lot_count:,} 件")
                report_lines.append(f"     (同一SKU内で相対的に古いロットをレベル1-2へ移動)")
            else:
                report_lines.append(f"  📦 古いロット→取り口ロケ: なし")
            
            if hotspot_count > 0:
                hotspot_criteria = summary_report.get('hotspot_criteria', '')
                report_lines.append(f"  🔥 高出荷SKU→ホットスポット: {hotspot_count:,} 件")
                report_lines.append(f"     ({hotspot_criteria})")
            else:
                report_lines.append(f"  🔥 高出荷SKU→ホットスポット: なし")
            
            # Move reason breakdown
            reason_breakdown = summary_report.get('reason_breakdown', {})
            if reason_breakdown:
                report_lines.append("")
                report_lines.append("  ▶ 移動理由別の内訳:")
                for reason, count in list(reason_breakdown.items())[:10]:  # Top 10
                    report_lines.append(f"    • {reason}: {count:,}件")
            
            report_lines.append("")
            
            # SKU consolidation
            if summary_report.get('sku_consolidation'):
                before = summary_report['sku_consolidation'].get('before', {})
                after = summary_report['sku_consolidation'].get('after', {})
                before_detail = summary_report['sku_consolidation'].get('before_detail', {})
                after_detail = summary_report['sku_consolidation'].get('after_detail', {})
                
                # 変化があるSKUのみフィルタリング
                changed_skus = [sku for sku in before.keys() if before.get(sku, 0) != after.get(sku, 0)]
                
                if changed_skus:
                    report_lines.append("【SKU集約状況】")
                    
                    for sku in changed_skus[:5]:  # 最大5件
                        b_locs = before.get(sku, 0)
                        a_locs = after.get(sku, 0)
                        b_locs_list = before_detail.get(sku, [])
                        a_locs_list = after_detail.get(sku, [])
                        
                        if b_locs > a_locs:
                            improvement = f"✅ {b_locs}→{a_locs}ロケ"
                        elif b_locs < a_locs:
                            improvement = f"⚠️ {b_locs}→{a_locs}ロケ (分散)"
                        else:
                            continue  # Skip unchanged (safety check)
                        
                        report_lines.append(f"  {sku}: {improvement}")
                        
                        # 移動前のロケーション一覧（最大5件、7桁形式対応）
                        if b_locs_list:
                            display_count = min(5, len(b_locs_list))
                            locs_display = ', '.join(b_locs_list[:display_count])
                            if len(b_locs_list) > display_count:
                                locs_display += f" ...他{len(b_locs_list)-display_count}件"
                            report_lines.append(f"     移動前 ({b_locs}ロケ): {locs_display}")
                        
                        # 移動後のロケーション一覧（最大5件、7桁形式対応）
                        if a_locs_list:
                            display_count = min(5, len(a_locs_list))
                            locs_display = ', '.join(a_locs_list[:display_count])
                            if len(a_locs_list) > display_count:
                                locs_display += f" ...他{len(a_locs_list)-display_count}件"
                            report_lines.append(f"     移動後 ({a_locs}ロケ): {locs_display}")
                        
                        report_lines.append("")  # 空行で区切る
                    
                    # 最後の空行を削除
                    if report_lines and report_lines[-1] == "":
                        report_lines.pop()
            
            # Hard rules validation
            report_lines.append("【ハードルール検証】")
            lot_issues = summary_report.get('lot_mixing_issues', 0)
            if lot_issues > 0:
                report_lines.append(f"  ⚠️ ロット混在: {lot_issues}件検出")
                for detail in summary_report.get('lot_mixing_details', [])[:3]:
                    report_lines.append(f"     - {detail['location']} / {detail['sku']}: {detail['lot_count']}種類のロット")
            else:
                report_lines.append("  ✅ ロット混在なし")
            
            if summary_report.get('quality_breakdown'):
                qb = summary_report['quality_breakdown']
                report_lines.append(f"  良品在庫: {qb.get('good_items', 0):,} 行 / 全体 {qb.get('total_items', 0):,} 行")
            report_lines.append("")
            
            # Recommendations
            if summary_report.get('recommendations'):
                report_lines.append("【推奨事項】")
                for rec in summary_report['recommendations']:
                    report_lines.append(f"  • {rec}")
                report_lines.append("")
            
            report_lines.append("=" * 70)
            
            report_text = "\n".join(report_lines)
            print(f"[optimizer] Report formatted, {len(report_lines)} lines")
            
            # Store report for later publishing and retrieval
            global _LAST_SUMMARY_REPORT, _SUMMARY_REPORTS
            _LAST_SUMMARY_REPORT = report_text
            
            # trace_id別に保存（複数の最適化結果を保持可能）
            current_trace_id = get_current_trace_id()
            if current_trace_id:
                _SUMMARY_REPORTS[current_trace_id] = report_text
                # 古い結果を削除（メモリ管理）
                if len(_SUMMARY_REPORTS) > _SUMMARY_REPORTS_LIMIT:
                    oldest_keys = sorted(_SUMMARY_REPORTS.keys())[:len(_SUMMARY_REPORTS) - _SUMMARY_REPORTS_LIMIT]
                    for k in oldest_keys:
                        _SUMMARY_REPORTS.pop(k, None)
            
            summary_report_text = report_text
            summary_report_data = summary_report
            
            # Also log to console
            print("[optimizer] Summary Report:")
            print(report_text)
            
        except Exception as e:
            import traceback
            print(f"[optimizer] Failed to generate summary report: {e}")
            print(f"[optimizer] Traceback: {traceback.format_exc()}")
            summary_report_text = None
            summary_report_data = None
            pass
        
        # Send done event first
        try:
            _publish_progress(get_current_trace_id(), {
                "type": "done", "accepted": len(accepted),
                "message": f"最適化完了: {len(accepted)}件の移動案を承認"
            })
        except Exception:
            pass
        
        # Then send summary report (no delay needed)
        if summary_report_text:
            try:
                _publish_progress(get_current_trace_id(), {
                    "type": "summary_report",
                    "report": summary_report_text,
                    "data": summary_report_data,
                    "message": "総合評価レポート生成完了"
                })
                print("[optimizer] Summary report published via SSE")
            except Exception as e:
                print(f"[optimizer] Failed to publish summary report: {e}")
        
        # Cache moves for timeout recovery
        current_trace = get_current_trace_id()
        if current_trace and accepted:
            moves_data = [m.to_dict() if hasattr(m, 'to_dict') else m.__dict__ for m in accepted]
            cache_moves(current_trace, moves_data)
            print(f"[optimizer] Cached {len(moves_data)} moves for trace_id={current_trace}")
        
        return accepted

    return moves