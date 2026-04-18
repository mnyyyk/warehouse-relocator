from __future__ import annotations

import logging
import math
import re
import threading
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict, Set, Any
import asyncio
import json
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache

import pandas as pd
import numpy as np
import copy
import os
import secrets

logger = logging.getLogger(__name__)

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
        "level_vol_cap": 0,
        "foreign_sku": 0,
        "lot_mix": 0,
        "fifo": 0,
        "pack_band": 0,
        "chain_incomplete": 0,
        "other": 0,
    },
    "examples": {
        "oversize": [],
        "forbidden": [],
        "capacity": [],
        "level_vol_cap": [],
        "foreign_sku": [],
        "lot_mix": [],
        "fifo": [],
        "pack_band": [],
        "chain_incomplete": [],
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
            "level_vol_cap": 0,
            "foreign_sku": 0,
            "lot_mix": 0,
            "fifo": 0,
            "pack_band": 0,
            "chain_incomplete": 0,
            "other": 0,
        },
        "examples": {
            "oversize": [],
            "forbidden": [],
            "capacity": [],
            "level_vol_cap": [],
            "foreign_sku": [],
            "lot_mix": [],
            "fifo": [],
            "pack_band": [],
            "chain_incomplete": [],
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
_OPTIMIZER_LOCK = threading.Lock()

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
_MAX_TRACE_BUFFERS = 20  # _TRACE_BUFFER / _TRACE_SUBS に保持する最大トレース件数

def _publish_progress(trace_id: Optional[str], event: Dict[str, Any]) -> None:
    """Publish a progress event to subscribers and store in a small ring buffer."""
    try:
        if not trace_id:
            return
        tid = str(trace_id)
        payload = dict(event)
        payload.setdefault("ts", time.time())
        data = json.dumps(payload, ensure_ascii=False)

        # 古いトレースバッファをクリーンアップ（直近 _MAX_TRACE_BUFFERS 件のみ保持）
        if len(_TRACE_BUFFER) > _MAX_TRACE_BUFFERS:
            old_tids = list(_TRACE_BUFFER.keys())[:-_MAX_TRACE_BUFFERS]
            for old_tid in old_tids:
                _TRACE_BUFFER.pop(old_tid, None)
                old_subs = _TRACE_SUBS.pop(old_tid, [])
                for q in old_subs:
                    try:
                        q.put_nowait('{"type":"closed"}')
                    except Exception:
                        pass

        # Debug log for summary_report events
        if payload.get("type") == "summary_report":
            logger.debug(f"[_publish_progress] Publishing summary_report to trace_id={tid}")
            logger.debug(f"[_publish_progress] Payload keys: {list(payload.keys())}")
            logger.debug(f"[_publish_progress] Report length: {len(payload.get('report', ''))}")
            logger.debug(f"[_publish_progress] Subscribers count: {len(_TRACE_SUBS.get(tid) or [])}")

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
                logger.debug(f"[optimizer] trace bound (auto): {gen_id}")
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

_MAX_DROP_TRACES = 10  # _DROP_STORE に保持する最大トレース件数

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
    # LRU クリーンアップ: 古いエントリを削除して直近 _MAX_DROP_TRACES 件のみ保持
    if len(_DROP_STORE) > _MAX_DROP_TRACES:
        oldest_keys = list(_DROP_STORE.keys())[:-_MAX_DROP_TRACES]
        for k in oldest_keys:
            del _DROP_STORE[k]

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


def _consolidate_moves_for_summary(moves: List["Move"]) -> List["Move"]:
    """統合前の移動リストを、同一 SKU+ロット の最初の from_loc と最後の to_loc に
    圧縮する。中間ステップ（退避チェーン）を除去し、サマリーレポートや検証で
    正確な before/after 状態を計算できるようにする。
    from_loc == to_loc（元の場所に戻る）移動は除外する。
    """
    from collections import OrderedDict
    chains: OrderedDict[tuple, list] = OrderedDict()
    for m in moves:
        key = (m.sku_id, m.lot)
        if key not in chains:
            chains[key] = []
        chains[key].append(m)

    result: List["Move"] = []
    for (sku_id, lot), chain in chains.items():
        first_from = chain[0].from_loc
        last_to = chain[-1].to_loc
        if first_from == last_to:
            continue  # 元の場所に戻る → 実質移動なし
        # 最後の Move をベースに from_loc だけ差し替え
        base = chain[-1]
        result.append(Move(
            sku_id=base.sku_id,
            lot=base.lot,
            qty=base.qty,
            from_loc=first_from,
            to_loc=last_to,
            lot_date=base.lot_date,
            reason=base.reason,
            chain_group_id=base.chain_group_id,
            execution_order=base.execution_order,
            distance=base.distance,
        ))
    return result


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

    # 退避チェーンの中間ステップを統合し、最終的な移動のみで検証する
    moves = _consolidate_moves_for_summary(moves)
    if len(moves) == 0:
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
    logger.debug(f"[_generate_relocation_summary] Starting: moves={len(moves)}, rejected={rejected_count}")
    
    summary = {
        "total_planned": len(moves) + rejected_count,
        "pass_stats": pass_stats or {},
        "total_rejected": rejected_count,
        "total_accepted": len(moves),
    }
    
    if len(moves) == 0:
        summary["message"] = "移動案なし"
        logger.debug(f"[_generate_relocation_summary] No moves, returning early")
        return summary

    # 退避チェーンの中間ステップを統合し、最終移動のみでサマリーを計算する
    consolidated = _consolidate_moves_for_summary(moves)

    # Pass stats を統合後ベースで再計算
    def _pass_priority(m: "Move") -> int:
        reason = str(m.reason or "")
        # Pass-C: 集約（同一ロット統合）
        if "同一ロット統合" in reason:
            return 0
        # Pass-1: FIFO / 取口保管整列
        if any(k in reason for k in (
            "古ロット", "新ロット", "先入先出", "取口配置", "取口内整列", "取口内再配置",
            "保管内整列", "保管内再配置", "FIFO是正", "最古ロット優先", "FIFO順序",
            "段下げ", "段上げ", "同列内移動", "手前化", "同一レベル内移動",
            "スワップ準備退避",
        )):
            return 1
        # Pass-0: エリア再配置 / ゾーンスワップ
        if any(k in reason for k in (
            "入数帯是正", "エリア移動", "エリア", "列移動", "入口に近づく",
            "入口接近", "適正エリア", "位置最適化", "ゾーンスワップ",
        )):
            return 2
        # Pass-2: 圧縮 / 集約
        if any(k in reason for k in (
            "圧縮", "集約", "SKU集約", "混載枠", "配置最適化", "スペース確保",
            "大幅段下げ", "容量確保退避",
        )):
            return 3
        # Pass-3: AI
        if "AI" in reason:
            return 4
        return 5

    summary["pass_stats"] = {
        "pass1": sum(1 for m in consolidated if _pass_priority(m) == 1),
        "pass0": sum(1 for m in consolidated if _pass_priority(m) == 2),
        "pass2": sum(1 for m in consolidated if _pass_priority(m) == 3),
        "pass3": sum(1 for m in consolidated if _pass_priority(m) == 4),
    }

    # Extract move details (統合後ベース)
    move_skus = set(m.sku_id for m in consolidated)
    move_from_locs = set(m.from_loc for m in consolidated)
    move_to_locs = set(m.to_loc for m in consolidated)
    total_cases = sum(m.qty for m in consolidated)
    
    summary["final_move_count"] = len(consolidated)
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
            
            # After: original locations - from_locs + to_locs (統合後ベース)
            sku_moves = [m for m in consolidated if m.sku_id == sku]
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
    
    # Lot mixing check (same loc + same SKU + different lots) — 統合後ベース
    lot_mixing_issues = []
    loc_sku_lots: Dict[Tuple[str, str], Set[str]] = {}

    for m in consolidated:
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
    
    # Analyze move reasons (group by reason and count) — 統合後ベース
    reason_counts: Dict[str, int] = {}
    for m in consolidated:
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
    
    for m in consolidated:
        try:
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
    
    for m in consolidated:
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
    
    move_rate = len(consolidated) / len(inv_before) * 100 if len(inv_before) > 0 else 0
    if move_rate < 2:
        recommendations.append("移動率が低い（<2%）- より積極的な最適化が可能かもしれません")

    if lot_mixing_issues:
        recommendations.append(f"⚠️ ロット混在が{len(lot_mixing_issues)}件検出されました - ハードルール違反の可能性")

    if len(move_skus) < 10 and len(consolidated) > 50:
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
        logger.debug(f"[_generate_relocation_summary] Validation failed: {e}")
        summary["post_move_validation"] = {"error": str(e)}
    
    logger.debug(f"[_generate_relocation_summary] Completed: keys={list(summary.keys())}")
    return summary


# -------------------------------
@dataclass
class OptimizerConfig:
    # 0 or None -> unlimited moves
    max_moves: Optional[int] = None
    fill_rate: float = DEFAULT_FILL_RATE
    # 入数の許容レンジ（±比率）
    pack_tolerance_ratio: float = 0.10
    # 既存マッピングの"ほどよい"維持のための支配的入数のしきい値
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
    near_cols: tuple[int, ...] = tuple(range(24, 35))  # 24–34（small_pack_cols と一致）
    far_cols:  tuple[int, ...] = tuple(range(1, 12))   #  1–11 を「遠い帯」として優先
    band_pref_weight: float = 20.0               # 帯嗜好の効き具合（負がボーナスになる）
    promo_quality_keywords: tuple[str, ...] = ("販促資材", "販促", "什器", "資材")
    # --- Pick/Storage levels and area zoning ---
    pick_levels: tuple[int, ...] = (1, 2)           # 取口=1-2段
    storage_levels: tuple[int, ...] = (3, 4)        # 保管=3-4段
    # --- Pass toggles (フロントエンドからON/OFF制御) ---
    enable_pass_consolidate: bool = True   # 同一ロット集約
    enable_pass_fifo: bool = True          # FIFO是正（古いロット→取口）
    enable_pass_zone_swap: bool = True     # ゾーンスワップ（入数帯入替）
    enable_main_loop: bool = True          # 段下げ・動線最適化
    # --- Pass-0: Area (column band) rebalance control ---
    enable_pass0_area_rebalance: bool = True
    # 入数<=2 の緩和を Pass-0 でも尊重（True=帯外でも対象外とする）
    pass0_respect_smallpack_relax: bool = True
    # Pass-0 で候補とするターゲット段（優先順: 現在段→他段）
    pass0_target_levels: tuple[int, ...] = (1, 2, 3, 4)
    # Pass-0 の候補ロケーション数上限（パフォーマンス最適化）
    pass0_max_candidates_per_item: int = 50
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
    # 各列に許容する"ミックス列"枠（±10%を外れても許す枠数）
    mix_slots_per_col: int = 1
    # ミックス枠を消費する場合の微小なコスト
    mix_usage_penalty: float = 1.0
    # AI優先列配置のボーナス
    ai_preferred_column_bonus: float = 25.0
    # 絶対制約スイッチ類（AI案の最終ゲートなどで使用）
    hard_cap: bool = False            # 容量をハードに守る
    hard_fifo: bool = False           # リグレッションゲートでFIFO悪化を防止（ハードゲートは不要）
    strict_pack: Optional[str] = None # "A", "B", "A+B" を指定可（Noneは従来通り）
    exclude_oversize: bool = False    # 1ケースが棚容積上限を超えるSKUを除外
    # --- Time limit for main loop (メインループの時間制限) ---
    loop_time_limit: float = 1800.0   # seconds (30 minutes) - chain_depth=1で全行処理に必要
    # --- Eviction chain (bounded multi-step relocation) ---
    # chain_depth=1: 浅い連鎖のみ許可（パフォーマンスと効果のバランス）
    chain_depth: int = 3            # 0=disabled; 1=shallow; 2=deeper chains
    # 既定で控えめに有効化（性能と効果のバランス）
    eviction_budget: int = 200       # max number of eviction (auxiliary) moves
    touch_budget: int = 2000         # max number of distinct locations we can touch
    buffer_slots: int = 0           # reserved empty-ish slots for temporary staging (0=disabled)
    # Allow same-level moves if (column,depth) strictly improve ease
    allow_same_level_ease_improve: bool = True
    # Diagnostics / pipeline
    trace_id: Optional[str] = None
    # If True, run a hard-gate pass here to both finalize moves and record drop reasons
    auto_enforce: bool = True
    # --- Depth preference (奥行きの優先度) ---
    # 'front'  = 従来通り「浅いほど良い」
    # 'center' = 列ごとの奥行き数に応じて"中心に近いほど良い"（山形）
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
    max_source_locs_per_sku: Optional[int] = None

    # --- 元々空きだったロケへの複数SKU同居許可設定 ---
    # 元々複数SKU混在のロケは従来通り一切触らない（L6518-6562 のロジック維持）
    # 元々空きだったロケに限り、段別ルールで複数SKU同居を許可する
    allow_empty_loc_multi_sku: bool = True
    multi_sku_max_per_loc: int = 3
    multi_sku_level1_vol_cap: float = 0.9
    multi_sku_level2_vol_cap: float = 0.3
    multi_sku_pack_band_match: bool = True
    multi_sku_target_levels: tuple = (1, 2)
    # 複数SKU同居を許可するchain_group_idプレフィックス集合
    multi_sku_allowed_chain_prefixes: tuple = ("p1fifo", "swap_fifo_", "fifo_direct_", "p0rebal_", "p2consol_")
    # Lv1/Lv2 着地で段別容積上限を全着地（単独SKU含む）に常時適用
    enforce_level_vol_cap: bool = True


def _select_candidate_moves(
    moves_with_meta: List[Tuple[Move, int, float]],
    max_moves: Optional[int],
    max_source_locs_per_sku: Optional[int],
) -> Tuple[List[Move], int]:
    """候補リストに上限系の制約のみを適用して採択する。
    スワップペア（swap_/fifo_direct_/zone_swap_）は分断しない。"""
    selected: List[Move] = []
    sku_source_locs: Dict[str, Set[str]] = {}
    skipped_by_limit = 0
    _selected_chains: Set[str] = set()  # 既に selected に入ったチェーンID
    _seen_move_keys: Set[Tuple[str, str, str, str]] = set()  # dedup: (sku, lot, from, to)

    for m, _priority, _gain in moves_with_meta:
        _mk = (str(m.sku_id), str(m.lot), str(m.from_loc).zfill(8), str(m.to_loc).zfill(8))
        if _mk in _seen_move_keys:
            continue
        _seen_move_keys.add(_mk)
        cg = getattr(m, 'chain_group_id', None) or ""
        is_swap = cg.startswith("swap_") or cg.startswith("fifo_direct_") or cg.startswith("zone_swap_")

        # ペアの片方が既に selected にあるなら、もう片方は無条件で追加
        if is_swap and cg in _selected_chains:
            selected.append(m)
            continue

        if max_moves is not None and len(selected) >= max_moves:
            break

        if max_source_locs_per_sku is not None:
            sku_id = str(m.sku_id)
            from_loc = str(m.from_loc)

            if sku_id not in sku_source_locs:
                sku_source_locs[sku_id] = set()

            if from_loc not in sku_source_locs[sku_id]:
                if len(sku_source_locs[sku_id]) >= max_source_locs_per_sku:
                    skipped_by_limit += 1
                    continue
                sku_source_locs[sku_id].add(from_loc)

        selected.append(m)
        if is_swap:
            _selected_chains.add(cg)

    return selected, skipped_by_limit
    
# -------------------------------
# Additional helpers for hard constraints and pack clustering
# -------------------------------

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

def _resolve_move_dependencies(
    moves: List[Move],
    original_qty_by_loc_sku: Optional[Dict[Tuple[str, str], float]] = None,
) -> List[Move]:
    """Resolve execution order dependencies between moves.

    If Move A wants to place items at location X, but Move B needs to
    remove items FROM location X first, then B must execute before A.
    This function:
      1. Detects to_loc/from_loc conflicts (skipping self-referencing
         moves where from_loc == to_loc)
      2. Groups dependent moves under the same chain_group_id (dep_...)
      3. Assigns correct execution_order starting from 1 (evacuate first)
      4. Reorders the list so that within each group, the evacuator
         always appears before the placer in the CSV
      5. If original_qty_by_loc_sku is provided, detects partial evacuations
         (evac move qty < loc qty) and rejects the subsequent placer move to
         avoid multi-SKU mixing at the destination.

    Returns a new list of Move objects with updated chain_group_id and
    execution_order where dependencies exist.
    """
    if not moves:
        return moves

    # Build index: to_loc -> list of move indices that want to place there
    to_loc_index: Dict[str, List[int]] = defaultdict(list)
    # Build index: from_loc -> list of move indices that evacuate from there
    from_loc_index: Dict[str, List[int]] = defaultdict(list)

    for i, m in enumerate(moves):
        # Skip self-referencing moves (from_loc == to_loc) from BOTH
        # indices.  These represent in-place rearrangements (level/depth
        # changes within the same 8-digit slot code) and must NOT create
        # dependency edges that would misorder other moves.
        if m.from_loc == m.to_loc:
            continue
        # Skip swap chain_group moves — they are already self-contained
        # atomic pairs and should NOT be merged into dependency chains
        _cg = getattr(m, 'chain_group_id', None) or ""
        if _cg.startswith("swap_") or _cg.startswith("fifo_direct_") or _cg.startswith("zone_swap_"):
            continue
        to_loc_index[m.to_loc].append(i)
        from_loc_index[m.from_loc].append(i)

    # Find dependency pairs: move A wants to_loc X, move B from_loc X
    # B must execute before A
    # dep_graph[i] = set of indices that must execute BEFORE move i
    dep_graph: Dict[int, set] = defaultdict(set)
    # reverse: who depends on me
    reverse_deps: Dict[int, set] = defaultdict(set)

    # Track placer indices rejected due to partial evacuation
    _partial_evac_rejected: Set[int] = set()

    for loc, placers in to_loc_index.items():
        evacuators = from_loc_index.get(loc, [])
        for placer_idx in placers:
            if placer_idx in _partial_evac_rejected:
                continue
            placer_move = moves[placer_idx]
            for evac_idx in evacuators:
                if placer_idx == evac_idx:
                    continue
                evac_move = moves[evac_idx]
                # Check for partial evacuation: evac_move leaves SKU behind at from_loc
                if original_qty_by_loc_sku is not None:
                    evac_from = str(evac_move.from_loc).zfill(8)
                    evac_sku = str(evac_move.sku_id)
                    loc_qty = original_qty_by_loc_sku.get((evac_from, evac_sku), 0.0)
                    move_qty = float(evac_move.qty)
                    # 閾値 1e-9: float 精度だけをガード、物理的な残留は全て検出
                    # (下流の _sim_remaining_qty は `<= 0` で判定するため、残留 > 0 は常に "SKU残留" として扱う)
                    if loc_qty > move_qty + 1e-9:
                        # Partial evacuation: evac_move.sku_id will remain at evac_from.
                        # If placer wants to arrive at evac_from with a different SKU,
                        # this would cause multi-SKU mixing — reject the placer.
                        if (str(placer_move.to_loc).zfill(8) == evac_from
                                and str(placer_move.sku_id) != evac_sku):
                            _partial_evac_rejected.add(placer_idx)
                            logger.debug(
                                f"[dep_resolution] Partial evac reject: placer "
                                f"{placer_move.sku_id}->{evac_from} blocked by partial evac of "
                                f"{evac_sku} (loc={loc_qty:.4f}ケース, evac={move_qty:.4f}ケース, "
                                f"residual={loc_qty - move_qty:.4f}ケース)"
                            )
                            break  # no need to check other evacuators for this placer
                # evac must happen before placer
                if placer_idx not in _partial_evac_rejected:
                    dep_graph[placer_idx].add(evac_idx)
                    reverse_deps[evac_idx].add(placer_idx)

    # Remove rejected placers from dep_graph and reverse_deps (全方向クリーンアップ)
    if _partial_evac_rejected:
        logger.warning(
            f"[dep_resolution] Rejected {len(_partial_evac_rejected)} placer moves due to partial evacuation"
        )
        for rej_idx in _partial_evac_rejected:
            # rej_idx の outgoing edges (dep_graph[rej_idx]) を reverse_deps から除去
            if rej_idx in dep_graph:
                for evac_idx in list(dep_graph[rej_idx]):
                    reverse_deps[evac_idx].discard(rej_idx)
                del dep_graph[rej_idx]
            # rej_idx の incoming edges (reverse_deps[rej_idx]) を他ノードの dep_graph から除去
            if rej_idx in reverse_deps:
                for dep_idx in list(reverse_deps[rej_idx]):
                    if dep_idx in dep_graph:
                        dep_graph[dep_idx].discard(rej_idx)
                del reverse_deps[rej_idx]
        # 念のため全ノードの set から rej_idx 群を一括除去 (上記で漏れがあった場合の防御)
        for _dset in dep_graph.values():
            _dset.difference_update(_partial_evac_rejected)
        for _dset in reverse_deps.values():
            _dset.difference_update(_partial_evac_rejected)

    # dep_graph が空でも、partial-evac reject があれば最終フィルタで moves から除外する必要がある
    if not dep_graph and not _partial_evac_rejected:
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
    _cycle_chains: Set[str] = set()

    for group_indices in groups:
        group_chain_id = f"dep_{secrets.token_hex(6)}"

        if len(group_indices) < 2:
            # Single-member dependency node: still assign dep_ chain_group_id
            # and renumber execution_order to 1
            idx = next(iter(group_indices))
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
                execution_order=1,
                distance=old_move.distance,
            )
            dep_count += 1
            continue

        # Topological sort within the group (in-degree based / Kahn's)
        in_degree: Dict[int, int] = {idx: 0 for idx in group_indices}
        for idx in group_indices:
            for dep in dep_graph.get(idx, set()):
                if dep in group_indices:
                    in_degree[idx] = in_degree.get(idx, 0) + 1

        queue = [idx for idx in group_indices if in_degree[idx] == 0]
        topo_order = []
        while queue:
            queue.sort(key=lambda x: x)
            node = queue.pop(0)
            topo_order.append(node)
            for dependent in reverse_deps.get(node, set()):
                if dependent in group_indices:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        # Handle cycles: サイクルに含まれる移動は実行順序が決定できないため除外
        remaining = group_indices - set(topo_order)
        if remaining:
            logger.warning(
                f"[optimizer] dependency cycle detected in chain group: "
                f"{len(remaining)} moves removed (indices: {sorted(remaining)[:5]}...)"
            )
            for _ci in remaining:
                _cg = getattr(result[_ci], 'chain_group_id', None)
                if _cg:
                    _cycle_chains.add(_cg)
            # サイクルノードは topo_order に追加しない（除外）

        # Assign chain_group_id and execution_order to ALL members
        # execution_order always starts from 1
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

    # ------------------------------------------------------------------
    # Reorder: guarantee that within each dependency group the CSV row
    # order matches execution_order (evacuator rows come before placers).
    #
    # Strategy: collect all dep-group member indices, then rebuild the
    # final list by *replacing* each group block with its topo-sorted
    # version at the position of the group's earliest original member.
    # ------------------------------------------------------------------
    dep_indices = set()
    # Map: original index -> which group id (int) it belongs to
    idx_to_group: Dict[int, int] = {}
    for gid, g in enumerate(groups):
        dep_indices.update(g)
        for idx in g:
            idx_to_group[idx] = gid

    # For each group, build execution-order-sorted list of indices
    group_sorted: Dict[int, List[int]] = {}
    for gid, g in enumerate(groups):
        group_sorted[gid] = sorted(g, key=lambda i: result[i].execution_order or 0)

    final_list: List[Move] = []
    emitted_groups: set = set()

    for i in range(len(result)):
        if i in dep_indices:
            gid = idx_to_group[i]
            if gid not in emitted_groups:
                # Emit the entire group in execution_order at the
                # position of its earliest original member
                for idx in group_sorted[gid]:
                    final_list.append(result[idx])
                emitted_groups.add(gid)
            # else: already emitted – skip this member
        else:
            final_list.append(result[i])

    if dep_count > 0:
        logger.debug(f"[optimizer] Dependency resolution: {dep_count} moves in {len([g for g in groups if len(g) >= 2])} dependency groups")

    # H7: サイクルチェーンに含まれる移動を最終出力から除外
    if _cycle_chains:
        before_cycle_filter = len(final_list)
        final_list = [m for m in final_list if getattr(m, 'chain_group_id', None) not in _cycle_chains]
        logger.warning(f"[optimizer] Cycle filter: removed {before_cycle_filter - len(final_list)} moves in {len(_cycle_chains)} cycle groups")

    # 部分退避でリジェクトされた move を最終出力から除外
    # identity (id()) ベースで照合することで、同じ (from,to,sku) を持つ別 move を巻き込まないようにする
    if _partial_evac_rejected:
        before_rej = len(final_list)
        _rej_obj_ids = {id(moves[rej_idx]) for rej_idx in _partial_evac_rejected}
        final_list = [m for m in final_list if id(m) not in _rej_obj_ids]
        if len(final_list) < before_rej:
            logger.warning(f"[dep_resolution] Partial evac filter: removed {before_rej - len(final_list)} moves")

    return final_list


def _post_resolution_validate(
    moves: List[Move],
    original_skus_by_loc: Dict[str, Set[str]],
    original_lots_by_loc_sku: Dict[Tuple[str, str], Set[int]],
    blocked_dest_locs: Optional[Set[str]] = None,
    original_qty_by_loc_sku: Optional[Dict[Tuple[str, str], float]] = None,
    original_inv: Optional[pd.DataFrame] = None,
) -> List[Move]:
    """最終実行順序でSKU/ロット混在を再検証し、違反移動を除去する。

    enforce_constraints は移動をパス順に検証するが、
    _resolve_move_dependencies で並び替えた後の実行順序では
    退避が後回しになり混在が発生する場合がある。
    この関数は最終順序でシミュレーションし、違反を除去する。
    不完全チェーングループのメンバーも除去し、
    除去後に再検証を繰り返して依存関係の連鎖崩壊を防ぐ。
    """
    if not moves:
        return moves

    _blocked = blocked_dest_locs or set()
    current_moves = list(moves)  # 作業コピー
    _max_rounds = 5

    # ロット文字列ベースのインデックス構築（初期在庫から）
    _orig_lot_strings: Dict[Tuple[str, str], Set[str]] = {}
    if original_inv is not None and "ロット" in original_inv.columns and "ロケーション" in original_inv.columns and "商品ID" in original_inv.columns:
        for _loc_o, _sku_o, _lot_o in zip(
            original_inv["ロケーション"].astype(str),
            original_inv["商品ID"].astype(str),
            original_inv["ロット"].astype(str),
        ):
            _orig_lot_strings.setdefault((_loc_o, _sku_o), set()).add(_lot_o)

    for _round in range(_max_rounds):
        # シミュレーション状態を初期化
        sim_skus: Dict[str, Set[str]] = {k: set(v) for k, v in original_skus_by_loc.items()}
        sim_lots: Dict[Tuple[str, str], Set[int]] = {
            k: set(v) for k, v in original_lots_by_loc_sku.items()
        }
        sim_lot_strings: Dict[Tuple[str, str], Set[str]] = {
            k: set(v) for k, v in _orig_lot_strings.items()
        }
        # 残ケース数を正確に追跡（部分移動対応）
        sim_qty: Dict[Tuple[str, str], float] = {}
        if original_qty_by_loc_sku:
            sim_qty = dict(original_qty_by_loc_sku)
        else:
            # フォールバック: 各(loc, sku)に大きな値を設定
            for loc, skus in sim_skus.items():
                for sku in skus:
                    sim_qty[(loc, sku)] = 9999.0

        accepted_indices: List[int] = []
        rejected_chains: Set[str] = set()

        # 入力順（依存解決済み）で処理: スワップはメンバーが揃った時点でアトミック処理
        _pv_swap_members: Dict[str, List[Tuple[int, Move]]] = {}
        _pv_swap_ids: Set[str] = set()
        for i, m in enumerate(current_moves):
            cg = getattr(m, 'chain_group_id', None) or ""
            if cg.startswith("swap_") or cg.startswith("fifo_direct_") or cg.startswith("zone_swap_"):
                _pv_swap_members.setdefault(cg, []).append((i, m))
                _pv_swap_ids.add(cg)

        _pv_processed_swaps: Set[str] = set()
        for i, m in enumerate(current_moves):
            cg = getattr(m, 'chain_group_id', None) or ""

            if cg in _pv_swap_ids:
                # スワップグループ: 既に処理済みならスキップ
                if cg in _pv_processed_swaps:
                    continue
                _pv_processed_swaps.add(cg)

                if cg in rejected_chains:
                    continue

                _pv_sg_members = _pv_swap_members.get(cg, [])
                if len(_pv_sg_members) != 2:
                    rejected_chains.add(cg)
                    continue
                _pv_i1, _pv_m1 = _pv_sg_members[0]
                _pv_i2, _pv_m2 = _pv_sg_members[1]
                _pv_ok = True

                # blocked チェック
                for _, _pv_m in _pv_sg_members:
                    _pv_to = str(_pv_m.to_loc).zfill(8)
                    if _blocked and _pv_to in _blocked:
                        _pv_ok = False; break
                if not _pv_ok:
                    rejected_chains.add(cg)
                    continue

                # アトミックSKU混在チェック
                _pv_sku1, _pv_sku2 = str(_pv_m1.sku_id), str(_pv_m2.sku_id)
                _pv_to1, _pv_from1 = str(_pv_m1.to_loc).zfill(8), str(_pv_m1.from_loc).zfill(8)
                _pv_to2, _pv_from2 = str(_pv_m2.to_loc).zfill(8), str(_pv_m2.from_loc).zfill(8)

                _pv_skus_at_to1 = set(sim_skus.get(_pv_to1, set()))
                _pv_skus_at_to2 = set(sim_skus.get(_pv_to2, set()))
                _pv_qty_to1 = sim_qty.get((_pv_to1, _pv_sku2), 0.0) - float(_pv_m2.qty)
                _pv_after_to1 = set(_pv_skus_at_to1)
                if _pv_qty_to1 <= 0:
                    _pv_after_to1.discard(_pv_sku2)
                _pv_after_to1.add(_pv_sku1)

                _pv_qty_to2 = sim_qty.get((_pv_to2, _pv_sku1), 0.0) - float(_pv_m1.qty)
                _pv_after_to2 = set(_pv_skus_at_to2)
                if _pv_qty_to2 <= 0:
                    _pv_after_to2.discard(_pv_sku1)
                _pv_after_to2.add(_pv_sku2)

                if len(_pv_after_to1) > 1 or len(_pv_after_to2) > 1:
                    rejected_chains.add(cg)
                    continue

                # ロット混在チェック
                for _, _pv_m in _pv_sg_members:
                    _pv_sku_l = str(_pv_m.sku_id)
                    _pv_to_l = str(_pv_m.to_loc).zfill(8)
                    _pv_lot_key = _parse_lot_date_key(_pv_m.lot) if _pv_m.lot else UNKNOWN_LOT_KEY
                    if int(_pv_lot_key) == UNKNOWN_LOT_KEY:
                        continue
                    _pv_lsk = (_pv_to_l, _pv_sku_l)
                    _pv_ex_lots = sim_lots.get(_pv_lsk, set())
                    if _pv_ex_lots:
                        _pv_known = _pv_ex_lots - {UNKNOWN_LOT_KEY}
                        if _pv_known and int(_pv_lot_key) not in _pv_known:
                            _pv_ok = False; break
                if not _pv_ok:
                    rejected_chains.add(cg)
                    continue

                # 受理 & シミュレーション状態更新
                for _pv_i, _pv_m in _pv_sg_members:
                    accepted_indices.append(_pv_i)
                    _pv_sku_a = str(_pv_m.sku_id)
                    _pv_to_a = str(_pv_m.to_loc).zfill(8)
                    _pv_from_a = str(_pv_m.from_loc).zfill(8)
                    _pv_lot_a = _parse_lot_date_key(_pv_m.lot) if _pv_m.lot else UNKNOWN_LOT_KEY
                    sim_skus.setdefault(_pv_to_a, set()).add(_pv_sku_a)
                    sim_lots.setdefault((_pv_to_a, _pv_sku_a), set()).add(int(_pv_lot_a))
                    _pv_ls = str(_pv_m.lot) if _pv_m.lot else ""
                    if _pv_ls:
                        sim_lot_strings.setdefault((_pv_to_a, _pv_sku_a), set()).add(_pv_ls)
                    sim_qty[(_pv_to_a, _pv_sku_a)] = sim_qty.get((_pv_to_a, _pv_sku_a), 0.0) + float(_pv_m.qty)
                    sim_qty[(_pv_from_a, _pv_sku_a)] = sim_qty.get((_pv_from_a, _pv_sku_a), 0.0) - float(_pv_m.qty)
                    if sim_qty.get((_pv_from_a, _pv_sku_a), 0.0) <= 0:
                        _pv_fs = sim_skus.get(_pv_from_a)
                        if _pv_fs:
                            _pv_fs.discard(_pv_sku_a)
                            if not _pv_fs:
                                del sim_skus[_pv_from_a]
                        _pv_fk = (_pv_from_a, _pv_sku_a)
                        if _pv_fk in sim_lots:
                            del sim_lots[_pv_fk]
                        if _pv_fk in sim_lot_strings:
                            del sim_lot_strings[_pv_fk]
                # アトミックSKU更新
                sim_skus[_pv_to1] = _pv_after_to1
                sim_skus[_pv_to2] = _pv_after_to2
                continue

            # 非スワップmoveの処理
            to_loc = str(m.to_loc).zfill(8)
            from_loc = str(m.from_loc).zfill(8)
            sku = str(m.sku_id)
            lot_key = _parse_lot_date_key(m.lot) if m.lot else UNKNOWN_LOT_KEY

            # 既に却下済みチェーンに属する移動はスキップ
            if cg and cg in rejected_chains:
                continue

            # 非対象品質ロケーションへの移動を拒否
            if _blocked and to_loc in _blocked:
                if cg:
                    rejected_chains.add(cg)
                continue

            # SKU混在チェック
            existing_skus = sim_skus.get(to_loc, set())
            if existing_skus and not existing_skus <= {sku}:
                if cg:
                    rejected_chains.add(cg)
                continue

            # ロット混在チェック（UNKNOWN_LOT_KEY はスキップ）
            if int(lot_key) != UNKNOWN_LOT_KEY:
                lsk = (to_loc, sku)
                existing_lots = sim_lots.get(lsk, set())
                if existing_lots:
                    known_lots = existing_lots - {UNKNOWN_LOT_KEY}
                    if known_lots and int(lot_key) not in known_lots:
                        if cg:
                            rejected_chains.add(cg)
                        continue
                # ロット文字列ベース混在チェック（同日異ロットを区別）
                _lot_str_pv = str(m.lot) if m.lot else ""
                _existing_pv_strs = sim_lot_strings.get(lsk, set())
                if _lot_str_pv and _existing_pv_strs and _lot_str_pv not in _existing_pv_strs:
                    if cg:
                        rejected_chains.add(cg)
                    continue

            accepted_indices.append(i)

            # シミュレーション状態更新
            sim_skus.setdefault(to_loc, set()).add(sku)
            lsk_to = (to_loc, sku)
            sim_lots.setdefault(lsk_to, set()).add(int(lot_key))
            # ロット文字列更新
            _pv_lot_str_upd = str(m.lot) if m.lot else ""
            if _pv_lot_str_upd:
                sim_lot_strings.setdefault(lsk_to, set()).add(_pv_lot_str_upd)
            sim_qty[(to_loc, sku)] = sim_qty.get((to_loc, sku), 0.0) + float(m.qty)

            # 移動元: 残ケース追跡で正確にSKU除去
            sim_qty[(from_loc, sku)] = sim_qty.get((from_loc, sku), 0.0) - float(m.qty)
            if sim_qty.get((from_loc, sku), 0.0) <= 0:
                from_skus = sim_skus.get(from_loc)
                if from_skus:
                    from_skus.discard(sku)
                    if not from_skus:
                        del sim_skus[from_loc]
                # qty <= 0 = ロケーションから完全に出た → 全ロットを除去
                from_lsk = (from_loc, sku)
                if from_lsk in sim_lots:
                    del sim_lots[from_lsk]
                if from_lsk in sim_lot_strings:
                    del sim_lot_strings[from_lsk]

        # 不完全チェーングループを特定
        chain_counts_input: Dict[str, int] = {}
        for m in current_moves:
            cg = getattr(m, 'chain_group_id', None)
            if cg:
                chain_counts_input[cg] = chain_counts_input.get(cg, 0) + 1

        chain_counts_accepted: Dict[str, int] = {}
        for i in accepted_indices:
            cg = getattr(current_moves[i], 'chain_group_id', None)
            if cg:
                chain_counts_accepted[cg] = chain_counts_accepted.get(cg, 0) + 1

        incomplete = set()
        for cg, cnt in chain_counts_input.items():
            if chain_counts_accepted.get(cg, 0) < cnt:
                incomplete.add(cg)

        # 不完全チェーンのメンバーを除去
        if incomplete:
            accepted_indices = [
                i for i in accepted_indices
                if getattr(current_moves[i], 'chain_group_id', None) not in incomplete
            ]

        new_moves = [current_moves[i] for i in accepted_indices]
        removed = len(current_moves) - len(new_moves)

        logger.debug(f"[optimizer] Post-resolution round {_round+1}: "
              f"removed {removed} moves ({len(incomplete)} incomplete chains), "
              f"remaining {len(new_moves)}")

        if removed == 0:
            # 安定状態 — これ以上の除去なし
            break

        current_moves = new_moves  # 次のラウンドで再検証

    return current_moves


def _get_sku_pack_band(sku_id: str, pack_map: Optional[pd.Series], cfg: "OptimizerConfig") -> str:
    """SKUの入数からpack帯("small"/"medium"/"large")を返す。"""
    if pack_map is None:
        return "medium"
    try:
        pack_val = pack_map.get(sku_id)
        if pack_val is None or pd.isna(pack_val):
            return "medium"
        pack = float(pack_val)
        low_max = int(getattr(cfg, "pack_low_max", 12))
        high_min = int(getattr(cfg, "pack_high_min", 50))
        if pack <= low_max:
            return "small"
        elif pack >= high_min:
            return "large"
        else:
            return "medium"
    except Exception:
        return "medium"


# Lv1/Lv2 only. multi_sku_target_levels gates which levels the cap applies to.
def _get_level_vol_cap(level: int, cfg: "OptimizerConfig") -> Optional[float]:
    """Return the per-level volume cap for this level, or None if no cap."""
    if not getattr(cfg, "enforce_level_vol_cap", True):
        return None
    target_levels = getattr(cfg, "multi_sku_target_levels", (1, 2))
    if level not in target_levels:
        return None
    if level == 1:
        return getattr(cfg, "multi_sku_level1_vol_cap", 0.9)
    if level == 2:
        return getattr(cfg, "multi_sku_level2_vol_cap", 0.3)
    return None


def _check_multi_sku_coexistence(
    to_loc: str,
    sku_id: str,
    qty: int,
    pack_vol: float,
    original_receivable_locs: Set[str],
    sim_skus_by_loc: Dict[str, Set[str]],
    sim_vol_by_loc: Dict[str, float],
    sku_pack_band: Dict[str, str],
    level: int,
    cfg: "OptimizerConfig",
) -> Tuple[bool, Optional[str]]:
    """元々0-2SKUだったロケへの複数SKU同居ゲート（空き + 1-2SKUロケ対象）。

    Returns (allowed, reject_reason). reject_reason is None if allowed.
    """
    # 1. 受入可能ロケでなければ従来ロジックに委ねる（フォールスルー）
    if to_loc not in original_receivable_locs:
        return (False, None)

    # 2. 対象段でなければ却下
    target_levels = tuple(getattr(cfg, "multi_sku_target_levels", (1, 2)))
    if level not in target_levels:
        return (False, "multi_sku_level_disallowed")

    existing_skus = sim_skus_by_loc.get(to_loc, set())
    other_skus = existing_skus - {sku_id}

    # 3. 実質1SKU目（他SKUなし）→ SKU数/pack帯チェック不要、フォールスルー
    # No other SKUs means this is effectively a same-SKU addition or empty loc;
    # level_vol_cap and foreign_sku checks are handled by caller.
    if not other_skus:
        return (True, None)

    # 4. 同居後のSKU数チェック
    max_per_loc = int(getattr(cfg, "multi_sku_max_per_loc", 3))
    after_count = len(existing_skus | {sku_id})
    if after_count > max_per_loc:
        return (False, "multi_sku_count_exceeded")

    # 5. pack帯一致チェック
    if getattr(cfg, "multi_sku_pack_band_match", True):
        my_band = sku_pack_band.get(sku_id, "medium")
        for other_sku in other_skus:
            other_band = sku_pack_band.get(other_sku, "medium")
            if other_band != my_band:
                return (False, "multi_sku_pack_band_mismatch")

    # 段別容積上限チェックは capacity ゲート側に統一（全着地に常時適用）

    return (True, None)


def enforce_constraints(
    sku_master: pd.DataFrame,
    inventory: pd.DataFrame,
    moves: List[Move],
    *,
    cfg: OptimizerConfig | None = None,
    loc_master: Optional[pd.DataFrame] = None,
    original_skus_by_loc: Optional[Dict[str, Set[str]]] = None,
    original_qty_by_loc_sku: Optional[Dict[Tuple[str, str], float]] = None,
    blocked_dest_locs: Optional[Set[str]] = None,
    blocked_source_locs: Optional[Set[str]] = None,
    original_inv_lot_levels: Optional[Dict[Tuple[str, int], list]] = None,
    original_empty_locs: Optional[Set[str]] = None,
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
    can_receive_set: Optional[Set[str]] = None  # None = ロケマスタ未提供 → 全ロケ受入可
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
    # ロケーションを8桁ゼロ埋めに正規化（移動のto_locと形式を揃える）
    inv["ロケーション"] = inv["ロケーション"].astype(str).str.zfill(8)
    loc_str = inv["ロケーション"]
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

    # Fix2: ロット混在チェック用インデックス (loc, sku) → Set[lot_key]
    # enforce_constraints でも plan_relocation と同様の混在防止を適用する
    _sim_lots_by_loc_sku: Dict[Tuple[str, str], Set[int]] = {}
    for (_loc, _sku_id), _grp in sim_inv.groupby(["ロケーション", "商品ID"]):
        _sim_lots_by_loc_sku[(str(_loc), str(_sku_id))] = set(int(x) for x in _grp["lot_key"].tolist())

    # ロット文字列ベースの混在チェック用インデックス（同日異ロットを区別するため）
    _sim_lot_strings_by_loc_sku: Dict[Tuple[str, str], Set[str]] = {}
    if "ロット" in inv.columns:
        for (_loc2, _sku2), _grp2 in inv.groupby(["ロケーション", "商品ID"]):
            _sim_lot_strings_by_loc_sku[(str(_loc2), str(_sku2))] = set(_grp2["ロット"].astype(str).tolist())

    # 別SKU混在チェック用インデックス loc → Set[sku_id]
    # original_skus_by_loc が提供された場合、パス前の真の倉庫状態を使用。
    # enforce はこのインデックスを逐次更新しながら移動をシミュレーションする。
    # パスが inv をインプレース変更するため、sim_inv からの構築では
    # "既に移動済み" の状態になり、本来そこにいるSKUが見えなくなってしまう。
    if original_skus_by_loc is not None:
        _sim_skus_by_loc: Dict[str, Set[str]] = {
            k: set(v) for k, v in original_skus_by_loc.items()
        }
        logger.debug(f"[enforce] _sim_skus_by_loc from original_skus_by_loc: {len(_sim_skus_by_loc)} locs, sample={list(_sim_skus_by_loc.items())[:3]}")
    else:
        _sim_skus_by_loc: Dict[str, Set[str]] = {}
        for _loc_v, _sku_v in zip(sim_inv["ロケーション"].astype(str), sim_inv["商品ID"].astype(str)):
            _sim_skus_by_loc.setdefault(_loc_v, set()).add(_sku_v)
        logger.debug(f"[enforce] _sim_skus_by_loc from sim_inv: {len(_sim_skus_by_loc)} locs")

    # SKU除去判定用の残ケース数カウンタ (loc, sku) → remaining cases
    # original_qty_by_loc_sku が提供された場合はパス前の正確な数量を使用。
    # _sim_lots_by_loc_sku は変異後invから構築されるため、パスで移動済みのSKUの
    # 残ロット=∅ となり、_sim_skus_by_loc から不正にSKUが除去されるバグがあった。
    if original_qty_by_loc_sku is not None:
        _sim_remaining_qty: Dict[Tuple[str, str], float] = dict(original_qty_by_loc_sku)
    else:
        # fallback: sim_inv から構築
        _sim_remaining_qty: Dict[Tuple[str, str], float] = {}
        _qty_col = "qty_cases_move" if "qty_cases_move" in sim_inv.columns else None
        if _qty_col is None and "ケース" in sim_inv.columns:
            _qty_col = "ケース"
        for _ridx in range(len(sim_inv)):
            _rloc = str(sim_inv.iloc[_ridx]["ロケーション"])
            _rsku = str(sim_inv.iloc[_ridx]["商品ID"])
            _rqty = float(sim_inv.iloc[_ridx][_qty_col]) if _qty_col else 1.0
            _rkey = (_rloc, _rsku)
            _sim_remaining_qty[_rkey] = _sim_remaining_qty.get(_rkey, 0.0) + _rqty

    # FIFO高速チェック用インデックス (sku, col) → [(lot_key, lv)]
    # pd.concatによるO(n²)問題を回避し、O(1)ルックアップに置き換える
    _sim_inv_by_sku_col: Dict[Tuple[str, int], List[Tuple[int, int]]] = {}
    for (_sku_g, _col_g), _grp_g in sim_inv.groupby(["商品ID", "col"]):
        _sim_inv_by_sku_col[(str(_sku_g), int(_col_g))] = list(zip(
            _grp_g["lot_key"].astype(int).tolist(),
            _grp_g["lv"].astype(int).tolist(),
        ))

    # 複数SKU同居用: ロケ別シミュ容積マップ（O(1)更新用）
    # shelf_usage をベースに初期化（enforce内でも同じ値から開始）
    _sim_vol_by_loc: Dict[str, float] = dict(shelf_usage)

    # 複数SKU同居用: SKU → pack帯マップ（事前構築）
    _sku_pack_band_map: Dict[str, str] = {}
    if pack_map is not None:
        for _pb_sku in pack_map.index:
            _sku_pack_band_map[str(_pb_sku)] = _get_sku_pack_band(str(_pb_sku), pack_map, cfg)

    # 元々空きだったロケ集合（外部から渡された場合はそのまま使用）
    _original_receivable_locs_enforce: Set[str] = original_empty_locs if original_empty_locs is not None else set()

    # 入力movesをチェーン内execution_order順にソートして退避が先に処理されるようにする
    # これにより部分退避後のforeign_skuチェックが正しく動作する
    def _enforce_sort_key(m):
        cg = getattr(m, 'chain_group_id', None) or ""
        eo = getattr(m, 'execution_order', None) or 0
        try:
            eo = int(eo)
        except (ValueError, TypeError):
            eo = 0
        return (cg, eo)
    moves = sorted(moves, key=_enforce_sort_key)

    processed = 0
    step = max(1, len(moves) // 20) if moves else 1
    _rejected_chain_groups: Set[str] = set()  # TODO: 将来のチェーン伝播用に予約
    _deferred_foreign_sku: List = []  # foreign_sku で却下されたmoveを再評価用に保持
    _deferred_swap_groups: Dict[str, List[Move]] = {}  # foreign_skuで保留されたスワップ
    # 移動先衝突防止: to_loc → chain_group_id のマップ
    # 異なるチェーンが同一ロケーションを移動先に指定するのを防ぐ
    _dest_claimed_by: Dict[str, str] = {}

    # ---- スワップchain_groupのメンバー収集（入力順で処理するための前準備）----
    # swap_ / fifo_direct_ / zone_swap_ プレフィックスのchain_groupをインデックス化
    _swap_group_members: Dict[str, List[Move]] = {}
    _swap_group_ids: Set[str] = set()
    for m in moves:
        cg = getattr(m, 'chain_group_id', None) or ""
        if cg.startswith("swap_") or cg.startswith("fifo_direct_") or cg.startswith("zone_swap_"):
            _swap_group_members.setdefault(cg, []).append(m)
            _swap_group_ids.add(cg)

    # スワップのアトミック処理関数（メインループと再評価ループで共用）
    _swap_accepted = 0
    _swap_rejected = 0

    def _try_accept_swap_group(_sg_id: str, _sg_moves: List[Move]) -> bool:
        """スワップグループを検証し、受理できればaccepted/状態を更新してTrueを返す。"""
        nonlocal _swap_accepted, _swap_rejected
        if len(_sg_moves) != 2:
            _swap_rejected += len(_sg_moves)
            return False
        _sg_m1, _sg_m2 = _sg_moves[0], _sg_moves[1]
        _sg_ok = True

        # 各moveの基本チェック（容量・blocked・can_receive）
        for _sg_m in [_sg_m1, _sg_m2]:
            _sg_sku = str(_sg_m.sku_id)
            _sg_to = str(_sg_m.to_loc).zfill(8)
            _sg_from = str(_sg_m.from_loc).zfill(8)
            _sg_add_each = float(sku_vol_map.get(_sg_sku, 0.0) or 0.0)

            if blocked_source_locs and _sg_from in blocked_source_locs:
                _sg_ok = False; break
            if cfg.exclude_oversize and _sg_add_each > max_cap:
                _sg_ok = False; break
            if can_receive_set is not None and _sg_to not in can_receive_set:
                _sg_ok = False; break
            if blocked_dest_locs and _sg_to in blocked_dest_locs:
                _sg_ok = False; break

        if not _sg_ok:
            _swap_rejected += 2
            return False

        # スワップ後の状態をシミュレーション（アトミック）
        _sg_sku1 = str(_sg_m1.sku_id)
        _sg_sku2 = str(_sg_m2.sku_id)
        _sg_to1 = str(_sg_m1.to_loc).zfill(8)
        _sg_from1 = str(_sg_m1.from_loc).zfill(8)
        _sg_to2 = str(_sg_m2.to_loc).zfill(8)
        _sg_from2 = str(_sg_m2.from_loc).zfill(8)

        # foreign SKU チェック: 交換後の各ロケのSKU集合を計算
        # スワップの場合: m1(sku1)がfrom1→to1, m2(sku2)がfrom2→to2
        # ゾーンスワップでは from1==to2, from2==to1 (互いのロケを交換)
        #
        # 問題: _sim_remaining_qty は品質フィルタ前の全qty。同一SKUが複数品質で
        # 同一ロケに存在する場合、スワップで良品分だけ移動しても残存qty>0となり
        # SKUが退去しないと判定される。
        #
        # 解決: スワップでは「交換相手のロケに行く」のが前提なので、
        # 交換元SKUのdiscardは残存qtyに関わらず行う（スワップはロケ全体の交換）。
        # ただし、第三のSKU（交換対象でないSKU）がいる場合は却下。
        _sg_swap_skus = {_sg_sku1, _sg_sku2}
        _sg_skus_at_to1 = set(_sim_skus_by_loc.get(_sg_to1, set()))
        _sg_skus_at_to2 = set(_sim_skus_by_loc.get(_sg_to2, set()))

        # 第三のSKU（スワップ対象でないSKU）がいたら却下
        _sg_third_at_to1 = _sg_skus_at_to1 - _sg_swap_skus
        _sg_third_at_to2 = _sg_skus_at_to2 - _sg_swap_skus
        if _sg_third_at_to1 or _sg_third_at_to2:
            _swap_rejected += 2
            _dbg_reject("foreign_sku", {"swap_group": _sg_id}, note=f"third-party SKU at to1={_sg_third_at_to1} to2={_sg_third_at_to2}")
            if _swap_rejected <= 10:  # 最初の10件だけログ
                logger.warning(f"[enforce] swap rejected: third-party SKU to1={_sg_third_at_to1} to2={_sg_third_at_to2} sku1={_sg_sku1} sku2={_sg_sku2}")
            return False

        # スワップ後のSKU集合: 交換元SKUを無条件でdiscard、交換先SKUを追加
        _sg_skus_after_to1 = set(_sg_skus_at_to1)
        _sg_skus_after_to1.discard(_sg_sku2)  # m2のSKUがto1(=m2のfrom)から出る
        _sg_skus_after_to1.add(_sg_sku1)       # m1のSKUがto1に入る

        _sg_skus_after_to2 = set(_sg_skus_at_to2)
        _sg_skus_after_to2.discard(_sg_sku1)  # m1のSKUがto2(=m1のfrom)から出る
        _sg_skus_after_to2.add(_sg_sku2)       # m2のSKUがto2に入る

        # スワップ後に各ロケのSKU数が1以下か確認
        if len(_sg_skus_after_to1) > 1 or len(_sg_skus_after_to2) > 1:
            _swap_rejected += 2
            _dbg_reject("foreign_sku", {"swap_group": _sg_id}, note=f"after swap: to1={_sg_skus_after_to1}, to2={_sg_skus_after_to2}")
            if _swap_rejected <= 10:
                logger.warning(f"[enforce] swap rejected: after-swap to1={_sg_skus_after_to1} to2={_sg_skus_after_to2}")
            return False

        # 容量チェック用のqty計算（従来通り_sim_remaining_qtyベース）
        _sg_qty_at_to1_from = _sim_remaining_qty.get((_sg_to1, _sg_sku2), 0.0) - float(_sg_m2.qty)
        _sg_qty_at_to2_from = _sim_remaining_qty.get((_sg_to2, _sg_sku1), 0.0) - float(_sg_m1.qty)

        # 容量チェック（交換後）
        _sg_cap_ok = True
        for _sg_m in [_sg_m1, _sg_m2]:
            _sg_sku_c = str(_sg_m.sku_id)
            _sg_to_c = str(_sg_m.to_loc).zfill(8)
            _sg_add_each_c = float(sku_vol_map.get(_sg_sku_c, 0.0) or 0.0)
            _sg_add_vol_c = _sg_add_each_c * float(_sg_m.qty)
            _sg_sub_sku_c = str([_sg_m2, _sg_m1][[_sg_m1, _sg_m2].index(_sg_m)].sku_id)
            _sg_sub_vol_c = float(sku_vol_map.get(_sg_sub_sku_c, 0.0) or 0.0) * float([_sg_m2, _sg_m1][[_sg_m1, _sg_m2].index(_sg_m)].qty)
            _sg_used_c = float(shelf_usage.get(_sg_to_c, 0.0))
            _sg_limit_c = cap_by_loc.get(_sg_to_c, base_cap) if cap_by_loc else base_cap
            _sg_net_vol_c = _sg_used_c - _sg_sub_vol_c + _sg_add_vol_c
            if cfg.hard_cap and (_sg_net_vol_c > _sg_limit_c):
                _sg_cap_ok = False; break
            _sg_tlv_c = _parse_loc8(_sg_to_c)[0]
            _sg_level_cap_c = _get_level_vol_cap(_sg_tlv_c, cfg)
            if _sg_level_cap_c is not None and _sg_net_vol_c > _sg_level_cap_c:
                _sg_cap_ok = False; break
        if not _sg_cap_ok:
            _swap_rejected += 2
            return False

        # ロット混在チェック
        # 同一 SKU スワップ（FIFO 是正スワップ）の場合はスキップ。
        # スワップはロケーション全体の交換であり、移動先の既存ロットは
        # 相手 move によって同時に退出するため、実際の混在は発生しない。
        _sg_lot_ok = True
        if _sg_sku1 != _sg_sku2:
            for _sg_m in [_sg_m1, _sg_m2]:
                _sg_sku_l = str(_sg_m.sku_id)
                _sg_to_l = str(_sg_m.to_loc).zfill(8)
                _sg_lot_key = _parse_lot_date_key(_sg_m.lot)
                if _sg_lot_key == UNKNOWN_LOT_KEY:
                    continue
                _sg_lsk = (_sg_to_l, _sg_sku_l)
                _sg_existing_lots = _sim_lots_by_loc_sku.get(_sg_lsk, set())
                if _sg_existing_lots:
                    _sg_known = _sg_existing_lots - {UNKNOWN_LOT_KEY}
                    if _sg_known and int(_sg_lot_key) not in _sg_known:
                        _sg_lot_ok = False; break
        if not _sg_lot_ok:
            _swap_rejected += 2
            return False

        # FIFOチェック: スワップの各移動がFIFO違反を作らないか
        if cfg.hard_fifo:
            _sg_fifo_ok = True
            for _sg_m in [_sg_m1, _sg_m2]:
                _sg_sku_f = str(_sg_m.sku_id)
                _sg_to_f = str(_sg_m.to_loc).zfill(8)
                _sg_lot_key_f = _parse_lot_date_key(_sg_m.lot)
                if _sg_lot_key_f == UNKNOWN_LOT_KEY:
                    continue
                _sg_tlv_f, _sg_tcol_f, _ = _parse_loc8(_sg_to_f)
                if _violates_lot_level_rule_fast(_sg_sku_f, _sg_lot_key_f, _sg_tlv_f, _sg_tcol_f, _sim_inv_by_sku_col):
                    _sg_fifo_ok = False
                    break
            if not _sg_fifo_ok:
                _swap_rejected += 2
                _dbg_reject("fifo", {"swap_group": _sg_id}, note="swap FIFO violation")
                return False

        # 移動先衝突チェック
        # スワップでは to_loc = 相手の from_loc のため、スワップ自身の from_loc と
        # 一致する to_loc は「自分が空けるロケ」であり本来の衝突ではない。
        # 例: swap_fifo で m1:A→B, m2:B→A の場合、to_loc=B は m2.from_loc=B と同一。
        # この場合、他チェーンが B を to_loc として claimed していても衝突しない。
        _sg_from_locs = {str(_sg_m1.from_loc).zfill(8), str(_sg_m2.from_loc).zfill(8)}
        _sg_dest_conflict = False
        for _sg_m in [_sg_m1, _sg_m2]:
            _sg_to_dc = str(_sg_m.to_loc).zfill(8)
            # スワップ自身の from_loc と一致する to_loc は衝突とみなさない
            if _sg_to_dc in _sg_from_locs:
                continue
            _claimed_cg = _dest_claimed_by.get(_sg_to_dc)
            if _claimed_cg is not None and _claimed_cg != _sg_id:
                _sg_dest_conflict = True
                _dbg_reject("dest_conflict", {"swap_group": _sg_id, "to_loc": _sg_to_dc}, note=f"already claimed by {_claimed_cg}")
                break
        if _sg_dest_conflict:
            _swap_rejected += 2
            return False

        # 全チェック通過 → 2手とも受理 & シミュレーション状態更新
        for _sg_m in [_sg_m1, _sg_m2]:
            _sg_sku_a = str(_sg_m.sku_id)
            _sg_lot_key_a = _parse_lot_date_key(_sg_m.lot)
            _sg_to_a = str(_sg_m.to_loc).zfill(8)
            _sg_from_a = str(_sg_m.from_loc).zfill(8)
            _sg_tlv, _sg_tcol, _sg_tdep = _parse_loc8(_sg_to_a)
            _sg_flv, _sg_fcol, _sg_fdep = _parse_loc8(_sg_from_a)
            _sg_add_each_a = float(sku_vol_map.get(_sg_sku_a, 0.0) or 0.0)
            _sg_add_vol_a = _sg_add_each_a * float(_sg_m.qty)
            _sg_lot_date = _lot_key_to_datestr8(_sg_lot_key_a)

            accepted.append(Move(
                sku_id=_sg_sku_a, lot=_sg_m.lot, qty=_sg_m.qty,
                from_loc=_sg_from_a, to_loc=_sg_to_a,
                lot_date=_sg_lot_date, reason=_sg_m.reason or "",
                chain_group_id=_sg_m.chain_group_id,
                execution_order=_sg_m.execution_order,
                distance=getattr(_sg_m, 'distance', None),
            ))
            shelf_usage[_sg_to_a] = shelf_usage.get(_sg_to_a, 0.0) + _sg_add_vol_a
            shelf_usage[_sg_from_a] = max(0.0, shelf_usage.get(_sg_from_a, 0.0) - _sg_add_vol_a)
            _sim_vol_by_loc[_sg_to_a] = _sim_vol_by_loc.get(_sg_to_a, 0.0) + _sg_add_vol_a
            _sim_vol_by_loc[_sg_from_a] = max(0.0, _sim_vol_by_loc.get(_sg_from_a, 0.0) - _sg_add_vol_a)
            _sg_to_key_fc = (_sg_sku_a, int(_sg_tcol))
            _sim_inv_by_sku_col.setdefault(_sg_to_key_fc, []).append((int(_sg_lot_key_a), int(_sg_tlv)))
            _sg_from_key_fc = (_sg_sku_a, int(_sg_fcol))
            if _sg_from_key_fc in _sim_inv_by_sku_col:
                _entries = _sim_inv_by_sku_col[_sg_from_key_fc]
                _target = (int(_sg_lot_key_a), int(_sg_flv))
                for _ei, _ev in enumerate(_entries):
                    if _ev == _target:
                        _entries.pop(_ei)
                        break
            _sg_to_key = (_sg_to_a, _sg_sku_a)
            _sg_from_key = (_sg_from_a, _sg_sku_a)
            _sim_lots_by_loc_sku.setdefault(_sg_to_key, set()).add(int(_sg_lot_key_a))
            _sim_remaining_qty[(_sg_to_a, _sg_sku_a)] = _sim_remaining_qty.get((_sg_to_a, _sg_sku_a), 0.0) + float(_sg_m.qty)
            _sim_remaining_qty[(_sg_from_a, _sg_sku_a)] = _sim_remaining_qty.get((_sg_from_a, _sg_sku_a), 0.0) - float(_sg_m.qty)
            if _sg_from_key in _sim_lots_by_loc_sku:
                if _sim_remaining_qty.get((_sg_from_key[0], _sg_from_key[1]), 0.0) <= 0:
                    _sim_lots_by_loc_sku[_sg_from_key].discard(int(_sg_lot_key_a))
            _sg_lot_str = str(_sg_m.lot) if _sg_m.lot else ""
            if _sg_lot_str:
                _sim_lot_strings_by_loc_sku.setdefault(_sg_to_key, set()).add(_sg_lot_str)
                if _sg_from_key in _sim_lot_strings_by_loc_sku:
                    if _sim_remaining_qty.get((_sg_from_key[0], _sg_from_key[1]), 0.0) <= 0:
                        _sim_lot_strings_by_loc_sku[_sg_from_key].discard(_sg_lot_str)

        # _sim_skus_by_loc をアトミックに更新（2手分まとめて）
        _sim_skus_by_loc[_sg_to1] = _sg_skus_after_to1
        _sim_skus_by_loc[_sg_to2] = _sg_skus_after_to2
        for _sg_m, _sg_partner in [(_sg_m1, _sg_m2), (_sg_m2, _sg_m1)]:
            _sg_from_upd = str(_sg_m.from_loc).zfill(8)
            _sg_sku_upd = str(_sg_m.sku_id)
            if _sim_remaining_qty.get((_sg_from_upd, _sg_sku_upd), 0.0) <= 0:
                _sg_from_set = _sim_skus_by_loc.get(_sg_from_upd)
                if _sg_from_set:
                    _sg_from_set.discard(_sg_sku_upd)

        _swap_accepted += 2
        _dest_claimed_by[_sg_to1] = _sg_id
        _dest_claimed_by[_sg_to2] = _sg_id
        return True

    # ---- 統合メインループ: 入力順（トポロジカルソート済み）で処理 ----
    # 非スワップmoveは逐次処理、スワップはメンバーが揃った時点でアトミック処理。
    # これにより退避チェーンが先にSKUを移動し、その後のスワップが通るようになる。
    _processed_swap_groups: Set[str] = set()
    for m in moves:
        cg = getattr(m, 'chain_group_id', None) or ""
        if cg in _swap_group_ids:
            # スワップグループのメンバー: 既に処理済みならスキップ
            if cg in _processed_swap_groups:
                continue
            _processed_swap_groups.add(cg)
            # メンバーが揃っているか確認して処理
            _sg_moves = _swap_group_members.get(cg, [])
            if not _try_accept_swap_group(cg, _sg_moves):
                # foreign_sku が原因の場合のみ再評価候補に追加
                if len(_sg_moves) == 2:
                    _sg_m1, _sg_m2 = _sg_moves[0], _sg_moves[1]
                    _sg_to1 = str(_sg_m1.to_loc).zfill(8)
                    _sg_to2 = str(_sg_m2.to_loc).zfill(8)
                    _sg_sku1 = str(_sg_m1.sku_id)
                    _sg_sku2 = str(_sg_m2.sku_id)
                    _sg_skus_at_to1 = _sim_skus_by_loc.get(_sg_to1, set())
                    _sg_skus_at_to2 = _sim_skus_by_loc.get(_sg_to2, set())
                    _has_foreign1 = _sg_skus_at_to1 and not _sg_skus_at_to1 <= {_sg_sku1, _sg_sku2}
                    _has_foreign2 = _sg_skus_at_to2 and not _sg_skus_at_to2 <= {_sg_sku1, _sg_sku2}
                    if _has_foreign1 or _has_foreign2:
                        _deferred_swap_groups[cg] = _sg_moves
            continue

        # ---- 非スワップmoveの処理 ----
        sku = str(m.sku_id)
        lot_key = _parse_lot_date_key(m.lot)
        _is_unknown_lot = (lot_key == UNKNOWN_LOT_KEY)
        to_loc = str(m.to_loc).zfill(8)
        from_loc = str(m.from_loc).zfill(8)
        tlv, tcol, tdep = _parse_loc8(to_loc)
        flv, fcol, fdep = _parse_loc8(from_loc)
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

        # 0) blocked source gate (混在ロケからの移動ブロック)
        if blocked_source_locs and from_loc in blocked_source_locs:
            _dbg_reject("blocked_source", _mctx, note="source is multi-SKU mixed location")
            continue

        # 0b) oversize (compare against global max capacity if using master)
        if cfg.exclude_oversize and add_each > max_cap:
            _dbg_reject("oversize", _mctx, note=f"each={add_each:.3f}m3 > max_cap={max_cap:.3f}")
            continue

        # 1) can_receive gate (when master provided)
        if can_receive_set is not None and to_loc not in can_receive_set:
            _dbg_reject("forbidden", _mctx, note="destination cannot receive")
            continue

        # 1b) blocked destination gate (non-良品 quality items present)
        if blocked_dest_locs and to_loc in blocked_dest_locs:
            _dbg_reject("forbidden", _mctx, note="destination has non-target quality items")
            continue

        # 2c) 移動先衝突ゲート: 他チェーンが既にこのto_locを確保済みなら却下
        # 例外: 複数SKU同居が許可されるパス（p1fifo等）かつ元々空きロケは通過させる
        _this_cg = getattr(m, 'chain_group_id', None) or ""
        _dest_owner = _dest_claimed_by.get(to_loc)
        if _dest_owner is not None and _dest_owner != _this_cg:
            _dest_conflict_bypass = False
            if getattr(cfg, "allow_empty_loc_multi_sku", True) and to_loc in _original_receivable_locs_enforce:
                _dcb_prefixes = tuple(getattr(cfg, "multi_sku_allowed_chain_prefixes",
                    ("p1fifo", "swap_fifo_", "fifo_direct_", "p0rebal_", "p2consol_")))
                _dest_conflict_bypass = bool(_this_cg and any(_this_cg.startswith(p) for p in _dcb_prefixes))
            if not _dest_conflict_bypass:
                _dbg_reject("dest_conflict", _mctx, note=f"to_loc={to_loc} already claimed by chain {_dest_owner}")
                continue

        # 2) capacity gate (per-location if provided)
        used_to = float(shelf_usage.get(to_loc, 0.0))
        limit_to = cap_by_loc.get(to_loc, base_cap) if cap_by_loc else base_cap
        if cfg.hard_cap and (used_to + add_vol > limit_to):
            _dbg_reject("capacity", _mctx, note=f"used={used_to:.3f}, limit={limit_to:.3f}, add={add_vol:.3f}")
            continue
        # 2b) level volume cap gate (Lv1/Lv2 全着地に常時適用、hard_cap 非依存)
        _level_cap = _get_level_vol_cap(tlv, cfg)
        if _level_cap is not None and used_to + add_vol > _level_cap:
            _dbg_reject("level_vol_cap", _mctx, note=f"used={used_to:.3f}, level_cap={_level_cap:.3f}, add={add_vol:.3f}")
            continue

        # 2.4) 別SKU混在ゲート —— 移動先に自SKU以外が既にいたら一旦保留（後で再評価）
        _existing_skus_at_dest = _sim_skus_by_loc.get(to_loc, set())
        _foreign_sku_present = bool(_existing_skus_at_dest and not _existing_skus_at_dest <= {sku})
        if _foreign_sku_present:
            _is_allowed_pass = False
            if getattr(cfg, "allow_empty_loc_multi_sku", True):
                _this_cg_multi = getattr(m, 'chain_group_id', None) or ""
                _allowed_prefixes = tuple(getattr(cfg, "multi_sku_allowed_chain_prefixes",
                    ("p1fifo", "swap_fifo_", "fifo_direct_", "p0rebal_", "p2consol_")))
                _is_allowed_pass = bool(_this_cg_multi and any(
                    _this_cg_multi.startswith(p) for p in _allowed_prefixes
                ))
            if _is_allowed_pass:
                _coex_allowed, _coex_reason = _check_multi_sku_coexistence(
                    to_loc=to_loc,
                    sku_id=sku,
                    qty=int(m.qty),
                    pack_vol=float(sku_vol_map.get(sku, 0.0) or 0.0),
                    original_receivable_locs=_original_receivable_locs_enforce,
                    sim_skus_by_loc=_sim_skus_by_loc,
                    sim_vol_by_loc=_sim_vol_by_loc,
                    sku_pack_band=_sku_pack_band_map,
                    level=tlv,
                    cfg=cfg,
                )
                if not _coex_allowed:
                    if _coex_reason is None:
                        # フォールスルー: 従来のforeign_sku保留に委ねる
                        _deferred_foreign_sku.append(m)
                        continue
                    else:
                        logger.debug(f"[enforce] multi_sku coexistence rejected: {_coex_reason} sku={sku} to={to_loc}")
                        _dbg_reject("foreign_sku", _mctx, note=f"multi_sku_coex: {_coex_reason}")
                        continue
                # 許可: 続けてロット混在チェック等を実行
                logger.debug(f"[enforce] multi_sku coexistence allowed: sku={sku} to={to_loc} existing={_existing_skus_at_dest}")
            else:
                _deferred_foreign_sku.append(m)
                continue

        # 2.5) Fix2: ロット混在ゲート —— 同一SKU・異なるロットが同一ロケに混在するのを防ぐ
        # lot_key ベースのチェック（移動元が既知ロットの場合のみ）
        _lsk = (to_loc, sku)
        if not _is_unknown_lot:
            _existing_ec_lots = _sim_lots_by_loc_sku.get(_lsk, set())
            if _existing_ec_lots:
                _known_ec_lots = _existing_ec_lots - {UNKNOWN_LOT_KEY}
                if _known_ec_lots and int(lot_key) not in _known_ec_lots:
                    _dbg_reject("lot_mix", _mctx, note=f"lot_key={lot_key} conflicts with {_known_ec_lots}")
                    continue

            # 3) FIFO strict (same column)
            if cfg.hard_fifo:
                if _violates_lot_level_rule_fast(sku, lot_key, tlv, tcol, _sim_inv_by_sku_col):
                    _dbg_reject("fifo", _mctx, note=f"lot_key={lot_key}, target_lv={tlv}, col={tcol}")
                    continue

        # M11: ロット文字列ベース混在チェック（UNKNOWN_LOT_KEY でも常に実行）
        _lot_str_ec = str(m.lot) if m.lot else ""
        if _lot_str_ec:
            _existing_ec_lot_strs = _sim_lot_strings_by_loc_sku.get(_lsk, set())
            if _existing_ec_lot_strs and _lot_str_ec not in _existing_ec_lot_strs:
                _dbg_reject("lot_mix_str", _mctx, note=f"lot='{_lot_str_ec}' conflicts with {_existing_ec_lot_strs}")
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
            from_loc=from_loc,
            to_loc=to_loc,
            lot_date=lot_date_str,
            reason=move_reason,
            chain_group_id=getattr(m, 'chain_group_id', None),
            execution_order=getattr(m, 'execution_order', None),
            distance=getattr(m, 'distance', None),
        ))
        shelf_usage[to_loc] = shelf_usage.get(to_loc, 0.0) + add_vol
        shelf_usage[from_loc] = max(0.0, shelf_usage.get(from_loc, 0.0) - add_vol)
        # _sim_vol_by_loc を逐次更新（複数SKU同居の容積チェック用）
        _sim_vol_by_loc[to_loc] = _sim_vol_by_loc.get(to_loc, 0.0) + add_vol
        _sim_vol_by_loc[from_loc] = max(0.0, _sim_vol_by_loc.get(from_loc, 0.0) - add_vol)
        # 移動先を登録
        if _this_cg:
            _dest_claimed_by[to_loc] = _this_cg

        # O(1): dict インデックスを逐次更新（pd.concat による O(n²) を廃止）
        _to_key_fc = (sku, int(tcol))
        _sim_inv_by_sku_col.setdefault(_to_key_fc, []).append((int(lot_key), int(tlv)))
        _from_key_fc = (sku, int(fcol))
        if _from_key_fc in _sim_inv_by_sku_col:
            _entries = _sim_inv_by_sku_col[_from_key_fc]
            _target = (int(lot_key), int(flv))
            for _ei, _ev in enumerate(_entries):
                if _ev == _target:
                    _entries.pop(_ei)
                    break
        # Fix2: _sim_lots_by_loc_sku を逐次更新（移動先に追加、移動元から削除）
        _to_key = (to_loc, sku)
        _from_key = (from_loc, sku)
        _sim_lots_by_loc_sku.setdefault(_to_key, set()).add(int(lot_key))
        # _sim_remaining_qty を先に更新してから残量0以下の場合のみ削除する
        _sim_skus_by_loc.setdefault(to_loc, set()).add(sku)
        _sim_remaining_qty[(to_loc, sku)] = _sim_remaining_qty.get((to_loc, sku), 0.0) + float(m.qty)
        _sim_remaining_qty[(from_loc, sku)] = _sim_remaining_qty.get((from_loc, sku), 0.0) - float(m.qty)
        if _from_key in _sim_lots_by_loc_sku:
            if _sim_remaining_qty.get((_from_key[0], _from_key[1]), 0.0) <= 0:
                _sim_lots_by_loc_sku[_from_key].discard(int(lot_key))
        # _sim_lot_strings_by_loc_sku を逐次更新（ロット文字列ベース）
        _lot_str_upd = str(m.lot) if m.lot else ""
        if _lot_str_upd:
            _sim_lot_strings_by_loc_sku.setdefault(_to_key, set()).add(_lot_str_upd)
            if _from_key in _sim_lot_strings_by_loc_sku:
                if _sim_remaining_qty.get((_from_key[0], _from_key[1]), 0.0) <= 0:
                    _sim_lot_strings_by_loc_sku[_from_key].discard(_lot_str_upd)
        _from_sku_set = _sim_skus_by_loc.get(from_loc)
        if _from_sku_set:
            if _sim_remaining_qty.get((from_loc, sku), 0.0) <= 0:
                _from_sku_set.discard(sku)
                if not _from_sku_set:
                    del _sim_skus_by_loc[from_loc]

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

    # ---- foreign_sku 再評価ループ ----
    # 無効化: リトライで承認された移動が、enforce後のチェーン整合性チェックで
    # 退避移動の却下を本体移動に伝播できず、SKU混在を引き起こすため。
    # foreign_skuで却下された移動は復活させない。
    _max_retry = 0  # リトライ無効
    for _retry_round in range(_max_retry):
        if not _deferred_foreign_sku:
            break
        _still_deferred: List = []
        _recovered = 0
        for m in _deferred_foreign_sku:
            sku = str(m.sku_id)
            lot_key = _parse_lot_date_key(m.lot)
            _is_unknown_lot = (lot_key == UNKNOWN_LOT_KEY)
            to_loc = str(m.to_loc).zfill(8)
            from_loc = str(m.from_loc).zfill(8)
            tlv, tcol, tdep = _parse_loc8(to_loc)
            flv, fcol, fdep = _parse_loc8(from_loc)
            add_each = float(sku_vol_map.get(sku, 0.0) or 0.0)
            add_vol = add_each * float(m.qty)
            _mctx = {"sku_id": sku, "lot": m.lot, "from": from_loc, "to_loc": to_loc, "to_col": tcol, "qty_cases": m.qty}

            # 再チェック: blocked source
            if blocked_source_locs and from_loc in blocked_source_locs:
                continue

            # 再チェック: oversize
            if cfg.exclude_oversize and add_each > max_cap:
                _dbg_reject("oversize", _mctx, note=f"retry: each={add_each:.3f}m3 > max_cap={max_cap:.3f}")
                continue

            # 再チェック: can_receive
            if can_receive_set is not None and to_loc not in can_receive_set:
                _dbg_reject("forbidden", _mctx, note="retry: destination cannot receive")
                continue

            # 再チェック: blocked_dest_locs
            if blocked_dest_locs and to_loc in blocked_dest_locs:
                _dbg_reject("forbidden", _mctx, note="retry: destination has non-target quality items")
                continue

            # 再チェック: 移動先衝突
            _this_cg_r = getattr(m, 'chain_group_id', None) or ""
            _dest_owner_r = _dest_claimed_by.get(to_loc)
            if _dest_owner_r is not None and _dest_owner_r != _this_cg_r:
                _dbg_reject("dest_conflict", _mctx, note=f"retry: to_loc={to_loc} already claimed by {_dest_owner_r}")
                continue

            # 再チェック: 容量（メインループと同じ順序）
            used_to = float(shelf_usage.get(to_loc, 0.0))
            limit_to = cap_by_loc.get(to_loc, base_cap) if cap_by_loc else base_cap
            if cfg.hard_cap and (used_to + add_vol > limit_to):
                _dbg_reject("capacity", _mctx, note=f"retry: used={used_to:.3f}, limit={limit_to:.3f}")
                continue
            _level_cap_r = _get_level_vol_cap(tlv, cfg)
            if _level_cap_r is not None and used_to + add_vol > _level_cap_r:
                _dbg_reject("level_vol_cap", _mctx, note=f"retry: used={used_to:.3f}, level_cap={_level_cap_r:.3f}, add={add_vol:.3f}")
                continue

            # 再チェック: foreign_sku
            _existing_skus_at_dest = _sim_skus_by_loc.get(to_loc, set())
            if _existing_skus_at_dest and not _existing_skus_at_dest <= {sku}:
                _still_deferred.append(m)
                continue

            # ロット混在チェック: lot_key ベース（既知ロットの場合のみ）
            _lsk = (to_loc, sku)
            if not _is_unknown_lot:
                _existing_ec_lots = _sim_lots_by_loc_sku.get(_lsk, set())
                if _existing_ec_lots:
                    _known_ec_lots = _existing_ec_lots - {UNKNOWN_LOT_KEY}
                    if _known_ec_lots and int(lot_key) not in _known_ec_lots:
                        _dbg_reject("lot_mix", _mctx, note=f"retry: lot_key={lot_key} conflicts")
                        continue
                # FIFOチェック
                if cfg.hard_fifo:
                    if _violates_lot_level_rule_fast(sku, lot_key, tlv, tcol, _sim_inv_by_sku_col):
                        _dbg_reject("fifo", _mctx, note=f"retry: lot_key={lot_key}")
                        continue

            # M11: ロット文字列ベース混在チェック（UNKNOWN_LOT_KEY でも常に実行）
            _lot_str_retry = str(m.lot) if m.lot else ""
            if _lot_str_retry:
                _existing_retry_lot_strs = _sim_lot_strings_by_loc_sku.get(_lsk, set())
                if _existing_retry_lot_strs and _lot_str_retry not in _existing_retry_lot_strs:
                    _dbg_reject("lot_mix_str", _mctx, note=f"retry: lot='{_lot_str_retry}' conflicts")
                    continue

            # pack bandチェック
            if cfg.strict_pack and ("A" in str(cfg.strict_pack)):
                if pack_map is not None:
                    pack_est = float(pack_map.get(sku, float("nan")))
                    rep = rep_pack_by_col.get(tcol)
                    if rep and not _within_band(pack_est, rep, getattr(cfg, "pack_tolerance_ratio", 0.10)):
                        _dbg_reject("pack_band", _mctx, note=f"retry: pack={pack_est}")
                        continue

            # 全チェック通過 → 受理
            lot_date_str = _lot_key_to_datestr8(lot_key)
            move_reason = m.reason if m.reason else f"列{tcol}内で最適化 → エリア内在庫バランス改善"
            accepted.append(Move(
                sku_id=sku, lot=m.lot, qty=m.qty, from_loc=from_loc, to_loc=to_loc,
                lot_date=lot_date_str, reason=move_reason,
                chain_group_id=getattr(m, 'chain_group_id', None),
                execution_order=getattr(m, 'execution_order', None),
                distance=getattr(m, 'distance', None),
            ))
            # 移動先を登録
            if _this_cg_r:
                _dest_claimed_by[to_loc] = _this_cg_r
            # シミュレーション状態更新
            shelf_usage[to_loc] = shelf_usage.get(to_loc, 0.0) + add_vol
            shelf_usage[from_loc] = max(0.0, shelf_usage.get(from_loc, 0.0) - add_vol)
            # _max_retry=0 のため現在はデッドコードだが、将来リトライ有効化時に備えて同期更新
            _sim_vol_by_loc[to_loc] = _sim_vol_by_loc.get(to_loc, 0.0) + add_vol
            _sim_vol_by_loc[from_loc] = max(0.0, _sim_vol_by_loc.get(from_loc, 0.0) - add_vol)
            _to_key_fc = (sku, int(tcol))
            _sim_inv_by_sku_col.setdefault(_to_key_fc, []).append((int(lot_key), int(tlv)))
            _from_key_fc = (sku, int(fcol))
            if _from_key_fc in _sim_inv_by_sku_col:
                _entries = _sim_inv_by_sku_col[_from_key_fc]
                _target = (int(lot_key), int(flv))
                for _ei, _ev in enumerate(_entries):
                    if _ev == _target:
                        _entries.pop(_ei)
                        break
            _to_key = (to_loc, sku)
            _from_key = (from_loc, sku)
            _sim_lots_by_loc_sku.setdefault(_to_key, set()).add(int(lot_key))
            # _sim_remaining_qty を先に更新してから残量0以下の場合のみ削除する
            _sim_skus_by_loc.setdefault(to_loc, set()).add(sku)
            _sim_remaining_qty[(to_loc, sku)] = _sim_remaining_qty.get((to_loc, sku), 0.0) + float(m.qty)
            _sim_remaining_qty[(from_loc, sku)] = _sim_remaining_qty.get((from_loc, sku), 0.0) - float(m.qty)
            if _from_key in _sim_lots_by_loc_sku:
                if _sim_remaining_qty.get((_from_key[0], _from_key[1]), 0.0) <= 0:
                    _sim_lots_by_loc_sku[_from_key].discard(int(lot_key))
            # ロット文字列ベース更新
            _lot_str_retry_upd = str(m.lot) if m.lot else ""
            if _lot_str_retry_upd:
                _sim_lot_strings_by_loc_sku.setdefault(_to_key, set()).add(_lot_str_retry_upd)
                if _from_key in _sim_lot_strings_by_loc_sku:
                    if _sim_remaining_qty.get((_from_key[0], _from_key[1]), 0.0) <= 0:
                        _sim_lot_strings_by_loc_sku[_from_key].discard(_lot_str_retry_upd)
            _from_sku_set = _sim_skus_by_loc.get(from_loc)
            if _from_sku_set:
                if _sim_remaining_qty.get((from_loc, sku), 0.0) <= 0:
                    _from_sku_set.discard(sku)
                    if not _from_sku_set:
                        del _sim_skus_by_loc[from_loc]
            _recovered += 1

        _deferred_foreign_sku = _still_deferred
        logger.debug(f"[enforce] retry round {_retry_round+1}: recovered={_recovered}, still_deferred={len(_still_deferred)}")
        if _recovered == 0:
            break

    # 最終的にまだ却下されているforeign_skuをリジェクトログに記録
    for m in _deferred_foreign_sku:
        sku = str(m.sku_id)
        to_loc = str(m.to_loc).zfill(8)
        _mctx = {"sku_id": sku, "lot": m.lot, "from": str(m.from_loc).zfill(8), "to_loc": to_loc}
        _dbg_reject("foreign_sku", _mctx, note=f"dest has foreign SKU after {_max_retry} retries")

    # ---- deferred swap groups 再評価ループ ----
    # 無効化: foreign_skuリトライと同じ理由で無効化
    if _deferred_swap_groups:
        _swap_retry_max = 0  # リトライ無効
        for _sw_retry in range(_swap_retry_max):
            if not _deferred_swap_groups:
                break
            _still_deferred_swaps: Dict[str, List[Move]] = {}
            _sw_recovered = 0
            for _sg_id, _sg_moves in _deferred_swap_groups.items():
                if _try_accept_swap_group(_sg_id, _sg_moves):
                    _sw_recovered += 1
                else:
                    # まだ通過しない場合は再保留
                    if len(_sg_moves) == 2:
                        _sg_m1, _sg_m2 = _sg_moves[0], _sg_moves[1]
                        _sg_to1 = str(_sg_m1.to_loc).zfill(8)
                        _sg_to2 = str(_sg_m2.to_loc).zfill(8)
                        _sg_sku1 = str(_sg_m1.sku_id)
                        _sg_sku2 = str(_sg_m2.sku_id)
                        _sg_skus_at_to1 = _sim_skus_by_loc.get(_sg_to1, set())
                        _sg_skus_at_to2 = _sim_skus_by_loc.get(_sg_to2, set())
                        _has_foreign1 = _sg_skus_at_to1 and not _sg_skus_at_to1 <= {_sg_sku1, _sg_sku2}
                        _has_foreign2 = _sg_skus_at_to2 and not _sg_skus_at_to2 <= {_sg_sku1, _sg_sku2}
                        if _has_foreign1 or _has_foreign2:
                            _still_deferred_swaps[_sg_id] = _sg_moves
            _deferred_swap_groups = _still_deferred_swaps
            logger.debug(f"[enforce] swap retry round {_sw_retry+1}: recovered={_sw_recovered} swaps, still_deferred={len(_still_deferred_swaps)}")
            if _sw_recovered == 0:
                break

        # まだ却下されているスワップをリジェクトログに記録
        for _sg_id, _sg_moves in _deferred_swap_groups.items():
            if len(_sg_moves) == 2:
                _sg_m1, _sg_m2 = _sg_moves[0], _sg_moves[1]
                _sg_to1 = str(_sg_m1.to_loc).zfill(8)
                _sg_to2 = str(_sg_m2.to_loc).zfill(8)
                _sg_skus_at_to1 = _sim_skus_by_loc.get(_sg_to1, set())
                _sg_skus_at_to2 = _sim_skus_by_loc.get(_sg_to2, set())
                _dbg_reject("foreign_sku", {"swap_group": _sg_id},
                            note=f"swap deferred: to1_skus={_sg_skus_at_to1}, to2_skus={_sg_skus_at_to2}")

    if _swap_group_ids:
        logger.warning(f"[enforce] Swap processing: {_swap_accepted} accepted, {_swap_rejected} rejected from {len(_swap_group_ids)} swap groups")

    # ---- チェーングループ整合性チェック ----
    # チェーングループ内の一部が却下された場合、残りも除去する
    # (例: 立ち退きmoveが却下されたのにメインmoveだけ残ると、移動先にSKU混在が発生する)
    _input_chain_counts: Dict[str, int] = {}
    for m in moves:
        _cg = getattr(m, 'chain_group_id', None)
        if _cg:
            _input_chain_counts[_cg] = _input_chain_counts.get(_cg, 0) + 1
    _accepted_chain_counts: Dict[str, int] = {}
    for a in accepted:
        _cg = getattr(a, 'chain_group_id', None)
        if _cg:
            _accepted_chain_counts[_cg] = _accepted_chain_counts.get(_cg, 0) + 1
    _incomplete_chains: Set[str] = set()
    for _cg, _cnt in _input_chain_counts.items():
        if _accepted_chain_counts.get(_cg, 0) != _cnt:
            _incomplete_chains.add(_cg)
    if _incomplete_chains:
        _before = len(accepted)
        accepted = [a for a in accepted
                    if not (getattr(a, 'chain_group_id', None) and
                            a.chain_group_id in _incomplete_chains)]
        _removed = _before - len(accepted)
        if _removed:
            logger.debug(f"[optimizer] chain integrity: removed {_removed} moves from {len(_incomplete_chains)} incomplete chain groups")
            _dbg_reject("chain_incomplete", {"count": _removed},
                        note=f"chain_groups={_incomplete_chains}")

    # ---- 事後SKU混在検証: 全移動適用後のSKU混在数がbaseline以下になるまで貪欲法で除去 ----
    # original_skus_by_loc（品質フィルタ前の全在庫）を使用してSKU混在をカウント
    if accepted and original_skus_by_loc is not None and original_qty_by_loc_sku is not None:
        _sm_init_skus = original_skus_by_loc
        _sm_init_qty = original_qty_by_loc_sku
        _sm_baseline = sum(1 for v in _sm_init_skus.values() if len(v) > 1)
        _sm_post = _count_sku_mixing(accepted, _sm_init_skus, _sm_init_qty)
        if _sm_post > _sm_baseline:
            logger.warning(f"[enforce] post-SKU-mix: baseline={_sm_baseline}, post={_sm_post}, removing offenders...")
            _sm_checked: Set[str] = set()
            _sm_max_iters = 50
            for _sm_iter in range(_sm_max_iters):
                if _sm_post <= _sm_baseline:
                    break
                best_cg = None
                best_imp = 0
                best_post = _sm_post
                for a in accepted:
                    cg = getattr(a, 'chain_group_id', None) or ""
                    if cg in _sm_checked:
                        continue
                    if cg:
                        _sm_checked.add(cg)
                    trial = [x for x in accepted if (getattr(x, 'chain_group_id', None) or "") != cg] if cg else [x for x in accepted if x is not a]
                    t_sm = _count_sku_mixing(trial, _sm_init_skus, _sm_init_qty)
                    imp = _sm_post - t_sm
                    if imp > best_imp:
                        best_imp = imp
                        best_cg = cg
                        best_post = t_sm
                if best_cg is None or best_imp <= 0:
                    break
                accepted = [x for x in accepted if (getattr(x, 'chain_group_id', None) or "") != best_cg]
                _sm_post = best_post
            logger.warning(f"[enforce] post-SKU-mix done: {_sm_baseline}->{_sm_post}, moves={len(accepted)}")

    # ---- 事後FIFO検証: 全承認移動を適用したシミュレーションでFIFO違反を検出し除去 ----
    # ---- 事後FIFO検証: 移動を逐次適用し、各ステップでFIFO違反を検出・チェーン単位で除去 ----
    if cfg.hard_fifo and accepted:
        _max_fifo_rounds = 5
        for _fifo_round in range(_max_fifo_rounds):
            # 初期状態を構築（フィルタ前全在庫）
            _post_scl: Dict[Tuple[str, int], list] = {}
            if original_inv_lot_levels:
                _post_scl = {k: list(v) for k, v in original_inv_lot_levels.items()}
            elif "商品ID" in inv.columns and "col" in inv.columns and "lot_key" in inv.columns:
                for _pi in range(len(inv)):
                    _p_sku = str(inv.iloc[_pi]["商品ID"])
                    _p_col = int(inv.iloc[_pi].get("col", 0) or 0)
                    _p_lk = int(inv.iloc[_pi].get("lot_key", UNKNOWN_LOT_KEY) or UNKNOWN_LOT_KEY)
                    _p_lv = int(inv.iloc[_pi].get("lv", 0) or 0)
                    _p_qty = float(inv.iloc[_pi].get("qty_cases_move", inv.iloc[_pi].get("ケース", 0)) or 0)
                    if _p_lk != UNKNOWN_LOT_KEY and _p_qty > 0:
                        _post_scl.setdefault((_p_sku, _p_col), []).append((_p_lk, _p_lv))

            # 逐次適用: 各移動を1件ずつ適用し、適用時点でFIFO違反をチェック
            _fifo_bad_chains: Set[str] = set()
            _fifo_bad_indices: Set[int] = set()  # チェーンなし移動のインデックス追跡
            for _idx_a, a in enumerate(accepted):
                _a_sku = str(a.sku_id)
                _a_lk = _parse_lot_date_key(a.lot)
                if _a_lk == UNKNOWN_LOT_KEY:
                    continue
                _a_flv, _a_fcol, _ = _parse_loc8(str(a.from_loc).zfill(8))
                _a_tlv, _a_tcol, _ = _parse_loc8(str(a.to_loc).zfill(8))

                # 適用前にFIFO違反チェック: 移動先列の現在の状態で判定
                # strict inequality (同一段は許容) — _violates_lot_level_rule_fast と統一
                _entries = _post_scl.get((_a_sku, _a_tcol), [])
                _creates_violation = False
                for _ek, _elv in _entries:
                    if _ek == _a_lk:
                        continue
                    if (_a_lk < _ek and _a_tlv > _elv) or (_a_lk > _ek and _a_tlv < _elv):
                        _creates_violation = True
                        break

                if _creates_violation:
                    _cg = getattr(a, 'chain_group_id', None)
                    if _cg:
                        _fifo_bad_chains.add(_cg)
                    else:
                        _fifo_bad_indices.add(_idx_a)

                # 状態更新（違反有無にかかわらず適用 — チェーン除去は後で一括）
                _fk = (_a_sku, _a_fcol)
                if _fk in _post_scl:
                    for _ei, _ev in enumerate(_post_scl[_fk]):
                        if _ev == (_a_lk, _a_flv):
                            _post_scl[_fk].pop(_ei)
                            break
                _post_scl.setdefault((_a_sku, _a_tcol), []).append((_a_lk, _a_tlv))

            if not _fifo_bad_chains and not _fifo_bad_indices:
                break

            # チェーン単位 + インデックス単位で除去
            _before_fifo = len(accepted)
            accepted = [a for _idx_a, a in enumerate(accepted)
                        if _idx_a not in _fifo_bad_indices and
                        not (getattr(a, 'chain_group_id', None) and
                             a.chain_group_id in _fifo_bad_chains)]
            _removed_fifo = _before_fifo - len(accepted)
            logger.debug(f"[enforce] FIFO post-check round {_fifo_round+1}: removed {_removed_fifo} moves ({len(_fifo_bad_chains)} chains, {len(_fifo_bad_indices)} unchained)")
            if _removed_fifo == 0:
                break

    # debug: finalize accepted count
    _dbg_note_accepted(len(accepted))
    try:
        rej = _last_relocation_debug.get("rejections", {})
        if rej:
            breakdown = ", ".join(f"{k}={int(v)}" for k, v in rej.items())
            logger.warning(f"[optimizer] constraints accepted={len(accepted)}/{_last_relocation_debug.get('planned', len(moves))}; rejects: {breakdown}")
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
        near = set(getattr(cfg, "near_cols", getattr(cfg, "small_pack_cols", tuple(range(24, 35)))))
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
# Pass-C: 集約（同じSKU×ロットの分散在庫を1箇所にまとめる）
# ============================================================================
def _pass_consolidate(
    inv: pd.DataFrame,
    shelf_usage: Dict[str, float],
    cap_limit: float,
    cfg: OptimizerConfig,
    sku_vol_map: pd.Series,
    *,
    cap_by_loc: Optional[Dict[str, float]] = None,
    can_receive: Optional[Set[str]] = None,
    skus_by_loc: Optional[Dict[str, Set[str]]] = None,
    lots_by_loc_sku: Optional[Dict[Tuple[str, str], Set[int]]] = None,
    moved_indices: Optional[Set[int]] = None,
    blocked_dest_locs: Optional[Set[str]] = None,
    blocked_source_locs: Optional[Set[str]] = None,
) -> List[Move]:
    """同じSKU×同じロットの分散在庫を1箇所に集約する。

    ロケーションを空けることで、後続のFIFO是正やゾーンスワップで
    空いたロケーションを活用できるようにする。
    """
    moves: List[Move] = []

    if inv.empty:
        return moves

    # 有効な在庫のみ（容積あり、移動対象）
    filter_cond = inv["volume_each_case"] > 0
    if "is_movable" in inv.columns:
        filter_cond = filter_cond & (inv["is_movable"] == True)
    subset = inv[filter_cond].copy()
    if subset.empty:
        return moves

    _blocked = blocked_dest_locs or set()
    _blocked_src = blocked_source_locs or set()

    # 品質区分列を特定
    q_col = None
    if "quality_name" in subset.columns:
        q_col = "quality_name"
    elif "品質区分名" in subset.columns:
        q_col = "品質区分名"

    # (SKU, ロット文字列, 品質区分) でグループ化
    # ロットは文字列ベースで完全一致（lot_keyは日付正規化で同日異ロットを区別できないため）
    group_cols = ["商品ID", "ロット"]
    if q_col:
        group_cols.append(q_col)

    grouped = subset.groupby(group_cols)

    for group_key, group_df in grouped:
        if len(group_df) <= 1:
            continue  # 1箇所しかない → 集約不要

        # 各ロケーションの情報を収集
        loc_infos = []
        for idx, row in group_df.iterrows():
            from_loc = str(row["ロケーション"]).zfill(8)
            qty_cases = int(row.get("qty_cases_move") or 0)
            vol_each = float(row.get("volume_each_case") or 0.0)

            if qty_cases <= 0 or vol_each <= 0:
                continue

            # 既に移動済みのインデックスはスキップ
            if moved_indices is not None and idx in moved_indices:
                continue

            # blocked_source_locs（multi_sku_locs）はスキップ
            if from_loc in _blocked_src:
                continue

            loc_infos.append({
                "idx": idx,
                "loc": from_loc,
                "qty_cases": qty_cases,
                "vol_each": vol_each,
                "need_vol": qty_cases * vol_each,
                "row": row,
            })

        if len(loc_infos) <= 1:
            continue  # 有効ロケ1つ以下 → 集約不要

        # 移動先を選ぶ: 最も容量に余裕があるロケーション（= used / cap が最小）
        # cap_by_loc があればロケ個別容量を、なければ base_cap を使用
        base_cap = cap_limit
        best_dest = None
        best_remaining = -1.0

        for li in loc_infos:
            loc = li["loc"]

            # can_receive チェック
            if can_receive is not None and loc not in can_receive:
                continue

            # blocked_dest チェック
            if loc in _blocked:
                continue

            loc_cap = cap_by_loc.get(loc, base_cap) if cap_by_loc else base_cap
            loc_used = float(shelf_usage.get(loc, 0.0))
            remaining = loc_cap - loc_used

            if remaining > best_remaining:
                best_remaining = remaining
                best_dest = li

        if best_dest is None:
            continue

        dest_loc = best_dest["loc"]
        dest_cap = cap_by_loc.get(dest_loc, base_cap) if cap_by_loc else base_cap

        # 移動先以外のロケから移動先へ集約
        for li in loc_infos:
            if li["loc"] == dest_loc:
                continue  # 移動先自身はスキップ

            src_loc = li["loc"]
            need_vol = li["need_vol"]

            # 容量チェック: 移動先の使用量 + 移動量 <= 移動先容量
            current_used = float(shelf_usage.get(dest_loc, 0.0))
            if current_used + need_vol > dest_cap:
                continue  # 容量不足 → この行はスキップ

            # 移動先がblocked_destでないこと（再確認）
            if dest_loc in _blocked:
                continue

            # can_receiveチェック（再確認）
            if can_receive is not None and dest_loc not in can_receive:
                continue

            # ロット日付文字列
            lot_key = int(li["row"].get("lot_key", UNKNOWN_LOT_KEY))
            lot_date_str = _lot_key_to_datestr8(lot_key)

            chain_id = f"p_consol_{secrets.token_hex(6)}"

            moves.append(Move(
                sku_id=str(li["row"]["商品ID"]),
                lot=str(li["row"].get("ロット") or ""),
                qty=li["qty_cases"],
                from_loc=src_loc,
                to_loc=dest_loc,
                lot_date=lot_date_str,
                reason="集約(同一ロット統合)",
                chain_group_id=chain_id,
                execution_order=1,
            ))

            # shelf_usage を更新
            shelf_usage[src_loc] = max(0.0, shelf_usage.get(src_loc, 0.0) - need_vol)
            shelf_usage[dest_loc] = shelf_usage.get(dest_loc, 0.0) + need_vol

            # skus_by_loc を逐次更新
            if skus_by_loc is not None:
                sku_str = str(li["row"]["商品ID"])
                skus_by_loc.setdefault(dest_loc, set()).add(sku_str)
                _from_skus = skus_by_loc.get(src_loc)
                if _from_skus:
                    # 移動元にこのSKUの他の行がまだあるか確認
                    _still_at_from = inv[
                        (inv["ロケーション"].astype(str) == src_loc) &
                        (inv["商品ID"].astype(str) == sku_str) &
                        (inv.index != li["idx"])
                    ]
                    if _still_at_from.empty:
                        _from_skus.discard(sku_str)

            # lots_by_loc_sku を逐次更新
            if lots_by_loc_sku is not None:
                sku_str = str(li["row"]["商品ID"])
                _to_key = (dest_loc, sku_str)
                lots_by_loc_sku.setdefault(_to_key, set()).add(lot_key)
                _from_key = (src_loc, sku_str)
                if _from_key in lots_by_loc_sku:
                    lots_by_loc_sku[_from_key].discard(lot_key)
                    if not lots_by_loc_sku[_from_key]:
                        del lots_by_loc_sku[_from_key]

            # 二重移動防止
            if moved_indices is not None:
                moved_indices.add(li["idx"])

            # inv のロケーション情報を更新（後続パスが更新後の状態を参照できるように）
            to_lv, to_col, to_dep = _parse_loc8(dest_loc)
            inv.at[li["idx"], "ロケーション"] = dest_loc
            inv.at[li["idx"], "lv"] = to_lv
            inv.at[li["idx"], "col"] = to_col
            inv.at[li["idx"], "dep"] = to_dep

    return moves


# ============================================================================
# Pass-FIFO-NEW: 最古ロットをLv1-2へ配置する単一統合FIFOパス
# ============================================================================

def _handle_same_column_swaps(
    sku: str,
    sku_sorted: pd.DataFrame,
    moves: list,
    shelf_usage: Dict[str, float],
    skus_by_loc: Optional[Dict[str, Set[str]]],
    lots_by_loc_sku: Optional[Dict[Tuple[str, str], Set[int]]],
    lot_strings_by_loc_sku: Optional[Dict[Tuple[str, str], Set[str]]],
    moved_indices: Optional[Set[int]],
    _can_add,
) -> None:
    """同列内で古ロット(Lv3-4) ↔ 新ロット(Lv1-2) スワップを生成する。"""
    for col_val, col_df in sku_sorted.groupby("col"):
        col_lv34 = col_df[col_df["lv"].isin([3, 4])].sort_values("lot_key", ascending=True)
        col_lv12 = col_df[col_df["lv"].isin([1, 2])].sort_values("lot_key", ascending=False)
        for (old_idx, old_row), (new_idx, new_row) in zip(
            col_lv34.iterrows(), col_lv12.iterrows()
        ):
            if not _can_add(2):
                return
            if moved_indices is not None and (old_idx in moved_indices or new_idx in moved_indices):
                continue
            old_lk = int(old_row["lot_key"])
            new_lk = int(new_row["lot_key"])
            if old_lk >= new_lk:
                continue  # FIFO違反なし
            old_loc = str(old_row["ロケーション"]).zfill(8)
            new_loc = str(new_row["ロケーション"]).zfill(8)
            # 他SKU混在チェック: どちらかのlocに自SKU以外がいたらスキップ
            if skus_by_loc is not None:
                old_loc_skus = skus_by_loc.get(old_loc, set())
                new_loc_skus = skus_by_loc.get(new_loc, set())
                if (old_loc_skus and not old_loc_skus <= {str(sku)}) or \
                   (new_loc_skus and not new_loc_skus <= {str(sku)}):
                    continue
            old_vol = float(old_row.get("volume_each_case", 0.0)) * int(old_row.get("qty_cases_move") or 0)
            new_vol = float(new_row.get("volume_each_case", 0.0)) * int(new_row.get("qty_cases_move") or 0)
            old_date = _lot_key_to_datestr8(old_lk)
            new_date = _lot_key_to_datestr8(new_lk)
            cg = f"swap_fifo_{secrets.token_hex(6)}"
            moves.append(Move(
                sku_id=str(sku),
                lot=str(new_row.get("ロット") or ""),
                qty=int(new_row.get("qty_cases_move") or 0),
                from_loc=new_loc,
                to_loc=old_loc,
                lot_date=new_date,
                reason="FIFO: 新ロット→Lv3-4 (同列スワップ退避)",
                chain_group_id=cg,
                execution_order=1,
            ))
            moves.append(Move(
                sku_id=str(sku),
                lot=str(old_row.get("ロット") or ""),
                qty=int(old_row.get("qty_cases_move") or 0),
                from_loc=old_loc,
                to_loc=new_loc,
                lot_date=old_date,
                reason="FIFO: 古ロット→Lv1-2 (同列スワップ)",
                chain_group_id=cg,
                execution_order=2,
            ))
            # shelf_usage: 同列スワップなので容積は相互に入れ替わるだけ
            shelf_usage[old_loc] = max(0.0, shelf_usage.get(old_loc, 0.0) - old_vol + new_vol)
            shelf_usage[new_loc] = max(0.0, shelf_usage.get(new_loc, 0.0) - new_vol + old_vol)
            # lots_by_loc_sku 更新
            if lots_by_loc_sku is not None:
                _old_key = (old_loc, str(sku))
                _new_key = (new_loc, str(sku))
                lots_by_loc_sku.setdefault(_old_key, set()).add(new_lk)
                lots_by_loc_sku[_old_key].discard(old_lk)
                lots_by_loc_sku.setdefault(_new_key, set()).add(old_lk)
                lots_by_loc_sku[_new_key].discard(new_lk)
            # lot_strings_by_loc_sku 更新
            if lot_strings_by_loc_sku is not None:
                _old_ls = str(old_row.get("ロット") or "")
                _new_ls = str(new_row.get("ロット") or "")
                _ok = (old_loc, str(sku))
                _nk = (new_loc, str(sku))
                lot_strings_by_loc_sku.setdefault(_ok, set()).add(_new_ls)
                lot_strings_by_loc_sku[_ok].discard(_old_ls)
                lot_strings_by_loc_sku.setdefault(_nk, set()).add(_old_ls)
                lot_strings_by_loc_sku[_nk].discard(_new_ls)
            if moved_indices is not None:
                moved_indices.add(old_idx)
                moved_indices.add(new_idx)


def _find_empty_pick_slot_soft_pack(
    sku: str,
    row: pd.Series,
    sku_cols: Set[int],
    empty_pick_slots_by_col: Dict[int, list],
    rep_pack_by_col: Optional[Dict[int, float]],
    tolerance: float,
    shelf_usage: Dict[str, float],
    cap_limit: float,
    cap_by_loc: Optional[Dict[str, float]],
    skus_by_loc: Optional[Dict[str, Set[str]]],
    lots_by_loc_sku: Optional[Dict[Tuple[str, str], Set[int]]],
    lot_strings_by_loc_sku: Optional[Dict[Tuple[str, str], Set[str]]],
) -> Optional[str]:
    """空きLv1-2スロットをpack帯soft・同列優先で探索する。"""
    need_vol = float(row.get("volume_each_case", 0.0)) * int(row.get("qty_cases_move") or 0)
    if need_vol <= 0:
        return None
    sku_pack = float(row.get("pack_per_case") or row.get("pack") or 0.0)
    row_lot_key = int(row.get("lot_key", 0))
    row_lot_str = str(row.get("ロット") or "")

    candidates = []
    for col_c, slot_list in empty_pick_slots_by_col.items():
        for (avail_cap_cached, loc) in slot_list:
            lv, col_loc, dep = _parse_loc8(loc)
            # 容量再確認 (キャッシュは古い可能性)
            used = float(shelf_usage.get(loc, 0.0))
            limit = cap_by_loc.get(loc, cap_limit) if cap_by_loc else cap_limit
            avail = limit - used
            if avail < need_vol:
                continue
            # 他SKU混在チェック
            if skus_by_loc is not None:
                existing = skus_by_loc.get(loc, set())
                if existing and not existing <= {str(sku)}:
                    continue
            # lot_key ベースのロット混在チェック
            if lots_by_loc_sku is not None:
                _lk_key = (loc, str(sku))
                _existing_lks = lots_by_loc_sku.get(_lk_key, set())
                if _existing_lks:
                    _known = _existing_lks - {UNKNOWN_LOT_KEY}
                    if _known and row_lot_key not in _known:
                        continue
            # ロット文字列ベースのロット混在チェック
            if lot_strings_by_loc_sku is not None:
                _ls_key = (loc, str(sku))
                _existing_ls = lot_strings_by_loc_sku.get(_ls_key, set())
                if _existing_ls and row_lot_str not in _existing_ls:
                    continue
            # スコア計算
            pack_score = 0.0
            if rep_pack_by_col and sku_pack > 0:
                rep = rep_pack_by_col.get(col_loc, 0.0)
                if rep > 0 and abs(rep - sku_pack) / max(rep, sku_pack) <= tolerance:
                    pack_score = -100.0
            same_col_bonus = -1000.0 if col_loc in sku_cols else 0.0
            ease = lv * 10000 + col_loc * 100 + dep
            score = pack_score + same_col_bonus + ease
            candidates.append((score, loc))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _find_swap_target_cross_column(
    sku: str,
    row: pd.Series,
    lv12_rows: pd.DataFrame,
    moved_indices: Optional[Set[int]],
    skus_by_loc: Optional[Dict[str, Set[str]]] = None,
) -> Optional[pd.Series]:
    """列をまたいだスワップ候補(Lv1-2にいる自SKUの新ロット行)を探す。"""
    row_lot_key = int(row.get("lot_key", 0))
    row_from_loc = str(row.get("ロケーション", "")).zfill(8)

    # row (Lv3-4) の from_loc に他SKUがいたらスワップ不可
    if skus_by_loc is not None:
        row_from_skus = skus_by_loc.get(row_from_loc, set())
        if row_from_skus and not row_from_skus <= {str(sku)}:
            return None

    # 新しいロットを持つLv1-2行を lot_key 降順で候補リスト化
    newer_rows = lv12_rows[lv12_rows["lot_key"] > row_lot_key].sort_values(
        "lot_key", ascending=False
    )
    for idx, cand_row in newer_rows.iterrows():
        if moved_indices is not None and idx in moved_indices:
            continue
        # cand_row (Lv1-2) の from_loc に他SKUがいたらスキップ
        if skus_by_loc is not None:
            cand_loc = str(cand_row.get("ロケーション", "")).zfill(8)
            cand_skus = skus_by_loc.get(cand_loc, set())
            if cand_skus and not cand_skus <= {str(sku)}:
                continue
        return cand_row
    return None


def _find_cross_sku_swap_target(
    sku: str,
    row: pd.Series,
    passing_inv: pd.DataFrame,
    pick_levels: tuple,
    skus_by_loc: Optional[Dict[str, Set[str]]],
    moved_indices: Optional[Set[int]],
    consumed_y_rows: Set[Tuple[str, str]],
) -> Optional[Tuple[str, pd.Series]]:
    """他SKU Y の Lv1-2 行とスワップする候補を探す (Policy 1)。

    条件:
      - X (row) の元ロケは 1-SKU クリーン
      - Y の元ロケは 1-SKU クリーン (Y 単独)
      - Y lot_key > X lot_key (X のほうが古い)
      - Y の当該行は Y の OLDEST ではない
      - Y は当該行以外に Lv1-2 を持っている
      - moved_indices で除外されていない
    """
    x_lot_key = int(row.get("lot_key", 0))
    x_from_loc = str(row.get("ロケーション", "")).zfill(8)

    # X の元ロケがクリーンか
    if skus_by_loc is not None:
        x_skus = skus_by_loc.get(x_from_loc, set())
        if x_skus and not x_skus <= {str(sku)}:
            return None

    # Lv1-2 の全行 (他SKU)、lot_key > X
    lv12_all = passing_inv[passing_inv["lv"].isin(pick_levels)]
    candidates = lv12_all[
        (lv12_all["商品ID"].astype(str) != str(sku)) &
        (lv12_all["lot_key"] > x_lot_key)
    ].sort_values("lot_key", ascending=False)

    # SKU別の全行を事前計算 (重複 filter を避ける)
    y_all_cache: Dict[str, pd.DataFrame] = {}

    for idx, cand_row in candidates.iterrows():
        if moved_indices is not None and idx in moved_indices:
            continue
        y_sku = str(cand_row["商品ID"])
        y_loc = str(cand_row["ロケーション"]).zfill(8)
        # consumed_y_rows チェック
        if (y_loc, y_sku) in consumed_y_rows:
            continue
        # Y の元ロケがクリーンか
        if skus_by_loc is not None:
            y_skus = skus_by_loc.get(y_loc, set())
            if y_skus and not y_skus <= {y_sku}:
                continue
        # Y の全行 (キャッシュ利用)
        if y_sku not in y_all_cache:
            y_all_cache[y_sku] = passing_inv[passing_inv["商品ID"].astype(str) == y_sku]
        y_all = y_all_cache[y_sku]
        if y_all.empty:
            continue
        y_min_lot = int(y_all["lot_key"].min())
        y_lot_key = int(cand_row["lot_key"])
        # 条件: Y の当該行は Y の OLDEST ではない
        if y_lot_key == y_min_lot:
            continue
        # 条件: Y は当該行以外に Lv1-2 を持っている
        y_lv12 = y_all[y_all["lv"].isin(pick_levels)]
        other_lv12 = y_lv12[y_lv12["ロケーション"].astype(str).str.zfill(8) != y_loc]
        if other_lv12.empty:
            continue
        return (y_loc, cand_row)
    return None


def _emit_move_fifo(
    moves: list,
    row: pd.Series,
    to_loc: str,
    reason: str,
    shelf_usage: Dict[str, float],
    sku_vol_map: Optional[pd.Series],
    skus_by_loc: Optional[Dict[str, Set[str]]],
    lots_by_loc_sku: Optional[Dict[Tuple[str, str], Set[int]]],
    lot_strings_by_loc_sku: Optional[Dict[Tuple[str, str], Set[str]]],
    moved_indices: Optional[Set[int]],
    empty_pick_slots_by_col: Optional[Dict[int, list]],
    row_idx,
) -> None:
    """単独移動Moveを生成し、各インデックスを更新する。"""
    sku = str(row.get("商品ID", ""))
    lot_key = int(row.get("lot_key", 0))
    qty = int(row.get("qty_cases_move") or 0)
    vol_each = float(row.get("volume_each_case", 0.0))
    need_vol = vol_each * qty
    from_loc = str(row.get("ロケーション", "")).zfill(8)
    to_loc_z = str(to_loc).zfill(8)
    lot_str = str(row.get("ロット") or "")
    lot_date = _lot_key_to_datestr8(lot_key)

    moves.append(Move(
        sku_id=sku,
        lot=lot_str,
        qty=qty,
        from_loc=from_loc,
        to_loc=to_loc_z,
        lot_date=lot_date,
        reason=reason,
        chain_group_id=None,
        execution_order=None,
    ))
    # shelf_usage 更新
    shelf_usage[from_loc] = max(0.0, shelf_usage.get(from_loc, 0.0) - need_vol)
    shelf_usage[to_loc_z] = shelf_usage.get(to_loc_z, 0.0) + need_vol
    # skus_by_loc 更新
    if skus_by_loc is not None:
        skus_by_loc.setdefault(to_loc_z, set()).add(sku)
    # lots_by_loc_sku 更新
    if lots_by_loc_sku is not None:
        _to_key = (to_loc_z, sku)
        lots_by_loc_sku.setdefault(_to_key, set()).add(lot_key)
        _from_key = (from_loc, sku)
        if _from_key in lots_by_loc_sku:
            lots_by_loc_sku[_from_key].discard(lot_key)
    # lot_strings_by_loc_sku 更新
    if lot_strings_by_loc_sku is not None:
        lot_strings_by_loc_sku.setdefault((to_loc_z, sku), set()).add(lot_str)
        _from_ls = (from_loc, sku)
        if _from_ls in lot_strings_by_loc_sku:
            lot_strings_by_loc_sku[_from_ls].discard(lot_str)
    # moved_indices 更新
    if moved_indices is not None:
        moved_indices.add(row_idx)
    # empty_pick_slots_by_col から到着ロケの容量を再チェック（使い切った場合は除去）
    if empty_pick_slots_by_col is not None:
        to_lv, to_col, to_dep = _parse_loc8(to_loc_z)
        if to_lv in (1, 2):
            slot_list = empty_pick_slots_by_col.get(to_col, [])
            # 容量が 0 以下になったスロットを除去
            new_list = [(cap, l) for (cap, l) in slot_list if l != to_loc_z or (cap - need_vol) > 0]
            empty_pick_slots_by_col[to_col] = new_list


def _emit_swap_pair_fifo(
    moves: list,
    old_row: pd.Series,
    new_row: pd.Series,
    old_idx,
    new_idx,
    reason_old: str,
    reason_new: str,
    shelf_usage: Dict[str, float],
    skus_by_loc: Optional[Dict[str, Set[str]]],
    lots_by_loc_sku: Optional[Dict[Tuple[str, str], Set[int]]],
    lot_strings_by_loc_sku: Optional[Dict[Tuple[str, str], Set[str]]],
    moved_indices: Optional[Set[int]],
) -> None:
    """列間スワップMoveペアを生成し、各インデックスを更新する。"""
    sku = str(old_row.get("商品ID", ""))
    old_lk = int(old_row.get("lot_key", 0))
    new_lk = int(new_row.get("lot_key", 0))
    old_qty = int(old_row.get("qty_cases_move") or 0)
    new_qty = int(new_row.get("qty_cases_move") or 0)
    old_vol_each = float(old_row.get("volume_each_case", 0.0))
    new_vol_each = float(new_row.get("volume_each_case", 0.0))
    old_vol = old_vol_each * old_qty
    new_vol = new_vol_each * new_qty
    old_loc = str(old_row.get("ロケーション", "")).zfill(8)
    new_loc = str(new_row.get("ロケーション", "")).zfill(8)
    old_lot_str = str(old_row.get("ロット") or "")
    new_lot_str = str(new_row.get("ロット") or "")
    old_date = _lot_key_to_datestr8(old_lk)
    new_date = _lot_key_to_datestr8(new_lk)
    cg = f"swap_fifo_{secrets.token_hex(6)}"

    # Move 1: 新ロット Lv1-2 → Lv3-4
    moves.append(Move(
        sku_id=sku,
        lot=new_lot_str,
        qty=new_qty,
        from_loc=new_loc,
        to_loc=old_loc,
        lot_date=new_date,
        reason=reason_new,
        chain_group_id=cg,
        execution_order=1,
    ))
    # Move 2: 古ロット Lv3-4 → Lv1-2
    moves.append(Move(
        sku_id=sku,
        lot=old_lot_str,
        qty=old_qty,
        from_loc=old_loc,
        to_loc=new_loc,
        lot_date=old_date,
        reason=reason_old,
        chain_group_id=cg,
        execution_order=2,
    ))
    # shelf_usage 更新
    shelf_usage[old_loc] = max(0.0, shelf_usage.get(old_loc, 0.0) - old_vol + new_vol)
    shelf_usage[new_loc] = max(0.0, shelf_usage.get(new_loc, 0.0) - new_vol + old_vol)
    # skus_by_loc 更新 (同SKUなので追加のみ)
    if skus_by_loc is not None:
        skus_by_loc.setdefault(old_loc, set()).add(sku)
        skus_by_loc.setdefault(new_loc, set()).add(sku)
    # lots_by_loc_sku 更新
    if lots_by_loc_sku is not None:
        _ok = (old_loc, sku)
        _nk = (new_loc, sku)
        lots_by_loc_sku.setdefault(_ok, set()).add(new_lk)
        lots_by_loc_sku[_ok].discard(old_lk)
        lots_by_loc_sku.setdefault(_nk, set()).add(old_lk)
        lots_by_loc_sku[_nk].discard(new_lk)
    # lot_strings_by_loc_sku 更新
    if lot_strings_by_loc_sku is not None:
        _ok_ls = (old_loc, sku)
        _nk_ls = (new_loc, sku)
        lot_strings_by_loc_sku.setdefault(_ok_ls, set()).add(new_lot_str)
        lot_strings_by_loc_sku[_ok_ls].discard(old_lot_str)
        lot_strings_by_loc_sku.setdefault(_nk_ls, set()).add(old_lot_str)
        lot_strings_by_loc_sku[_nk_ls].discard(new_lot_str)
    if moved_indices is not None:
        moved_indices.add(old_idx)
        moved_indices.add(new_idx)


def _emit_cross_sku_swap(
    moves: list,
    x_row: pd.Series,
    y_loc: str,
    y_row: pd.Series,
    shelf_usage: Dict[str, float],
    skus_by_loc: Optional[Dict[str, Set[str]]],
    lots_by_loc_sku: Optional[Dict[Tuple[str, str], Set[int]]],
    lot_strings_by_loc_sku: Optional[Dict[Tuple[str, str], Set[str]]],
    moved_indices: Optional[Set[int]],
    consumed_y_rows: Set[Tuple[str, str]],
) -> None:
    """クロスSKUスワップ: X(Lv3-4) ↔ Y(Lv1-2) の2手 atomic Move を生成。

    chain_group_id = swap_cross_sku_{hex}
    enforce_constraints の swap_ プレフィックス atomic 検証が効く。
    """
    x_sku = str(x_row.get("商品ID", ""))
    x_from = str(x_row.get("ロケーション", "")).zfill(8)
    x_lot_key = int(x_row.get("lot_key", 0))
    x_lot_str = str(x_row.get("ロット") or "")
    x_qty = int(x_row.get("qty_cases_move") or 0)
    x_vol = float(x_row.get("volume_each_case", 0.0)) * x_qty
    x_lot_date = _lot_key_to_datestr8(x_lot_key)

    y_sku = str(y_row.get("商品ID", ""))
    y_from = str(y_loc).zfill(8)
    y_lot_key = int(y_row.get("lot_key", 0))
    y_lot_str = str(y_row.get("ロット") or "")
    y_qty = int(y_row.get("qty_cases_move") or 0)
    y_vol = float(y_row.get("volume_each_case", 0.0)) * y_qty
    y_lot_date = _lot_key_to_datestr8(y_lot_key)

    cg = f"swap_cross_sku_{secrets.token_hex(6)}"

    # Move 1: Y → X の元ロケ (退避)
    moves.append(Move(
        sku_id=y_sku, lot=y_lot_str, qty=y_qty,
        from_loc=y_from, to_loc=x_from,
        lot_date=y_lot_date,
        reason="FIFO: 他SKU退避 (クロスSKUスワップ)",
        chain_group_id=cg, execution_order=1,
    ))
    # Move 2: X → Y の元ロケ (配置)
    moves.append(Move(
        sku_id=x_sku, lot=x_lot_str, qty=x_qty,
        from_loc=x_from, to_loc=y_from,
        lot_date=x_lot_date,
        reason="FIFO: 最古→Lv1-2 (クロスSKUスワップ)",
        chain_group_id=cg, execution_order=2,
    ))

    # shelf_usage: X と Y の容積を入れ替え
    shelf_usage[x_from] = max(0.0, shelf_usage.get(x_from, 0.0) - x_vol + y_vol)
    shelf_usage[y_from] = max(0.0, shelf_usage.get(y_from, 0.0) - y_vol + x_vol)

    # skus_by_loc: x_from は X → Y, y_from は Y → X
    if skus_by_loc is not None:
        if x_from in skus_by_loc:
            skus_by_loc[x_from].discard(x_sku)
            skus_by_loc[x_from].add(y_sku)
        else:
            skus_by_loc[x_from] = {y_sku}
        if y_from in skus_by_loc:
            skus_by_loc[y_from].discard(y_sku)
            skus_by_loc[y_from].add(x_sku)
        else:
            skus_by_loc[y_from] = {x_sku}

    # lots_by_loc_sku 更新
    if lots_by_loc_sku is not None:
        _k_xf = (x_from, x_sku)
        if _k_xf in lots_by_loc_sku:
            lots_by_loc_sku[_k_xf].discard(x_lot_key)
            if not lots_by_loc_sku[_k_xf]:
                del lots_by_loc_sku[_k_xf]
        _k_yf = (y_from, y_sku)
        if _k_yf in lots_by_loc_sku:
            lots_by_loc_sku[_k_yf].discard(y_lot_key)
            if not lots_by_loc_sku[_k_yf]:
                del lots_by_loc_sku[_k_yf]
        lots_by_loc_sku.setdefault((x_from, y_sku), set()).add(y_lot_key)
        lots_by_loc_sku.setdefault((y_from, x_sku), set()).add(x_lot_key)

    # lot_strings_by_loc_sku 更新
    if lot_strings_by_loc_sku is not None:
        _kx = (x_from, x_sku)
        if _kx in lot_strings_by_loc_sku:
            lot_strings_by_loc_sku[_kx].discard(x_lot_str)
            if not lot_strings_by_loc_sku[_kx]:
                del lot_strings_by_loc_sku[_kx]
        _ky = (y_from, y_sku)
        if _ky in lot_strings_by_loc_sku:
            lot_strings_by_loc_sku[_ky].discard(y_lot_str)
            if not lot_strings_by_loc_sku[_ky]:
                del lot_strings_by_loc_sku[_ky]
        lot_strings_by_loc_sku.setdefault((x_from, y_sku), set()).add(y_lot_str)
        lot_strings_by_loc_sku.setdefault((y_from, x_sku), set()).add(x_lot_str)

    # moved_indices に両行の index を追加
    if moved_indices is not None:
        if hasattr(x_row, 'name') and x_row.name is not None:
            moved_indices.add(x_row.name)
        if hasattr(y_row, 'name') and y_row.name is not None:
            moved_indices.add(y_row.name)

    # consumed_y_rows に追加
    consumed_y_rows.add((y_from, y_sku))


def _pass_fifo_to_pick(
    inv: pd.DataFrame,
    shelf_usage: Dict[str, float],
    cap_limit: float,
    cfg: "OptimizerConfig",
    *,
    cap_by_loc: Optional[Dict[str, float]] = None,
    can_receive: Optional[Set[str]] = None,
    blocked_dest_locs: Optional[Set[str]] = None,
    skus_by_loc: Optional[Dict[str, Set[str]]] = None,
    lots_by_loc_sku: Optional[Dict[Tuple[str, str], Set[int]]] = None,
    lot_strings_by_loc_sku: Optional[Dict[Tuple[str, str], Set[str]]] = None,
    moved_indices: Optional[Set[int]] = None,
    sku_vol_map: Optional[pd.Series] = None,
    rep_pack_by_col: Optional[Dict[int, float]] = None,
    budget_left: Optional[int] = None,
    trace_id: Optional[str] = None,
) -> List[Move]:
    """最古ロットをLv1-2へ配置するFIFO整列パス。

    アルゴリズム:
      1. SKUをグローバル最古ロット昇順でソート
      2. 各SKUについて:
         a. 同列スワップ (Lv3-4古 ↔ Lv1-2新)
         b. 空きLv1-2スロット探索 (pack帯soft尊重, 同列優先)
         c. 自SKU列間スワップ (pack帯無視)
      3. 副作用でshelf_usage/skus_by_loc/lots_by_loc_sku/moved_indicesを更新
    """
    moves: List[Move] = []

    def _can_add(k: int) -> bool:
        return (budget_left is None) or (len(moves) + k <= budget_left)

    if inv.empty:
        logger.debug("[pass_fifo_to_pick] inv is empty")
        return moves

    # 内部追跡セット: moved_indices が None でもフェーズA処理済み行をフェーズBで除外する
    _effective_moved: Set[int] = moved_indices if moved_indices is not None else set()
    # フェーズC: 消費済み Y 行の追跡 (loc, sku) キー
    _consumed_y_rows: Set[Tuple[str, str]] = set()

    # 入力フィルタ
    filter_cond = (inv["volume_each_case"] > 0) & (inv["lot_key"] != UNKNOWN_LOT_KEY)
    if "is_movable" in inv.columns:
        filter_cond = filter_cond & (inv["is_movable"] == True)
    subset = inv[filter_cond].copy()
    if subset.empty:
        logger.debug("[pass_fifo_to_pick] subset is empty after filtering")
        return moves

    pick_levels = tuple(int(x) for x in getattr(cfg, "pick_levels", (1, 2)))
    storage_levels = tuple(int(x) for x in getattr(cfg, "storage_levels", (3, 4)))
    tolerance = float(getattr(cfg, "pack_tolerance_ratio", 0.10))

    # SKU毎の最古ロット日付で昇順ソート
    sku_groups = []
    for sku, df in subset.groupby("商品ID"):
        if len(df) <= 1:
            continue
        lot_keys = df["lot_key"].unique()
        if len(lot_keys) <= 1:
            continue  # 全行同一日付ならFIFO不要
        oldest = int(df["lot_key"].min())
        sku_groups.append((oldest, str(sku), df))

    sku_groups.sort(key=lambda x: x[0])  # 最古SKUから処理

    # グローバル空きスロット管理 (Lv1-2の空きロケ一覧をcol別に管理)
    empty_pick_slots_by_col: Dict[int, list] = defaultdict(list)
    for loc in shelf_usage.keys():
        if loc in PLACEHOLDER_LOCS:
            continue
        lv, col, dep = _parse_loc8(loc)
        if lv not in pick_levels:
            continue
        if can_receive is not None and loc not in can_receive:
            continue
        if blocked_dest_locs and loc in blocked_dest_locs:
            continue
        used = float(shelf_usage.get(loc, 0.0))
        limit = cap_by_loc.get(loc, cap_limit) if cap_by_loc else cap_limit
        if limit - used <= 0:
            continue
        if skus_by_loc:
            existing = skus_by_loc.get(loc, set())
            if len(existing) > 1:
                continue  # 複数SKU混在ロケは候補外
        empty_pick_slots_by_col[col].append((limit - used, loc))

    phase_a_count = 0
    phase_b_empty_count = 0
    phase_b_swap_count = 0
    phase_c_swap_count = 0
    skipped_no_slot = 0
    skipped_no_swap = 0

    for oldest_key, sku, sku_df in sku_groups:
        if not _can_add(1):
            break

        # 行をlot_key昇順で整列
        sku_sorted = sku_df.sort_values("lot_key", kind="mergesort")
        lv12_rows = sku_sorted[sku_sorted["lv"].isin(pick_levels)]
        lv34_rows = sku_sorted[sku_sorted["lv"].isin(storage_levels)]
        sku_cols = set(int(c) for c in sku_sorted["col"].unique())

        pre_move_count = len(moves)

        # フェーズA: 同列スワップ (SKU内完結)
        _handle_same_column_swaps(
            sku, sku_sorted,
            moves, shelf_usage, skus_by_loc, lots_by_loc_sku,
            lot_strings_by_loc_sku, _effective_moved, _can_add,
        )
        phase_a_count += len(moves) - pre_move_count

        # フェーズB: Lv3-4のままの最古K行を解消
        # K = 目標: 最古 K 行が Lv1-2 にあること。
        # 最低 1 行は確保 (Lv1-2 がゼロなら最古1行を移動)、
        # 既に Lv1-2 にあればその行数ぶん古い順に入替 (spec: (a) 案)
        K = max(1, len(lv12_rows))
        target_rows = sku_sorted.head(K)

        current_target_state = [
            (idx, r) for idx, r in target_rows.iterrows()
            if idx not in _effective_moved
            and int(r["lv"]) not in pick_levels
        ]

        for (row_idx, row) in current_target_state:
            if not _can_add(1):
                break

            # (b1) 空きスロット探索
            best_loc = _find_empty_pick_slot_soft_pack(
                sku=sku, row=row, sku_cols=sku_cols,
                empty_pick_slots_by_col=empty_pick_slots_by_col,
                rep_pack_by_col=rep_pack_by_col,
                tolerance=tolerance,
                shelf_usage=shelf_usage, cap_limit=cap_limit, cap_by_loc=cap_by_loc,
                skus_by_loc=skus_by_loc, lots_by_loc_sku=lots_by_loc_sku,
                lot_strings_by_loc_sku=lot_strings_by_loc_sku,
            )
            if best_loc is not None:
                _emit_move_fifo(
                    moves, row, to_loc=best_loc,
                    reason="FIFO: 最古→Lv1-2 (空きスロット)",
                    shelf_usage=shelf_usage, sku_vol_map=sku_vol_map,
                    skus_by_loc=skus_by_loc, lots_by_loc_sku=lots_by_loc_sku,
                    lot_strings_by_loc_sku=lot_strings_by_loc_sku,
                    moved_indices=_effective_moved,
                    empty_pick_slots_by_col=empty_pick_slots_by_col,
                    row_idx=row_idx,
                )
                phase_b_empty_count += 1
                continue

            # (b2) 自SKU列間スワップ
            if not _can_add(2):
                break

            # lv12_rowsを移動済みを除外して再取得
            lv12_current = sku_sorted[
                sku_sorted["lv"].isin(pick_levels)
            ]
            swap_target_row = _find_swap_target_cross_column(
                sku=sku, row=row, lv12_rows=lv12_current,
                moved_indices=_effective_moved,
                skus_by_loc=skus_by_loc,
            )
            if swap_target_row is not None:
                swap_idx = swap_target_row.name
                _emit_swap_pair_fifo(
                    moves,
                    old_row=row, new_row=swap_target_row,
                    old_idx=row_idx, new_idx=swap_idx,
                    reason_old="FIFO: 古ロット→Lv1-2 (列間スワップ)",
                    reason_new="FIFO: 新ロット→Lv3-4 (退避)",
                    shelf_usage=shelf_usage,
                    skus_by_loc=skus_by_loc, lots_by_loc_sku=lots_by_loc_sku,
                    lot_strings_by_loc_sku=lot_strings_by_loc_sku,
                    moved_indices=_effective_moved,
                )
                phase_b_swap_count += 1
            else:
                # フェーズC: クロスSKUスワップ (A/B-1/B-2 が全て失敗した後の最後の手段)
                if not _can_add(2):
                    break
                cross_result = _find_cross_sku_swap_target(
                    sku=sku, row=row,
                    passing_inv=subset,
                    pick_levels=pick_levels,
                    skus_by_loc=skus_by_loc,
                    moved_indices=_effective_moved,
                    consumed_y_rows=_consumed_y_rows,
                )
                if cross_result is not None:
                    y_loc, y_row = cross_result
                    _emit_cross_sku_swap(
                        moves, x_row=row, y_loc=y_loc, y_row=y_row,
                        shelf_usage=shelf_usage,
                        skus_by_loc=skus_by_loc,
                        lots_by_loc_sku=lots_by_loc_sku,
                        lot_strings_by_loc_sku=lot_strings_by_loc_sku,
                        moved_indices=_effective_moved,
                        consumed_y_rows=_consumed_y_rows,
                    )
                    phase_c_swap_count += 1
                else:
                    _record_drop("no_swap_target", {
                        "sku_id": sku,
                        "lot": str(row.get("ロット") or ""),
                        "from_loc": str(row.get("ロケーション") or ""),
                    })
                    skipped_no_swap += 1

    logger.debug(
        f"[pass_fifo_to_pick] phaseA={phase_a_count} phaseB_empty={phase_b_empty_count} "
        f"phaseB_swap={phase_b_swap_count} phaseC_cross={phase_c_swap_count} "
        f"skip_no_swap={skipped_no_swap} total_moves={len(moves)}"
    )
    _publish_progress(trace_id, {
        "phase": "pass_fifo_to_pick",
        "moves": len(moves),
        "phase_a": phase_a_count,
        "phase_b_empty": phase_b_empty_count,
        "phase_b_swap": phase_b_swap_count,
        "phase_c_cross": phase_c_swap_count,
    })
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

    # 移動距離ペナルティ（タイブレーカー: 同スコアなら近い移動を優先）
    try:
        _from_col = int(row.get("col") or 0)
        _from_dep = int(row.get("dep") or 0)
        _from_lv = int(row.get("lv") or 0)
        _move_dist = abs(tlv - _from_lv) + abs(tcol - _from_col) + abs(tdep - _from_dep)
        score += _move_dist * 0.1
    except Exception:
        pass

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
    blocked_dest_locs: Optional[Set[str]] = None,
    skus_by_loc: Optional[Dict[str, Set[str]]] = None,
    prefiltered_locs: Optional[List[str]] = None,
    first_fit: bool = False,
    depths_by_col: Optional[Dict[int, Tuple[int, float]]] = None,
) -> Optional[str]:
    """Choose a plausible destination location for the whole row.
    prefiltered_locs: 事前フィルタ済みロケーションリスト（空き容量順など）。
    指定時はこのリストのみスキャンし、shelf_usage全走査を回避。
    first_fit: True の場合、スコアが有限の最初の候補を即返す（eviction chain用高速モード）。
    depths_by_col: 呼び出し元で事前計算済みの列別最大奥行き情報。Noneの場合のみ内部で計算する。
    """
    from_loc = str(row["ロケーション"])
    best_to = None
    best_score = math.inf
    # 列ごとの最大奥行き本数と中心を事前計算（呼び出し元で渡されていない場合のみ）
    if depths_by_col is None:
        try:
            if cfg and getattr(cfg, "depth_preference", "front") == "center":
                depths_by_col = {}
                if not inv.empty and "col" in inv.columns and "dep" in inv.columns:
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
                        depths_by_col[c_int] = (max_dep, float(center))
        except Exception:
            depths_by_col = None

    _scan_locs = prefiltered_locs if prefiltered_locs is not None else list(shelf_usage.keys())
    for to_loc in _scan_locs:
        if to_loc in PLACEHOLDER_LOCS or to_loc == from_loc or to_loc in avoid_locs:
            continue
        if can_receive is not None and to_loc not in can_receive:
            continue
        if blocked_dest_locs and to_loc in blocked_dest_locs:
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
        # FIFO: 移動先列で同一SKUの他ロットとFIFO違反を起こさないかチェック
        # 列間移動でもFIFO違反を防止する（以前は同一列内のみチェックしていた）
        lot_key = int(row.get("lot_key") or UNKNOWN_LOT_KEY)
        if lot_key != UNKNOWN_LOT_KEY:
            inv_lot_levels = _GLOBAL_INV_INDEXES.get("inv_lot_levels_by_sku_col", {})
            if inv_lot_levels:
                if _violates_lot_level_rule_fast(str(row["商品ID"]), lot_key, tlv, tcol, inv_lot_levels):
                    continue
            else:
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
            s -= 200.0  # 空きロケ優先ボーナス（pack_cluster_penalty スケールに合わせて調整）
        # ゾーン適正ペナルティ: 移動先列がこの品の入数帯ゾーンに合わない場合ペナルティ
        try:
            _pack_val = row.get("pack_est")
            if _pack_val is not None and pd.notna(_pack_val):
                _pv = float(_pack_val)
                _large = set(range(1, 12)); _medium = set(range(12, 35)); _small = set(range(35, 42))
                if _pv >= 50:
                    _target_zone = _small
                elif _pv <= 12:
                    _target_zone = _large
                else:
                    _target_zone = _medium
                if tcol not in _target_zone:
                    s += 500.0  # ゾーン外ペナルティ（ゾーン内候補があればそちらを選ぶ）
        except Exception:
            pass
        # foreign SKU ゲート（別SKUがいる → 常にスキップ）
        # 退避は呼び出し元（Pass-0/Pass-2）で evict_foreign_skus 付き退避チェーンで処理する
        if skus_by_loc is not None:
            _orig_skus = skus_by_loc.get(to_loc, set())
            if _orig_skus and not _orig_skus <= {str(row["商品ID"])}:
                continue
        if s < best_score:
            best_score = s
            best_to = str(to_loc)
            if first_fit and best_score < math.inf:
                return best_to
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
    blocked_dest_locs: Optional[Set[str]] = None,
    skus_by_loc: Optional[Dict[str, Set[str]]] = None,
    lots_by_loc_sku: Optional[Dict[Tuple[str, str], Set[int]]] = None,
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
    
    # FIFO lot-level rule (O(1) dict参照)
    _inv_lot_levels = _GLOBAL_INV_INDEXES.get("inv_lot_levels_by_sku_col", {})
    if _inv_lot_levels and _violates_lot_level_rule_fast(sku_val, lot_key, target_level, tcol, _inv_lot_levels):
        return _CandidateEvalResult(
            to_loc=to_loc, target_level=target_level, tcol=tcol, tdep=tdep,
            score=math.inf, area_needs_mix=False, candidate_ev_chain=[],
            failure_reason="fifo"
        )

    # Hard rule: 移動先に別SKUが存在する場合は禁止（O(1) dict参照）
    if skus_by_loc is not None:
        _dest_skus = skus_by_loc.get(to_loc, set())
        if _dest_skus and not _dest_skus <= {str(sku_val)}:
            return _CandidateEvalResult(
                to_loc=to_loc, target_level=target_level, tcol=tcol, tdep=tdep,
                score=math.inf, area_needs_mix=False, candidate_ev_chain=[],
                failure_reason="foreign_sku"
            )

    # Hard rule: 同一SKUで異なるロットは同一ロケーションに置かない（O(1) dict参照）
    try:
        exists_mixed = False
        lookup_key = (str(to_loc), str(sku_val))

        # 1) Check existing inventory lots via pre-built dict
        if lots_by_loc_sku is not None:
            existing_lots = lots_by_loc_sku.get(lookup_key, set())
            if existing_lots:
                if int(lot_key) == UNKNOWN_LOT_KEY:
                    exists_mixed = True
                else:
                    _known = existing_lots - {UNKNOWN_LOT_KEY}
                    if _known and int(lot_key) not in _known:
                        exists_mixed = True

        # 2) Check already planned moves to this location with same SKU
        if not exists_mixed and lookup_key in planned_lots_by_loc_sku:
            planned_lots = planned_lots_by_loc_sku[lookup_key]
            if int(lot_key) == UNKNOWN_LOT_KEY:
                if planned_lots:
                    exists_mixed = True
            elif planned_lots and int(lot_key) not in planned_lots:
                exists_mixed = True

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
    # level volume cap: reject unconditionally (business rule)
    _cand_level_cap = _get_level_vol_cap(tlv, cfg)
    if _cand_level_cap is not None and used + need_vol > _cand_level_cap:
        return _CandidateEvalResult(
            to_loc=to_loc, target_level=target_level, tcol=tcol, tdep=tdep,
            score=math.inf, area_needs_mix=False, candidate_ev_chain=[],
            failure_reason="level_vol_cap"
        )
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
                can_receive=can_receive_set if can_receive_set is not None else None,
                hard_pack_A=hard_pack_A,
                ease_weight=getattr(cfg, "ease_weight", 0.0001),
                cfg=cfg,
                chain_group_id=eviction_chain_group_id,
                execution_order_start=1,
                blocked_dest_locs=blocked_dest_locs,
                skus_by_loc=skus_by_loc,
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
    evict_foreign_skus: Optional[str] = None,
    blocked_dest_locs: Optional[Set[str]] = None,
    skus_by_loc: Optional[Dict[str, Set[str]]] = None,
    prefiltered_locs: Optional[List[str]] = None,
    depths_by_col: Optional[Dict[int, Tuple[int, float]]] = None,
) -> Optional[List[Move]]:
    """Try to free `need_vol` capacity on `target_loc` by evicting whole rows.
    Returns a list of eviction moves in execution order if successful; otherwise None.
    This mutates `inv` and `shelf_usage` when it succeeds.
    If evict_foreign_skus is set (to the placing SKU id), also evict all rows
    belonging to OTHER SKUs at target_loc regardless of capacity.
    """
    # Quick check
    used = float(shelf_usage.get(target_loc, 0.0))
    limit = cap_by_loc.get(target_loc, cap_limit) if cap_by_loc else cap_limit
    free = limit - used
    if free >= need_vol and not evict_foreign_skus:
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
        # Fast path: use pre-built index, with KeyError fallback for stale/cross-tenant indexes
        try:
            in_rows = inv.loc[row_indices].copy()
        except KeyError:
            in_rows = inv[inv["ロケーション"].astype(str) == str(target_loc)].copy()
        # Ensure location filter matches (index may be stale after inv mutations)
        in_rows = in_rows[in_rows["ロケーション"].astype(str) == str(target_loc)]
    
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

    # evict_foreign_skus が指定されている場合、対象SKU以外の行のみ退避
    if evict_foreign_skus:
        in_rows_sorted = in_rows_sorted[in_rows_sorted["商品ID"].astype(str) != str(evict_foreign_skus)]
        if in_rows_sorted.empty:
            # 外部SKUなし → 容量チェックのみで判定
            if free >= need_vol:
                return []
            # 容量も足りない → 退避不可
            return None

    chain: List[Move] = []
    tried = 0
    max_rows = int(getattr(cfg, "max_chain_rows_per_target", 12))
    max_iterations = max(max_rows, 50)  # 安全上限
    iteration_count = 0
    for _, row in in_rows_sorted.iterrows():
        iteration_count += 1
        if iteration_count > max_iterations:
            logger.debug(f"[optimizer] _plan_eviction_chain: 最大反復回数({max_iterations})到達 - ループ中断")
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
            blocked_dest_locs=blocked_dest_locs,
            skus_by_loc=skus_by_loc,
            prefiltered_locs=prefiltered_locs,
            first_fit=True,
            depths_by_col=depths_by_col,
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
                blocked_dest_locs=blocked_dest_locs,
                skus_by_loc=skus_by_loc,
                prefiltered_locs=prefiltered_locs,
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

        # skus_by_locも更新（退避先にSKUを追加、元ロケから除去）
        # 注意: skus_by_locのdiscardはしない（部分退避で他の行が残る場合、
        # discardすると後続の移動が「空きロケ」と誤判断してSKU混在を作る）
        # SKUが完全に退去したかどうかはenforce_constraintsの_sim_remaining_qtyで正確に判定する
        if skus_by_loc is not None:
            skus_by_loc.setdefault(str(dest), set()).add(sku)

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
            if not evict_foreign_skus:
                return chain
            # evict_foreign_skus モード: 全外部SKU行を退避し終わるまで続ける
            # （ループ対象は外部SKUのみにフィルタ済みなので、ループ完走=全退避）

        # else continue trying more rows in target_loc

    # ループ完走: evict_foreign_skus なら外部SKUは全退避済み → 容量もチェック
    if evict_foreign_skus and chain:
        used_final = float(shelf_usage.get(target_loc, 0.0))
        limit_final = cap_by_loc.get(target_loc, cap_limit) if cap_by_loc else cap_limit
        if limit_final - used_final >= need_vol:
            return chain
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
    skus_by_loc: Optional[Dict[str, Set[str]]] = None,
    lots_by_loc_sku: Optional[Dict[Tuple[str, str], Set[int]]] = None,
    moved_indices: Optional[Set[int]] = None,
    blocked_dest_locs: Optional[Set[str]] = None,
    original_skus_by_loc: Optional[Dict[str, Set[str]]] = None,
    oldest_lot_by_sku: Optional[Dict[str, int]] = None,
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
    _cand_filter = (inv["volume_each_case"] > 0) & (inv["qty_cases_move"] > 0)
    if "is_movable" in inv.columns:
        _cand_filter = _cand_filter & (inv["is_movable"] == True)
    cand = inv[_cand_filter].copy()
    if cand.empty:
        return moves

    # 走査順：『帯外度』が高いものを先に（代表入数との差/ルール違反の明確さで近似）
    # ベクトル化実装: apply(axis=1) を避けてDataFrame全体に一括演算
    _small_cols_f = frozenset(getattr(cfg, "small_pack_cols", range(35, 42)))
    _medium_cols_f = frozenset(getattr(cfg, "medium_pack_cols", range(12, 35)))
    _large_cols_f = frozenset(getattr(cfg, "large_pack_cols", range(1, 12)))
    _low_max_f = int(getattr(cfg, "pack_low_max", 12))
    _high_min_f = int(getattr(cfg, "pack_high_min", 50))
    _promo_kw_f = tuple(getattr(cfg, "promo_quality_keywords", ("販促資材", "販促", "什器", "資材")))

    _qcol_f = "品質区分名" if "品質区分名" in cand.columns else ("quality_name" if "quality_name" in cand.columns else None)
    if _qcol_f:
        import re as _re
        _promo_pat_f = "|".join(_re.escape(k) for k in _promo_kw_f)
        _is_promo_vec = cand[_qcol_f].fillna("").str.contains(_promo_pat_f, regex=True, na=False)
    else:
        _is_promo_vec = pd.Series(False, index=cand.index)

    _pack_vec = pd.to_numeric(cand.get("pack_est", pd.Series(dtype=float, index=cand.index)), errors="coerce")
    _col_vec = cand["col"].astype(int)

    _is_large_zone = _is_promo_vec | (_pack_vec >= _high_min_f)
    _is_small_zone = (~_is_large_zone) & (_pack_vec <= _low_max_f)
    _is_medium_zone = ~_is_large_zone & ~_is_small_zone

    _in_allowed_vec = (
        (_is_large_zone & _col_vec.isin(_large_cols_f)) |
        (_is_small_zone & _col_vec.isin(_small_cols_f)) |
        (_is_medium_zone & _col_vec.isin(_medium_cols_f))
    )
    _rep_vec = _col_vec.map(rep_pack_by_col)
    _diff_vec = ((_pack_vec - _rep_vec).abs() / _rep_vec.replace(0, float("nan"))).fillna(0.0)
    cand["__misalign"] = (~_in_allowed_vec).astype(float) * (1.0 + _diff_vec)
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

    # ロケーション→SKUセットのインデックス（別SKU混在防止用）
    # 外部から渡されたマップがあればそれを使用（パス横断で整合性を維持）
    if skus_by_loc is not None:
        _skus_at_loc = skus_by_loc
    else:
        _skus_at_loc: Dict[str, Set[str]] = {}
        if "商品ID" in inv.columns and "ロケーション" in inv.columns:
            for _loc_v, _sku_v in zip(inv["ロケーション"].astype(str), inv["商品ID"].astype(str)):
                _skus_at_loc.setdefault(_loc_v, set()).add(_sku_v)
    # ロケーション×SKU別の行数カウンタ（O(1)残存チェック用）
    _sku_count_by_loc: Dict[Tuple[str, str], int] = {}
    if "商品ID" in inv.columns and "ロケーション" in inv.columns:
        for _cl, _cs in zip(inv["ロケーション"].astype(str), inv["商品ID"].astype(str)):
            _sku_count_by_loc[(_cl, _cs)] = _sku_count_by_loc.get((_cl, _cs), 0) + 1

    # 退避チェーン用: 空き容量上位ロケーションリスト（2765→200に絞り高速化）
    _bdl = blocked_dest_locs or set()
    _base_eligible = [
        loc for loc in shelf_usage.keys()
        if loc not in PLACEHOLDER_LOCS and loc not in _bdl
        and (can_receive is None or loc in can_receive)
    ]
    _max_prefilter = int(getattr(cfg, "eviction_dest_scan_limit", 200))
    _prefiltered_locs: List[str] = sorted(
        _base_eligible,
        key=lambda loc: (cap_by_loc.get(loc, cap_limit) if cap_by_loc else cap_limit) - shelf_usage.get(loc, 0.0),
        reverse=True
    )[:_max_prefilter]
    _prefilter_refresh_interval = 50  # N手ごとにリフレッシュ
    _p0_eviction_attempts = 0
    _p0_max_eviction_attempts = int(getattr(cfg, "pass0_max_eviction_attempts", 100))

    # 1件ずつ帯外を是正
    max_pass0_iterations = len(cand) * 2  # 候補数の2倍を上限
    _pass0_time_limit = float(getattr(cfg, "pass0_time_limit_sec", 1200))
    _pass0_start = time.perf_counter()
    iteration_count = 0
    for idx, row in cand.iterrows():
        # 二重移動防止
        if moved_indices is not None and idx in moved_indices:
            continue
        iteration_count += 1
        if iteration_count > max_pass0_iterations:
            logger.debug(f"[optimizer] pass0_area_rebalance: 最大反復回数({max_pass0_iterations})到達 - Pass-0完了")
            break
        # タイムリミット
        if iteration_count % 10 == 0:
            _elapsed = time.perf_counter() - _pass0_start
            if _elapsed > _pass0_time_limit:
                logger.debug(f"[optimizer] pass0_area_rebalance: タイムリミット({_pass0_time_limit:.0f}s)到達 iter={iteration_count} moves={len(moves)} - Pass-0完了")
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

            # 最古ロット保護: 取口(Lv1-2)にある最古ロットは上段に移動しない
            if oldest_lot_by_sku and cur_lv <= 2:
                _row_sku = str(row.get("商品ID", ""))
                _row_lot_key = int(row.get("lot_key") or UNKNOWN_LOT_KEY)
                if _row_lot_key != UNKNOWN_LOT_KEY:
                    _sku_oldest = oldest_lot_by_sku.get(_row_sku)
                    if _sku_oldest is not None and _row_lot_key == _sku_oldest:
                        continue  # 最古ロットを取口から動かさない

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
            _bdl = blocked_dest_locs or set()
            # 追加フィルタ（プレースホルダ・送元・受入可否・非対象品質ロケ除外）
            if from_loc:
                if can_receive is None:
                    eligible_locations = {loc for loc in eligible_locations if loc not in PLACEHOLDER_LOCS and loc != from_loc and loc not in _bdl}
                else:
                    eligible_locations = {loc for loc in eligible_locations if loc not in PLACEHOLDER_LOCS and loc != from_loc and loc in can_receive and loc not in _bdl}
            else:
                if can_receive is not None:
                    eligible_locations = {loc for loc in eligible_locations if loc in can_receive and loc not in PLACEHOLDER_LOCS and loc not in _bdl}
                else:
                    eligible_locations = {loc for loc in eligible_locations if loc not in PLACEHOLDER_LOCS and loc not in _bdl}

            best_choice: Optional[Tuple[str, int, int, int]] = None
            best_score = math.inf
            best_ev_chain: List[Move] = []
            best_ev_chain_group_id: Optional[str] = None

            # 候補ロケ:事前フィルタ済みのみをループ
            # パフォーマンス最適化: 候補が多すぎる場合は容量空き順でトップN件のみ評価
            eligible_count = len(eligible_locations)
            max_candidates_to_check = int(getattr(cfg, "pass0_max_candidates_per_item", 10))
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
                    logger.debug(f"[optimizer] pass0 item {iteration_count}/{len(cand)}: 候補ロケ探索中 {checked_count}/{eligible_count}")
                elif checked_count <= 10:
                    pass  # 最初の10件は静か

                tlv, tcol, tdep = _coords_map.get(str(to_loc), _parse_loc8(str(to_loc)))

                # パフォーマンス最適化：容量チェックを早期実行
                used = float(shelf_usage.get(to_loc, 0.0))
                limit = cap_by_loc.get(to_loc, cap_limit) if cap_by_loc else cap_limit
                needs_eviction = used + need_vol > limit

                # 別SKU混在防止: 移動先に自SKU以外が既にいたら退避が必要
                # _skus_at_loc（パス中の変異状態）のみで判定。
                # original_skus_by_loc との OR は、Pass-FIFO 移動済みSKUが復活し
                # false positive（不要な退避）を引き起こすため使わない。
                _sku_str = str(row["商品ID"])
                _existing_skus = _skus_at_loc.get(str(to_loc), set())
                _has_foreign_sku = bool(_existing_skus and not _existing_skus <= {_sku_str})
                if _has_foreign_sku:
                    needs_eviction = True

                # ロット混在防止: 同SKUでも異なるロットが既にいたら退避が必要
                if lots_by_loc_sku and not needs_eviction:
                    _p0_lot_key = int(row.get("lot_key") or UNKNOWN_LOT_KEY)
                    _p0_lsk = (str(to_loc), str(row["商品ID"]))
                    _p0_existing_lots = lots_by_loc_sku.get(_p0_lsk, set())
                    if _p0_existing_lots:
                        _p0_known = _p0_existing_lots - {UNKNOWN_LOT_KEY}
                        if _p0_known and _p0_lot_key not in _p0_known:
                            needs_eviction = True

                # チェーン許可がない場合は容量不足/SKU混在/ロット混在で除外
                if needs_eviction and not (getattr(cfg, "chain_depth", 0) and getattr(cfg, "eviction_budget", 0) and getattr(cfg, "touch_budget", 0)):
                    continue

                # FIFO：同一列内のみ厳密（別列へは制限なし）
                if int(tcol) == cur_col:
                    lot_key = int(row.get("lot_key") or UNKNOWN_LOT_KEY)
                    if lot_key != UNKNOWN_LOT_KEY:
                        _p0_lot_lvls = _GLOBAL_INV_INDEXES.get("inv_lot_levels_by_sku_col", {})
                        if _p0_lot_lvls and _violates_lot_level_rule_fast(str(row["商品ID"]), lot_key, tlv, tcol, _p0_lot_lvls):
                            continue

                candidate_ev_chain: List[Move] = []
                candidate_ev_chain_group_id: Optional[str] = None
                # 容量不足時のチェーン処理（試行上限あり）
                if needs_eviction:
                    _p0_eviction_attempts += 1
                    if _p0_eviction_attempts > _p0_max_eviction_attempts:
                        continue
                    # 退避チェーンは状態を変更するため、候補比較せず最初の成功で採用
                    # スナップショット保存（失敗時ロールバック用）
                    _snap_shelf = dict(shelf_usage)
                    _snap_skus = {k: set(v) for k, v in skus_by_loc.items()} if skus_by_loc is not None else None
                    budget = _ChainBudget(
                        depth_left=int(getattr(cfg, "chain_depth", 0)),
                        evictions_left=int(getattr(cfg, "eviction_budget", 0)),
                        touch_left=int(getattr(cfg, "touch_budget", 0)),
                        touched=set([from_loc, to_loc]),
                    )
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
                        evict_foreign_skus=str(row["商品ID"]) if _has_foreign_sku else None,
                        blocked_dest_locs=blocked_dest_locs,
                        skus_by_loc=skus_by_loc,
                        prefiltered_locs=_prefiltered_locs,
                    )
                    if chain is None:
                        # ロールバック
                        shelf_usage.clear()
                        shelf_usage.update(_snap_shelf)
                        if skus_by_loc is not None and _snap_skus is not None:
                            skus_by_loc.clear()
                            skus_by_loc.update(_snap_skus)
                        continue
                    candidate_ev_chain = chain
                    # _plan_eviction_chain が inv/shelf_usage を変更済みなので
                    # 他の候補ロケを比較できない → このロケを即採用
                    best_choice = (str(to_loc), tlv, tcol, tdep)
                    best_ev_chain = candidate_ev_chain
                    best_ev_chain_group_id = candidate_ev_chain_group_id
                    break

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
                    # _skus_at_loc / _sku_count_by_loc を更新（退避チェーン反映）
                    _skus_at_loc.setdefault(str(mv.to_loc), set()).add(str(mv.sku_id))
                    _ck_from = (str(mv.from_loc), str(mv.sku_id))
                    _ck_to = (str(mv.to_loc), str(mv.sku_id))
                    _sku_count_by_loc[_ck_from] = _sku_count_by_loc.get(_ck_from, 1) - 1
                    _sku_count_by_loc[_ck_to] = _sku_count_by_loc.get(_ck_to, 0) + 1
                    _from_set = _skus_at_loc.get(str(mv.from_loc))
                    if _from_set and _sku_count_by_loc.get(_ck_from, 0) <= 0:
                        _from_set.discard(str(mv.sku_id))
                        if not _from_set:
                            del _skus_at_loc[str(mv.from_loc)]
                    # lots_by_loc_sku 更新（退避チェーン反映）
                    if lots_by_loc_sku is not None:
                        _ev_lk = int(mv.lot_date or 0) if mv.lot_date else UNKNOWN_LOT_KEY
                        _ev_to_key = (str(mv.to_loc), str(mv.sku_id))
                        lots_by_loc_sku.setdefault(_ev_to_key, set()).add(_ev_lk)
                        _ev_from_key = (str(mv.from_loc), str(mv.sku_id))
                        if _ev_from_key in lots_by_loc_sku:
                            lots_by_loc_sku[_ev_from_key].discard(_ev_lk)
                            if not lots_by_loc_sku[_ev_from_key]:
                                del lots_by_loc_sku[_ev_from_key]
                    if moved_indices is not None and len(idxs) > 0:
                        moved_indices.add(ridx)

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
            # _skus_at_loc / _sku_count_by_loc を更新（本体移動反映）
            _sku_str = str(row["商品ID"])
            _skus_at_loc.setdefault(str(to_loc), set()).add(_sku_str)
            _ck_main_from = (from_loc, _sku_str)
            _ck_main_to = (str(to_loc), _sku_str)
            _sku_count_by_loc[_ck_main_from] = _sku_count_by_loc.get(_ck_main_from, 1) - 1
            _sku_count_by_loc[_ck_main_to] = _sku_count_by_loc.get(_ck_main_to, 0) + 1
            _from_set2 = _skus_at_loc.get(from_loc)
            if _from_set2 and _sku_count_by_loc.get(_ck_main_from, 0) <= 0:
                _from_set2.discard(_sku_str)
                if not _from_set2:
                    del _skus_at_loc[from_loc]
            # lots_by_loc_sku 更新（本体移動反映）
            if lots_by_loc_sku is not None:
                _p0_to_key = (str(to_loc), _sku_str)
                lots_by_loc_sku.setdefault(_p0_to_key, set()).add(int(lk))
                _p0_from_key = (from_loc, _sku_str)
                if _p0_from_key in lots_by_loc_sku:
                    lots_by_loc_sku[_p0_from_key].discard(int(lk))
                    if not lots_by_loc_sku[_p0_from_key]:
                        del lots_by_loc_sku[_p0_from_key]
            # FIFOインデックス更新（次の候補評価で正しい状態を参照するため）
            if lk != UNKNOWN_LOT_KEY:
                _p0_fifo_idx = _GLOBAL_INV_INDEXES.get("inv_lot_levels_by_sku_col")
                if _p0_fifo_idx is not None:
                    _p0_fk = (_sku_str, int(row.get("col",0) or 0))
                    if _p0_fk in _p0_fifo_idx:
                        _p0_target = (int(lk), int(row.get("lv",0) or 0))
                        for _pi2, _pv2 in enumerate(_p0_fifo_idx[_p0_fk]):
                            if _pv2 == _p0_target:
                                _p0_fifo_idx[_p0_fk].pop(_pi2)
                                break
                    _p0_tk = (_sku_str, int(tcol))
                    _p0_fifo_idx.setdefault(_p0_tk, []).append((int(lk), int(tlv)))
            if moved_indices is not None:
                moved_indices.add(idx)
            # 定期的に退避先候補リストをリフレッシュ（shelf_usage変化を反映）
            if len(moves) % _prefilter_refresh_interval == 0:
                _prefiltered_locs = sorted(
                    _base_eligible,
                    key=lambda loc: (cap_by_loc.get(loc, cap_limit) if cap_by_loc else cap_limit) - shelf_usage.get(loc, 0.0),
                    reverse=True
                )[:_max_prefilter]
        except Exception:
            continue

    if moves:
        try:
            logger.debug(f"[optimizer] pass0_area_rebalance moves={len(moves)}")
        except Exception:
            pass
    return moves


def _pass0_zone_swaps(
    inv: pd.DataFrame,
    shelf_usage: Dict[str, float],
    cap_limit: float,
    cfg: OptimizerConfig,
    rep_pack_by_col: Dict[int, float],
    sku_vol_map: pd.Series,
    *,
    cap_by_loc: Optional[Dict[str, float]] = None,
    can_receive: Optional[Set[str]] = None,
    budget_left: Optional[int] = None,
    skus_by_loc: Optional[Dict[str, Set[str]]] = None,
    lots_by_loc_sku: Optional[Dict[Tuple[str, str], Set[int]]] = None,
    moved_indices: Optional[Set[int]] = None,
    blocked_dest_locs: Optional[Set[str]] = None,
    oldest_lot_by_sku: Optional[Dict[str, int]] = None,
    original_skus_by_loc: Optional[Dict[str, Set[str]]] = None,
) -> List[Move]:
    """ゾーン違反アイテム同士をスワップして、空きロケなしでゾーン配置を修正する。

    Pass-0 area_rebalance では空きロケがないと修正できないが、
    far↔near / far↔mid / near↔mid の組み合わせで相互に違反しているアイテムを
    スワップすることで、空きロケなしに違反を解消できる。

    1スワップ = 2手 としてカウントし、budget_left を消費する。
    """
    # Note: enable_pass_zone_swap is checked by the caller; only check strict_pack_area here
    if not getattr(cfg, "strict_pack_area", True):
        return []

    moves: List[Move] = []

    def _can_add(k: int) -> bool:
        return (budget_left is None) or (len(moves) + int(k) <= int(budget_left))

    if inv.empty:
        return moves

    # ゾーン定義
    small_cols: Set[int] = set(getattr(cfg, "small_pack_cols", range(35, 42)))
    medium_cols: Set[int] = set(getattr(cfg, "medium_pack_cols", range(12, 35)))
    large_cols: Set[int] = set(getattr(cfg, "large_pack_cols", range(1, 12)))
    low_max = int(getattr(cfg, "pack_low_max", 12))
    high_min = int(getattr(cfg, "pack_high_min", 50))

    def _target_zone(row: pd.Series) -> str:
        """このアイテムが配置されるべきゾーン名を返す。"""
        allowed = _allowed_cols_for_row(row, cfg)
        if allowed <= large_cols:
            return "far"
        if allowed <= small_cols:
            return "near"
        return "mid"

    def _actual_zone(col: int) -> str:
        """現在の列番号からゾーン名を返す。"""
        if col in large_cols:
            return "far"
        if col in small_cols:
            return "near"
        if col in medium_cols:
            return "mid"
        return "unknown"

    # 対象抽出: volume > 0, qty > 0, 移動済みでない
    _zs_cand_filter = (
        (inv.get("volume_each_case", pd.Series(dtype=float)) > 0)
        & (inv.get("qty_cases_move", pd.Series(dtype=float)) > 0)
    )
    if "is_movable" in inv.columns:
        _zs_cand_filter = _zs_cand_filter & (inv["is_movable"] == True)
    cand = inv[_zs_cand_filter].copy()
    if cand.empty:
        return moves

    # pack_est<=2 は緩和対象外（_pass0_area_rebalance と同様）
    if getattr(cfg, "pass0_respect_smallpack_relax", True):
        try:
            pack_s = pd.to_numeric(cand.get("pack_est", pd.Series(dtype=float, index=cand.index)), errors="coerce")
            cand = cand[pack_s.fillna(0) > 2.0]
        except Exception:
            pass

    if cand.empty:
        return moves

    # ゾーン違反アイテムを収集
    # actual_zone != target_zone のものだけリスト化
    violation_list: List[Dict] = []
    for idx, row in cand.iterrows():
        if moved_indices is not None and idx in moved_indices:
            continue
        try:
            cur_col = int(row.get("col") or 0)
            if cur_col == 0:
                continue
            from_loc = str(row.get("ロケーション", ""))
            if not from_loc or from_loc in PLACEHOLDER_LOCS:
                continue
            az = _actual_zone(cur_col)
            tz = _target_zone(row)
            if az == tz or az == "unknown":
                continue
            vol = float(row.get("volume_each_case", 0.0) or 0.0) * float(row.get("qty_cases_move", 0) or 0)
            if vol <= 0:
                continue
            # 最古ロット保護: 取口(Lv1-2)にある最古ロットは上段に移動しない
            _item_sku = str(row.get("商品ID", ""))
            _item_lv = int(row.get("lv") or 0)
            _item_lot_key = int(row.get("lot_key") or UNKNOWN_LOT_KEY)
            if oldest_lot_by_sku and _item_lv <= 2 and _item_lot_key != UNKNOWN_LOT_KEY:
                _sku_oldest = oldest_lot_by_sku.get(_item_sku)
                if _sku_oldest is not None and _item_lot_key == _sku_oldest:
                    continue  # 最古ロットを取口から動かさない
            violation_list.append({
                "idx": idx,
                "row": row,
                "from_loc": from_loc,
                "actual_zone": az,
                "target_zone": tz,
                "vol": vol,
                "sku": _item_sku,
                "lot": str(row.get("ロット", "") or ""),
                "lot_key": _item_lot_key,
                "lot_date": _lot_key_to_datestr8(_item_lot_key),
                "qty": int(row.get("qty_cases_move") or 0),
            })
        except Exception:
            continue

    if not violation_list:
        return moves

    # ゾーンペアごとにグルーピング
    # key: (actual_zone, target_zone) -> list of violations
    zone_groups: Dict[Tuple[str, str], List[Dict]] = {}
    for v in violation_list:
        key = (v["actual_zone"], v["target_zone"])
        zone_groups.setdefault(key, []).append(v)

    # スワップ可能なゾーンペアを定義
    # (X, Y) と (Y, X) の組み合わせ（または (Y, X) と (X, *) でも相互是正可能）
    swap_zone_pairs = [
        ("far", "near"),
        ("far", "mid"),
        ("near", "mid"),
    ]

    # 使用済みロケーション（1スワップに1度しか参加できない）
    used_locs: Set[str] = set()
    _bdl = blocked_dest_locs or set()

    for zone_x, zone_y in swap_zone_pairs:
        if not _can_add(2):
            break

        # A: actual=X, target=Y（Yゾーンに行きたい）
        # B: actual=Y, target=X or target=MID acceptable（Xゾーンに行くとOK）
        group_ax_ty = zone_groups.get((zone_x, zone_y), [])
        # B は actual=Y かつ target=X のもの優先、なければ actual=Y かつ target≠Y のもの
        group_ay_tx = zone_groups.get((zone_y, zone_x), [])
        # AとBをペアリング（体積が近い順）
        if not group_ax_ty or not group_ay_tx:
            continue

        # 体積近い順でマッチング（グリーディ）
        # A を vol 昇順でソート、Bも vol 昇順でソートしてマッチ
        sorted_a = sorted(group_ax_ty, key=lambda v: v["vol"])
        sorted_b = sorted(group_ay_tx, key=lambda v: v["vol"])

        b_used_flags = [False] * len(sorted_b)

        for a in sorted_a:
            if not _can_add(2):
                break
            if a["from_loc"] in used_locs:
                continue
            if moved_indices is not None and a["idx"] in moved_indices:
                continue

            best_b_idx: Optional[int] = None
            best_vol_diff = math.inf

            for bi, b in enumerate(sorted_b):
                if b_used_flags[bi]:
                    continue
                if b["from_loc"] in used_locs or b["from_loc"] == a["from_loc"]:
                    continue
                if moved_indices is not None and b["idx"] in moved_indices:
                    continue

                # --- ゾーン適正チェック: 交換後にゾーン違反が増えないか ---
                a_loc = a["from_loc"]
                b_loc = b["from_loc"]
                _a_from_col = int(str(a_loc).zfill(8)[3:6])
                _b_from_col = int(str(b_loc).zfill(8)[3:6])
                _a_dest_col = _b_from_col  # AはBのロケに行く
                _b_dest_col = _a_from_col  # BはAのロケに行く
                _a_target = a["target_zone"]
                _b_target = b["target_zone"]
                # 交換前のゾーン違反数
                _before_violations = 0
                if _actual_zone(_a_from_col) != _a_target: _before_violations += 1
                if _actual_zone(_b_from_col) != _b_target: _before_violations += 1
                # 交換後のゾーン違反数
                _after_violations = 0
                if _actual_zone(_a_dest_col) != _a_target: _after_violations += 1
                if _actual_zone(_b_dest_col) != _b_target: _after_violations += 1
                # 違反が増えるならスキップ（改善or現状維持のみ許可）
                if _after_violations > _before_violations:
                    continue

                # --- 実行可能性チェック ---
                # a_loc, b_loc は上のゾーン適正チェックで定義済み

                # blocked / can_receive チェック
                if a_loc in _bdl or b_loc in _bdl:
                    continue
                if can_receive is not None:
                    if a_loc not in can_receive or b_loc not in can_receive:
                        continue

                # 容量チェック: A の vol が B のロケに入るか
                a_vol = a["vol"]
                b_vol = b["vol"]
                b_loc_limit = cap_by_loc.get(b_loc, cap_limit) if cap_by_loc else cap_limit
                b_loc_used = float(shelf_usage.get(b_loc, 0.0))
                # スワップなので B が出てから A が入る: 空き = limit - (used - b_vol)
                b_space_after = b_loc_limit - (b_loc_used - b_vol)
                if a_vol > b_space_after + 1e-9:
                    continue

                a_loc_limit = cap_by_loc.get(a_loc, cap_limit) if cap_by_loc else cap_limit
                a_loc_used = float(shelf_usage.get(a_loc, 0.0))
                a_space_after = a_loc_limit - (a_loc_used - a_vol)
                if b_vol > a_space_after + 1e-9:
                    continue

                # ロット混在チェック（B が A のロケに入る / A が B のロケに入る）
                if lots_by_loc_sku is not None:
                    # A が B.from_loc に入る: B.from_loc に A.sku の他ロットがいないか
                    _bl_sk = (b_loc, a["sku"])
                    _existing_lots_bl = lots_by_loc_sku.get(_bl_sk, set())
                    if _existing_lots_bl:
                        _known = _existing_lots_bl - {UNKNOWN_LOT_KEY}
                        if _known and a["lot_key"] not in _known:
                            continue

                    # B が A.from_loc に入る: A.from_loc に B.sku の他ロットがいないか
                    _al_sk = (a_loc, b["sku"])
                    _existing_lots_al = lots_by_loc_sku.get(_al_sk, set())
                    if _existing_lots_al:
                        _known2 = _existing_lots_al - {UNKNOWN_LOT_KEY}
                        if _known2 and b["lot_key"] not in _known2:
                            continue

                # FIFOチェック: スワップ先の列で同一SKUの他ロットとFIFO違反を起こさないか
                # _violates_lot_level_rule_fast を使用（グローバルインデックス参照）
                _fifo_ok = True
                _zs_lot_lvl_idx = _GLOBAL_INV_INDEXES.get("inv_lot_levels_by_sku_col", {})
                for _item, _dest_loc in [(a, b_loc), (b, a_loc)]:
                    _item_lot_key = _item["lot_key"]
                    if _item_lot_key == UNKNOWN_LOT_KEY:
                        continue
                    _dest_lv, _dest_col, _ = _parse_loc8(_dest_loc)
                    if _zs_lot_lvl_idx and _violates_lot_level_rule_fast(
                        _item["sku"], _item_lot_key, _dest_lv, _dest_col, _zs_lot_lvl_idx
                    ):
                        _fifo_ok = False
                        break
                if not _fifo_ok:
                    continue

                # 1SKU/ロケチェック: スワップなのでAが出てBが入る→元のロケにAのSKUしかいないはず
                # A.from_loc: A が出た後に B.sku が入るが、A.sku 以外がいる場合は混在
                # 1SKU/ロケチェック: スワップ後に各ロケに1SKUだけか確認
                # enforce_constraintsと同じ基準を使うため、original_skus_by_loc（品質フィルタ前）も参照
                _check_skus = original_skus_by_loc if original_skus_by_loc is not None else skus_by_loc
                if _check_skus is not None:
                    _a_skus = _check_skus.get(a_loc, set())
                    if _a_skus and not _a_skus <= {a["sku"]}:
                        continue  # A のロケに別SKUが混在（品質フィルタ前ベース）
                    _b_skus = _check_skus.get(b_loc, set())
                    if _b_skus and not _b_skus <= {b["sku"]}:
                        continue  # B のロケに別SKUが混在（品質フィルタ前ベース）

                vol_diff = abs(a_vol - b_vol)
                if vol_diff < best_vol_diff:
                    best_vol_diff = vol_diff
                    best_b_idx = bi

            if best_b_idx is None:
                continue

            b = sorted_b[best_b_idx]
            b_used_flags[best_b_idx] = True
            a_loc = a["from_loc"]
            b_loc = b["from_loc"]
            a_vol = a["vol"]
            b_vol = b["vol"]

            # Move 生成
            swap_id = f"zone_swap_{secrets.token_hex(6)}"
            a_lv, a_col, a_dep = _parse_loc8(a_loc)
            b_lv, b_col, b_dep = _parse_loc8(b_loc)
            # 入数情報を取得
            _a_pack = int(a["row"].get("pack_est", 0) or 0)
            _b_pack = int(b["row"].get("pack_est", 0) or 0)
            # ゾーン名を日本語化
            _zone_names = {"far": "遠方(重量品)", "mid": "中間", "near": "近接(軽量品)"}
            _a_target_name = _zone_names.get(a["target_zone"], a["target_zone"])
            _b_target_name = _zone_names.get(b["target_zone"], b["target_zone"])
            reason_a = (
                f"ゾーンスワップ: 入数{_a_pack}は{_a_target_name}ゾーンが適正 → "
                f"列{a_col}→{b_col}へ移動。"
                f"交換相手{b['sku']}(入数{_b_pack})は列{b_col}→{a_col}({_b_target_name}ゾーン)へ"
            )
            reason_b = (
                f"ゾーンスワップ: 入数{_b_pack}は{_b_target_name}ゾーンが適正 → "
                f"列{b_col}→{a_col}へ移動。"
                f"交換相手{a['sku']}(入数{_a_pack})は列{a_col}→{b_col}({_a_target_name}ゾーン)へ"
            )

            mv_a = Move(
                sku_id=a["sku"],
                lot=a["lot"],
                qty=a["qty"],
                from_loc=a_loc.zfill(8),
                to_loc=b_loc.zfill(8),
                lot_date=a["lot_date"],
                reason=reason_a,
                chain_group_id=swap_id,
                execution_order=1,
            )
            mv_b = Move(
                sku_id=b["sku"],
                lot=b["lot"],
                qty=b["qty"],
                from_loc=b_loc.zfill(8),
                to_loc=a_loc.zfill(8),
                lot_date=b["lot_date"],
                reason=reason_b,
                chain_group_id=swap_id,
                execution_order=2,
            )
            moves.append(mv_a)
            moves.append(mv_b)

            # 状態更新
            # shelf_usage: A出 → B入 at B_loc, B出 → A入 at A_loc
            shelf_usage[a_loc] = max(0.0, shelf_usage.get(a_loc, 0.0) - a_vol + b_vol)
            shelf_usage[b_loc] = max(0.0, shelf_usage.get(b_loc, 0.0) - b_vol + a_vol)

            # inv DataFrame のロケーション更新
            try:
                inv.at[a["idx"], "ロケーション"] = b_loc.zfill(8)
                inv.at[a["idx"], "lv"] = b_lv
                inv.at[a["idx"], "col"] = b_col
                inv.at[a["idx"], "dep"] = b_dep
            except Exception:
                pass
            try:
                inv.at[b["idx"], "ロケーション"] = a_loc.zfill(8)
                inv.at[b["idx"], "lv"] = a_lv
                inv.at[b["idx"], "col"] = a_col
                inv.at[b["idx"], "dep"] = a_dep
            except Exception:
                pass

            # skus_by_loc 更新（スワップなので単純入れ替え）
            if skus_by_loc is not None:
                _a_skus = skus_by_loc.get(a_loc, set())
                _b_skus = skus_by_loc.get(b_loc, set())
                _a_skus.discard(a["sku"])
                _a_skus.add(b["sku"])
                _b_skus.discard(b["sku"])
                _b_skus.add(a["sku"])
                if _a_skus:
                    skus_by_loc[a_loc] = _a_skus
                elif a_loc in skus_by_loc:
                    del skus_by_loc[a_loc]
                if _b_skus:
                    skus_by_loc[b_loc] = _b_skus
                elif b_loc in skus_by_loc:
                    del skus_by_loc[b_loc]

            # lots_by_loc_sku 更新
            if lots_by_loc_sku is not None:
                # A が B_loc へ
                _bl_a_key = (b_loc, a["sku"])
                lots_by_loc_sku.setdefault(_bl_a_key, set()).add(a["lot_key"])
                _al_a_key = (a_loc, a["sku"])
                if _al_a_key in lots_by_loc_sku:
                    lots_by_loc_sku[_al_a_key].discard(a["lot_key"])
                    if not lots_by_loc_sku[_al_a_key]:
                        del lots_by_loc_sku[_al_a_key]
                # B が A_loc へ
                _al_b_key = (a_loc, b["sku"])
                lots_by_loc_sku.setdefault(_al_b_key, set()).add(b["lot_key"])
                _bl_b_key = (b_loc, b["sku"])
                if _bl_b_key in lots_by_loc_sku:
                    lots_by_loc_sku[_bl_b_key].discard(b["lot_key"])
                    if not lots_by_loc_sku[_bl_b_key]:
                        del lots_by_loc_sku[_bl_b_key]

            # FIFOインデックス更新: 次のスワップのFIFOチェックが正しい状態を参照するため
            _zs_fifo_idx = _GLOBAL_INV_INDEXES.get("inv_lot_levels_by_sku_col")
            if _zs_fifo_idx is not None:
                # A: (sku_a, col_a) から除去, (sku_a, col_b) に追加
                _a_from_key = (a["sku"], a_col)
                if _a_from_key in _zs_fifo_idx:
                    _target_a = (a["lot_key"], a_lv)
                    for _zi, _zv in enumerate(_zs_fifo_idx[_a_from_key]):
                        if _zv == _target_a:
                            _zs_fifo_idx[_a_from_key].pop(_zi)
                            break
                _a_to_key = (a["sku"], b_col)
                _zs_fifo_idx.setdefault(_a_to_key, []).append((a["lot_key"], b_lv))
                # B: (sku_b, col_b) から除去, (sku_b, col_a) に追加
                _b_from_key = (b["sku"], b_col)
                if _b_from_key in _zs_fifo_idx:
                    _target_b = (b["lot_key"], b_lv)
                    for _zi, _zv in enumerate(_zs_fifo_idx[_b_from_key]):
                        if _zv == _target_b:
                            _zs_fifo_idx[_b_from_key].pop(_zi)
                            break
                _b_to_key = (b["sku"], a_col)
                _zs_fifo_idx.setdefault(_b_to_key, []).append((b["lot_key"], a_lv))

            # moved_indices 更新
            if moved_indices is not None:
                moved_indices.add(a["idx"])
                moved_indices.add(b["idx"])

            # used_locs 更新
            used_locs.add(a_loc)
            used_locs.add(b_loc)

    logger.warning(f"[optimizer] pass0_zone_swaps: {len(moves) // 2} スワップ生成 ({len(moves)} 手)")
    return moves


# ---------------------------------------------------------------------------
# Final regression gate: 移動前より悪化しない移動セットを返す
# ---------------------------------------------------------------------------

def _count_fifo_violations(
    inv_lot_levels: Dict[Tuple[str, int], list],
) -> int:
    """FIFO違反数をカウント（同一列内で古ロットが高段にある (sku,col) の数）。
    ソート後の隣接比較で、各(sku,col)につき最大1件。"""
    violations = 0
    for (_sku, _col), entries in inv_lot_levels.items():
        if len(entries) < 2:
            continue
        sorted_entries = sorted(entries, key=lambda x: x[1])  # lv昇順
        for i in range(len(sorted_entries) - 1):
            lk_i, lv_i = sorted_entries[i]
            lk_j, lv_j = sorted_entries[i + 1]
            if lk_i == lk_j:
                continue
            # 下段に新しいロット、上段に古いロット → 違反
            if lk_i > lk_j:
                violations += 1
                break  # (sku,col)あたり1件
    return violations


def _count_lot_mixing(
    moves: list,
    original_lot_strings_by_loc_sku: Dict[Tuple[str, str], Set[str]],
) -> int:
    """移動適用後にロット混在が発生するロケ数をカウント。"""
    sim_lots: Dict[Tuple[str, str], Set[str]] = {k: set(v) for k, v in original_lot_strings_by_loc_sku.items()}
    for m in moves:
        sku = str(m.sku_id)
        to_loc = str(m.to_loc).zfill(8)
        from_loc = str(m.from_loc).zfill(8)
        lot = str(m.lot) if m.lot else ""
        if lot in ("nan", "None", ""):
            continue
        sim_lots.setdefault((to_loc, sku), set()).add(lot)
        fk = (from_loc, sku)
        if fk in sim_lots:
            sim_lots[fk].discard(lot)
    mixing = 0
    for _k, lots in sim_lots.items():
        if len(lots) > 1:
            mixing += 1
    return mixing


def _count_sku_mixing(
    moves: list,
    original_skus_by_loc: Dict[str, Set[str]],
    original_qty_by_loc_sku: Dict[Tuple[str, str], float],
    log_details: bool = False,
) -> int:
    """移動適用後にSKU混在が発生するロケ数をカウント。

    スワップペア（同一 chain_group_id を持つ2移動）はアトミックに適用する。
    逐次処理すると中間状態でSKU混在が誤検出されるため。
    """
    sim_skus: Dict[str, Set[str]] = {k: set(v) for k, v in original_skus_by_loc.items()}
    sim_qty: Dict[Tuple[str, str], float] = dict(original_qty_by_loc_sku)
    _new_mix_count = 0

    # スワップペアをアトミック処理するためグループ化
    # chain_group_id が swap_/zone_swap_/swap_fifo_ で始まるものはペアとして扱う
    _swap_prefixes = ("swap_", "zone_swap_", "fifo_direct_")
    _cg_groups: Dict[str, list] = {}
    _ordered_keys: list = []
    for m in moves:
        cg = getattr(m, 'chain_group_id', None) or ""
        is_swap_cg = any(cg.startswith(p) for p in _swap_prefixes)
        if is_swap_cg and cg:
            if cg not in _cg_groups:
                _cg_groups[cg] = []
                _ordered_keys.append(("swap", cg))
            _cg_groups[cg].append(m)
        else:
            _ordered_keys.append(("single", m))

    def _apply_single(m, mi):
        nonlocal _new_mix_count
        sku = str(m.sku_id)
        to_loc = str(m.to_loc).zfill(8)
        from_loc = str(m.from_loc).zfill(8)
        qty = float(m.qty)
        existing = sim_skus.get(to_loc, set())
        if existing and not existing <= {sku}:
            if log_details and _new_mix_count < 5:
                _from_qty = sim_qty.get((from_loc, sku), 0)
                _dest_qtys = {s: sim_qty.get((to_loc, s), 0) for s in existing - {sku}}
                logger.warning(f"[sku_mix_detail] #{mi} {sku}->{to_loc} existing={existing-{sku}} dest_qty={_dest_qtys} from_qty={_from_qty} move_qty={qty}")
            _new_mix_count += 1
        sim_skus.setdefault(to_loc, set()).add(sku)
        sim_qty[(to_loc, sku)] = sim_qty.get((to_loc, sku), 0.0) + qty
        sim_qty[(from_loc, sku)] = sim_qty.get((from_loc, sku), 0.0) - qty
        if sim_qty.get((from_loc, sku), 0.0) <= 0:
            fs = sim_skus.get(from_loc)
            if fs:
                fs.discard(sku)
                if not fs:
                    del sim_skus[from_loc]

    mi = 0
    seen_swap_cgs: set = set()
    for key in _ordered_keys:
        if key[0] == "single":
            _apply_single(key[1], mi)
            mi += 1
        else:
            cg = key[1]
            if cg in seen_swap_cgs:
                continue
            seen_swap_cgs.add(cg)
            swap_moves = _cg_groups[cg]
            if len(swap_moves) == 2:
                # アトミック適用: まず両方の from から除去、次に両方の to へ追加
                sm1, sm2 = swap_moves[0], swap_moves[1]
                for sm in (sm1, sm2):
                    s = str(sm.sku_id)
                    fl = str(sm.from_loc).zfill(8)
                    q = float(sm.qty)
                    sim_qty[(fl, s)] = sim_qty.get((fl, s), 0.0) - q
                    if sim_qty.get((fl, s), 0.0) <= 0:
                        fs = sim_skus.get(fl)
                        if fs:
                            fs.discard(s)
                            if not fs:
                                del sim_skus[fl]
                for sm in (sm1, sm2):
                    s = str(sm.sku_id)
                    tl = str(sm.to_loc).zfill(8)
                    q = float(sm.qty)
                    sim_skus.setdefault(tl, set()).add(s)
                    sim_qty[(tl, s)] = sim_qty.get((tl, s), 0.0) + q
                mi += 2
            else:
                # ペア以外（1つまたは3つ以上）は逐次処理
                for sm in swap_moves:
                    _apply_single(sm, mi)
                    mi += 1

    result = sum(1 for v in sim_skus.values() if len(v) > 1)
    if log_details:
        logger.warning(f"[sku_mix_detail] total new mixing moves: {_new_mix_count}, final mix locs: {result}")
    return result


def _apply_moves_to_lot_levels(
    base: Dict[Tuple[str, int], list],
    moves: list,
) -> Dict[Tuple[str, int], list]:
    """移動リストを適用した後の inv_lot_levels を返す。"""
    result: Dict[Tuple[str, int], list] = {k: list(v) for k, v in base.items()}
    for m in moves:
        lk = _parse_lot_date_key(m.lot)
        if lk == UNKNOWN_LOT_KEY:
            continue
        f_lv, f_col, _ = _parse_loc8(str(m.from_loc).zfill(8))
        t_lv, t_col, _ = _parse_loc8(str(m.to_loc).zfill(8))
        sku = str(m.sku_id)
        # from側から除去
        fk = (sku, f_col)
        if fk in result:
            for idx, (ek, elv) in enumerate(result[fk]):
                if ek == lk and elv == f_lv:
                    result[fk].pop(idx)
                    break
        # to側に追加
        result.setdefault((sku, t_col), []).append((lk, t_lv))
    return result


def _final_regression_gate(
    accepted: list,
    original_inv_lot_levels: Dict[Tuple[str, int], list],
    original_lot_strings_by_loc_sku: Dict[Tuple[str, str], Set[str]],
    *,
    max_rounds: int = 3,
    original_skus_by_loc: Optional[Dict[str, Set[str]]] = None,
    original_qty_by_loc_sku: Optional[Dict[Tuple[str, str], float]] = None,
) -> list:
    """移動前より FIFO 違反数・ロット混在数・SKU混在数が悪化しない移動セットを返す。

    1. 初期状態の FIFO 違反数・ロット混在数・SKU混在数を計算
    2. 全移動適用後に悪化しているか確認
    3. 悪化する場合、移動を逆順に走査し悪化原因の移動をチェーン単位で除去
    4. 除去後に再チェック（最大 max_rounds 回反復）
    """
    if not accepted:
        return accepted

    baseline_fifo = _count_fifo_violations(original_inv_lot_levels)
    baseline_mix = _count_lot_mixing([], original_lot_strings_by_loc_sku)
    baseline_sku_mix = _count_sku_mixing([], original_skus_by_loc or {}, original_qty_by_loc_sku or {}) if original_skus_by_loc else 0
    _total_entries = sum(len(v) for v in original_inv_lot_levels.values())
    _total_groups = len(original_inv_lot_levels)
    logger.warning(f"[regression-gate] baseline: FIFO={baseline_fifo}, mix={baseline_mix}, sku_mix={baseline_sku_mix}, entries={_total_entries}, groups={_total_groups}")
    # FIFO baseline と外部検証で数件の差が生じるため、わずかなFIFO増加は許容する
    # ただしロット混在は厳密に悪化を許容しない
    _fifo_tolerance = 0  # FIFO悪化を許容しない

    for _round in range(max_rounds):
        # 逐次適用方式: 移動を1件ずつ適用し、FIFO違反やロット混在やSKU混在を作る移動をマーク
        _scl = {k: list(v) for k, v in original_inv_lot_levels.items()}
        _lot_str_sim: Dict[Tuple[str, str], Set[str]] = {}
        if original_lot_strings_by_loc_sku:
            _lot_str_sim = {k: set(v) for k, v in original_lot_strings_by_loc_sku.items()}
        # SKU混在チェック用の逐次シミュレーション
        _sku_sim: Dict[str, Set[str]] = {k: set(v) for k, v in (original_skus_by_loc or {}).items()}
        _sku_qty_sim: Dict[Tuple[str, str], float] = dict(original_qty_by_loc_sku or {})

        _bad_chains: Set[str] = set()
        _bad_indices: Set[int] = set()

        for idx, a in enumerate(accepted):
            _a_sku = str(a.sku_id)
            _a_lk = _parse_lot_date_key(a.lot)
            _a_flv, _a_fcol, _ = _parse_loc8(str(a.from_loc).zfill(8))
            _a_tlv, _a_tcol, _ = _parse_loc8(str(a.to_loc).zfill(8))
            _a_lot_str = str(a.lot) if a.lot else ""
            _a_to_loc = str(a.to_loc).zfill(8)
            _a_from_loc = str(a.from_loc).zfill(8)

            is_bad = False
            _a_cg = getattr(a, 'chain_group_id', None) or ""
            _is_swap = (
                _a_cg.startswith("swap_")
                or _a_cg.startswith("fifo_direct_")
                or _a_cg.startswith("zone_swap_")
            )

            # スワップは逐次チェックをスキップ（相手が同時退出するため誤検出）
            # 最終状態は _count_sku_mixing / _count_lot_mixing / _count_fifo_violations で確認する
            if not _is_swap:
                # FIFOチェック: この移動が移動先列でFIFO違反を作るか
                if _a_lk != UNKNOWN_LOT_KEY:
                    _entries = _scl.get((_a_sku, _a_tcol), [])
                    for _ek, _elv in _entries:
                        if _ek == _a_lk:
                            continue
                        if (_a_lk < _ek and _a_tlv > _elv) or (_a_lk > _ek and _a_tlv < _elv):
                            is_bad = True
                            break

                # ロット混在チェック: この移動が移動先で新たなロット混在を作るか
                if not is_bad and _a_lot_str and _a_lot_str != "nan":
                    _lsk = (_a_to_loc, _a_sku)
                    _existing_lots = _lot_str_sim.get(_lsk, set())
                    if _existing_lots and _a_lot_str not in _existing_lots:
                        is_bad = True

                # SKU混在チェック: この移動が移動先で新たなSKU混在を作るか
                if not is_bad and original_skus_by_loc is not None:
                    _existing_skus = _sku_sim.get(_a_to_loc, set())
                    if _existing_skus and not _existing_skus <= {_a_sku}:
                        is_bad = True

            if is_bad:
                cg = getattr(a, 'chain_group_id', None)
                if cg:
                    _bad_chains.add(cg)
                else:
                    _bad_indices.add(idx)

            # 状態更新（badかどうかに関わらず、後続チェックのために適用）
            if _a_lk != UNKNOWN_LOT_KEY:
                _fk = (_a_sku, _a_fcol)
                if _fk in _scl:
                    for _ei, _ev in enumerate(_scl[_fk]):
                        if _ev == (_a_lk, _a_flv):
                            _scl[_fk].pop(_ei)
                            break
                _scl.setdefault((_a_sku, _a_tcol), []).append((_a_lk, _a_tlv))

            if _a_lot_str and _a_lot_str != "nan":
                _to_lsk = (_a_to_loc, _a_sku)
                _lot_str_sim.setdefault(_to_lsk, set()).add(_a_lot_str)
                _from_lsk = (_a_from_loc, _a_sku)
                if _from_lsk in _lot_str_sim:
                    _lot_str_sim[_from_lsk].discard(_a_lot_str)

            # SKU逐次更新
            _sku_sim.setdefault(_a_to_loc, set()).add(_a_sku)
            _sku_qty_sim[(_a_to_loc, _a_sku)] = _sku_qty_sim.get((_a_to_loc, _a_sku), 0.0) + float(a.qty)
            _sku_qty_sim[(_a_from_loc, _a_sku)] = _sku_qty_sim.get((_a_from_loc, _a_sku), 0.0) - float(a.qty)
            if _sku_qty_sim.get((_a_from_loc, _a_sku), 0.0) <= 0:
                _fs = _sku_sim.get(_a_from_loc)
                if _fs:
                    _fs.discard(_a_sku)
                    if not _fs:
                        del _sku_sim[_a_from_loc]

        if not _bad_chains and not _bad_indices:
            # 逐次チェックでは違反なし → 最終状態で確認
            post_lot_levels = _apply_moves_to_lot_levels(original_inv_lot_levels, accepted)
            post_fifo = _count_fifo_violations(post_lot_levels)
            post_mix = _count_lot_mixing(accepted, original_lot_strings_by_loc_sku)
            post_sku_mix = _count_sku_mixing(accepted, original_skus_by_loc or {}, original_qty_by_loc_sku or {}, log_details=True) if original_skus_by_loc else 0

            if post_fifo <= baseline_fifo and post_mix <= baseline_mix and post_sku_mix <= baseline_sku_mix:
                logger.warning(
                    f"[regression-gate] round {_round+1}: OK "
                    f"(FIFO {baseline_fifo}->{post_fifo}, mix {baseline_mix}->{post_mix}, sku_mix {baseline_sku_mix}->{post_sku_mix})"
                )
                break

            # 逐次では検出できないが最終状態で悪化 → 貪欲法で追加除去
            logger.warning(
                f"[regression-gate] round {_round+1}: sequential OK but final state regressed "
                f"(FIFO {baseline_fifo}->{post_fifo}, mix {baseline_mix}->{post_mix}, sku_mix {baseline_sku_mix}->{post_sku_mix}), "
                f"switching to greedy removal"
            )
            # 悪影響の大きいチェーンを1つずつ除去
            _checked = set()
            while post_fifo > baseline_fifo or post_mix > baseline_mix or post_sku_mix > baseline_sku_mix:
                best_cg = None
                best_improvement = 0
                best_fifo = post_fifo
                best_mix = post_mix
                best_sku_mix = post_sku_mix
                for a in accepted:
                    cg = getattr(a, 'chain_group_id', None) or ""
                    if cg and cg in _checked: continue
                    if cg: _checked.add(cg)
                    trial = [x for x in accepted if (getattr(x,'chain_group_id',None) or "") != cg] if cg else [x for x in accepted if x is not a]
                    t_ll = _apply_moves_to_lot_levels(original_inv_lot_levels, trial)
                    t_f = _count_fifo_violations(t_ll)
                    t_m = _count_lot_mixing(trial, original_lot_strings_by_loc_sku)
                    t_sm = _count_sku_mixing(trial, original_skus_by_loc or {}, original_qty_by_loc_sku or {}) if original_skus_by_loc else 0
                    imp = (post_fifo-t_f if post_fifo>baseline_fifo else 0) + (post_mix-t_m if post_mix>baseline_mix else 0) + (post_sku_mix-t_sm if post_sku_mix>baseline_sku_mix else 0)
                    if imp > best_improvement:
                        best_improvement = imp
                        best_cg = cg
                        best_fifo = t_f
                        best_mix = t_m
                        best_sku_mix = t_sm
                if best_cg is None or best_improvement <= 0:
                    break
                accepted = [x for x in accepted if (getattr(x,'chain_group_id',None) or "") != best_cg]
                post_fifo = best_fifo
                post_mix = best_mix
                post_sku_mix = best_sku_mix

            logger.warning(
                f"[regression-gate] round {_round+1}: greedy done "
                f"(FIFO {baseline_fifo}->{post_fifo}, mix {baseline_mix}->{post_mix}, "
                f"moves={len(accepted)})"
            )
            if post_fifo <= baseline_fifo and post_mix <= baseline_mix and post_sku_mix <= baseline_sku_mix:
                break

        # チェーン単位で除去
        _before = len(accepted)
        accepted = [
            a for idx, a in enumerate(accepted)
            if idx not in _bad_indices and
            not (getattr(a, 'chain_group_id', None) and a.chain_group_id in _bad_chains)
        ]
        _removed = _before - len(accepted)

        post_lot_levels = _apply_moves_to_lot_levels(original_inv_lot_levels, accepted)
        post_fifo = _count_fifo_violations(post_lot_levels)
        post_mix = _count_lot_mixing(accepted, original_lot_strings_by_loc_sku)
        post_sku_mix = _count_sku_mixing(accepted, original_skus_by_loc or {}, original_qty_by_loc_sku or {}) if original_skus_by_loc else 0

        logger.warning(
            f"[regression-gate] round {_round+1}: removed {_removed} moves "
            f"({len(_bad_chains)} chains, {len(_bad_indices)} unchained) "
            f"(FIFO {baseline_fifo}->{post_fifo}, mix {baseline_mix}->{post_mix}, sku_mix {baseline_sku_mix}->{post_sku_mix})"
        )

        if _removed == 0:
            break

    return accepted


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
    can_receive_set: Optional[Set[str]] = None  # None = ロケマスタ未提供 → 全ロケ受入可
    if loc_master is not None and not loc_master.empty:
        # Narrow candidate destinations to the specified blocks/qualities if columns exist
        orig_slots = len(loc_master)
        lm_scoped = _filter_loc_master_by_block_quality(loc_master, block_filter, quality_filter)
        cap_by_loc, can_receive_set = _cap_map_from_master(lm_scoped, getattr(cfg, "fill_rate", DEFAULT_FILL_RATE))
        logger.debug(f"[optimizer] location_master provided: slots={len(cap_by_loc)} receivable={len(can_receive_set)} (filtered from {orig_slots})")
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
    logger.debug(f"[optimizer] sku_master cols={list(sku_master.columns)} rows={len(sku_master)}")
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
        inv = inv.reset_index(drop=True)
    logger.debug(f"[optimizer] after block_filter={list(block_filter) if block_filter is not None else None} rows={len(inv)}")
    try:
        _publish_progress(get_current_trace_id(), {
            "type": "info", "phase": "filter",
            "message": f"ブロックフィルター適用後: {len(inv)}行"
        })
    except Exception:
        pass

    # --- 別SKU混在チェック用マップ（ブロック・品質・引当フィルタ全て適用前に構築）---
    # 全ブロック・全品質・引当在庫も含めた完全な loc→Set[sku] マップを作成する。
    # ブロックフィルタ後に構築すると、同じ物理ロケにある別ブロックの在庫が見えず、
    # 品質フィルタ後に構築すると、販促物/什器のみのロケーションが空に見え、
    # 引当フィルタ後に構築すると、引当在庫しかないロケーションが空に見え、
    # 別SKUの移動先として選ばれてしまう。
    # inventory（フィルタ前の元データ）から構築する。
    _original_skus_by_loc: Dict[str, Set[str]] = {}
    _original_lots_by_loc_sku: Dict[Tuple[str, str], Set[int]] = {}
    _pre_q_locs_src = inventory["ロケーション"].astype(str).str.replace('.0', '', regex=False).str.zfill(8)
    _pre_q_skus_src = inventory["商品ID"].astype(str)
    _pre_q_lots_src = inventory["ロット"].map(_parse_lot_date_key) if "ロット" in inventory.columns else pd.Series([UNKNOWN_LOT_KEY] * len(inventory), index=inventory.index)
    _pre_q_qty_col_src = "cases" if "cases" in inventory.columns else ("ケース" if "ケース" in inventory.columns else None)
    _pre_q_lot_strs_src = inventory["ロット"].astype(str) if "ロット" in inventory.columns else pd.Series([""] * len(inventory), index=inventory.index)
    # FIFOインデックス用にはcasesではなく在庫数(qty)を使用（cases=0でもqty>0の行を正しくカウントするため）
    _qty_col_for_fifo = "在庫数(引当数を含む)" if "在庫数(引当数を含む)" in inventory.columns else _pre_q_qty_col_src
    _pre_q_qtys_src = pd.to_numeric(inventory[_qty_col_for_fifo], errors="coerce").fillna(0.0) if _qty_col_for_fifo else pd.to_numeric(inventory.get("在庫数(引当数を含む)", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    # ブロックフィルタ後の在庫からも参照用にlocs/skusを構築
    _pre_q_locs = inv["ロケーション"].astype(str).str.replace('.0', '', regex=False).str.zfill(8)
    _pre_q_skus = inv["商品ID"].astype(str)
    _pre_q_lots = inv["ロット"].map(_parse_lot_date_key) if "ロット" in inv.columns else pd.Series([UNKNOWN_LOT_KEY] * len(inv), index=inv.index)
    _pre_q_qty_col = "cases" if "cases" in inv.columns else ("ケース" if "ケース" in inv.columns else None)
    _pre_q_lot_strs = inv["ロット"].astype(str) if "ロット" in inv.columns else pd.Series([""] * len(inv), index=inv.index)
    _pre_q_qtys = pd.to_numeric(inv[_pre_q_qty_col], errors="coerce").fillna(0.0) if _pre_q_qty_col else pd.to_numeric(inv.get("在庫数(引当数を含む)", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    _original_qty_by_loc_sku_full: Dict[Tuple[str, str], float] = {}
    _original_lot_strings_by_loc_sku: Dict[Tuple[str, str], Set[str]] = {}
    _original_inv_lot_levels: Dict[Tuple[str, int], list] = {}  # (sku, col) -> [(lot_key, lv)]
    # フィルタ前の全在庫（全ブロック・全品質）からマップ構築
    # casesベースのqtyマップ（SKU mixing, enforceのqty tracking用）
    _pre_q_cases_src = pd.to_numeric(inventory[_pre_q_qty_col_src], errors="coerce").fillna(0.0) if _pre_q_qty_col_src else pd.Series(0.0, index=inventory.index)
    # pack_qty 参照マップ (sku_master ベース, floor された cases カラムを信頼せずに正確な case 数を計算する)
    _pack_lookup_full: Dict[str, float] = {}
    if sku_master is not None and not sku_master.empty:
        _sm_key = "sku_id" if "sku_id" in sku_master.columns else "商品ID"
        _sm_pack_col = "入数" if "入数" in sku_master.columns else ("pack_qty" if "pack_qty" in sku_master.columns else None)
        if _sm_pack_col:
            for _sk_raw, _pq_raw in zip(sku_master[_sm_key].astype(str), pd.to_numeric(sku_master[_sm_pack_col], errors="coerce").fillna(0.0)):
                try:
                    _pv = float(_pq_raw)
                    if _pv > 0:
                        _pack_lookup_full[_sk_raw] = _pv
                except Exception:
                    pass
    for _loc_v, _sku_v, _lot_v, _qty_v, _lot_str_v, _cases_v in zip(_pre_q_locs_src, _pre_q_skus_src, _pre_q_lots_src, _pre_q_qtys_src, _pre_q_lot_strs_src, _pre_q_cases_src):
        if float(_qty_v) <= 0:
            continue
        _original_skus_by_loc.setdefault(_loc_v, set()).add(_sku_v)
        _original_lots_by_loc_sku.setdefault((_loc_v, _sku_v), set()).add(int(_lot_v))
        # 正確なケース数: qty(pieces) / pack_qty。
        # DB の cases フィールドは floor されている可能性があるため信頼しない
        # (例: qty=464, pack=60 なら実質 7.733 ケースだが cases=7.0 と格納される)。
        # これにより _resolve_move_dependencies の部分退避検出が誤判定する。
        _pack_v_full = _pack_lookup_full.get(str(_sku_v), 0.0)
        if _pack_v_full > 0:
            _effective_cases = float(_qty_v) / _pack_v_full
        else:
            # pack_qty 不明: cases フィールドにフォールバック (従来通り)
            _effective_cases = float(_cases_v) if float(_cases_v) > 0 else 1.0
        _original_qty_by_loc_sku_full[(_loc_v, _sku_v)] = _original_qty_by_loc_sku_full.get((_loc_v, _sku_v), 0.0) + _effective_cases
        if _lot_str_v and _lot_str_v != "nan":
            _original_lot_strings_by_loc_sku.setdefault((_loc_v, _sku_v), set()).add(_lot_str_v)
        # FIFOインデックス（フィルタ前全在庫）を構築
        _ol_lk = int(_lot_v)
        if _ol_lk != UNKNOWN_LOT_KEY and float(_qty_v) > 0:
            _ol_lv, _ol_col, _ = _parse_loc8(_loc_v)
            _original_inv_lot_levels.setdefault((_sku_v, _ol_col), []).append((_ol_lk, _ol_lv))
    logger.debug(f"[optimizer] original_skus_by_loc (pre-quality/alloc filter): {len(_original_skus_by_loc)} locs")
    logger.debug(f"[optimizer] original_lots_by_loc_sku (pre-quality/alloc filter): {len(_original_lots_by_loc_sku)} entries")

    if quality_filter is not None:
        qset = set(str(x) for x in quality_filter)
        inv = inv[inv["quality_name"].astype(str).isin(qset)].copy()
        inv = inv.reset_index(drop=True)
    logger.debug(f"[optimizer] after quality_filter={list(quality_filter) if quality_filter is not None else None} rows={len(inv)}")
    try:
        _publish_progress(get_current_trace_id(), {
            "type": "info", "phase": "filter",
            "message": f"品質フィルター適用後: {len(inv)}行"
        })
    except Exception:
        pass

    # --- 退避不可ロケーション（非対象品質のSKUが存在するロケ）を特定 ---
    # 品質フィルタ外の在庫が1行でも存在するロケーションは移動先として使用不可。
    # SKU単位ではなく行単位で判定する（同じSKUが良品と外装不良の両方にある場合も検出）。
    _filtered_locs: Set[str] = set()
    _f_locs = inv["ロケーション"].astype(str).str.replace('.0', '', regex=False).str.zfill(8)
    for _fl in _f_locs:
        _filtered_locs.add(_fl)
    _pre_q_loc_set: Set[str] = set(_pre_q_locs)
    _blocked_dest_locs: Set[str] = set()
    # 品質フィルタ前にあったがフィルタ後に消えたロケ = 非対象品質のみのロケ → ブロック
    # 品質フィルタ前にあってフィルタ後にもあるが行数が減ったロケ = 非対象品質の行もあるロケ → ブロック
    # → 簡潔に: 品質フィルタ前の全ロケ行数 vs フィルタ後の行数を比較
    from collections import Counter as _Counter
    _pre_q_loc_counts = _Counter(_pre_q_locs)
    _post_q_loc_counts = _Counter(_f_locs)
    for _bl_loc in _pre_q_loc_set:
        if _pre_q_loc_counts[_bl_loc] > _post_q_loc_counts.get(_bl_loc, 0):
            _blocked_dest_locs.add(_bl_loc)
    logger.debug(f"[optimizer] blocked_dest_locs (unevictable foreign SKU): {len(_blocked_dest_locs)} locs")

    # プレースホルダロケは移動元として扱わない
    inv = inv[~inv["ロケーション"].astype(str).isin(PLACEHOLDER_LOCS)].copy()
    inv = inv.reset_index(drop=True)

    # ロケマスタに存在するロケーションのみを移動対象とする
    if cap_by_loc:
        valid_locs_set = set(cap_by_loc.keys())
        before_count = len(inv)
        inv = inv[inv["ロケーション"].astype(str).str.zfill(8).isin(valid_locs_set)].copy()
        inv = inv.reset_index(drop=True)
        excluded_count = before_count - len(inv)
        logger.debug(f"[optimizer] ロケマスタ存在チェック: {excluded_count}件除外 → 残り{len(inv)}件")
        try:
            _publish_progress(get_current_trace_id(), {
                "type": "info", "phase": "filter",
                "message": f"ロケマスタ存在チェック: {excluded_count}件除外 → 残り{len(inv)}件"
            })
        except Exception:
            pass

    # 引当在庫は移動対象外: '引当数' が存在し、かつ >0 の行は除外
    # B4: fillna(1) で変換不能値を安全側（引当あり=除外）として扱う
    try:
        if "引当数" in inv.columns:
            inv = inv[pd.to_numeric(inv["引当数"], errors="coerce").fillna(1) <= 0].copy()
            inv = inv.reset_index(drop=True)
    except Exception:
        # 列がない/変換不可などは無視（従来通り全件対象）
        pass

    # --- FIFOインデックスは引当フィルタ前の全在庫（inventoryパラメータ）から構築済み ---
    # 引当済み在庫もFIFO違反としてカウントする（棚に物理的に存在するため）
    # _original_inv_lot_levels は L6787+L6795 で既に構築されている
    logger.warning(f"[optimizer] FIFOインデックス（引当含む全在庫）: {len(_original_inv_lot_levels)} groups, {sum(len(v) for v in _original_inv_lot_levels.values())} entries")

    # --- SKU 段ボール容積マップ（m³/ケース）
    sku_vol_map = _build_carton_volume_map(sku_master)
    logger.debug(f"[optimizer] carton_volume_map size={sku_vol_map.shape[0]}")
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
    logger.debug(f"[optimizer] Sample locations before parse: {sample_locs}")
    logger.debug(f"[optimizer] Location dtype: {inv['ロケーション'].dtype}")
    
    # Ensure locations are zero-padded 8-digit strings
    inv["ロケーション"] = inv["ロケーション"].astype(str).str.replace('.0', '').str.zfill(8)
    sample_locs_after = inv["ロケーション"].head(5).tolist()
    logger.debug(f"[optimizer] Sample locations after parse: {sample_locs_after}")

    # --- 複数SKU混在ロケーションを分類（移動元・移動先の除外を SKU数で分ける）
    # 品質フィルタ前の全在庫（_original_skus_by_loc）を使って混在判定
    # 品質フィルタ後のinvでは1SKUでも、フィルタ前に別品質の別SKUがいるロケを含める
    #
    # 原始SKU数による分類:
    #   0 (空き)  → _original_empty_locs（後段で構築）: 受入可
    #   1-2 SKU   → multi_sku_locs_receivable: 受入可（最大3SKUまで）
    #   3+ SKU    → multi_sku_locs_strict: 完全ブロック
    multi_sku_locs_strict: Set[str] = set()    # 3SKU以上: 移動元・移動先ともに完全ブロック
    multi_sku_locs_receivable: Set[str] = set()  # 1-2SKU: 移動元はブロック、移動先は受入可
    for _ml_loc, _ml_skus in _original_skus_by_loc.items():
        _ml_n = len(_ml_skus)
        if _ml_n >= 3:
            multi_sku_locs_strict.add(_ml_loc)
        elif _ml_n >= 1:
            multi_sku_locs_receivable.add(_ml_loc)

    # 後方互換: 旧 multi_sku_locs は safety-net / 最終除去で参照 → strict のみ（完全ブロック対象）
    multi_sku_locs = multi_sku_locs_strict

    # 移動元ブロック: 2SKU以上のロケ（receivable + strict）は移動元にしない
    _multi_sku_source_blocked = multi_sku_locs_strict | multi_sku_locs_receivable
    # 移動先ブロック: 3SKU以上のロケのみ（receivable は allow_empty_loc_multi_sku で受入可能）
    _multi_sku_dest_blocked = multi_sku_locs_strict

    if _multi_sku_source_blocked:
        before_inv_count = len(inv)
        # 移動元から除外: 2SKU以上の混在ロケにある在庫は一切移動しない
        _in_multi = inv["ロケーション"].isin(_multi_sku_source_blocked)
        inv = inv[~_in_multi].copy()
        inv = inv.reset_index(drop=True)
        excluded_inv_count = before_inv_count - len(inv)
        logger.debug(f"[optimizer] 複数SKU混在ロケ除外（移動元）: {excluded_inv_count}件を移動候補から除外")

        # 移動先から除外: can_receive_set から strict（3SKU以上）のみ削除
        if can_receive_set is not None:
            before_recv_count = len(can_receive_set)
            can_receive_set = can_receive_set - _multi_sku_dest_blocked
            excluded_recv_count = before_recv_count - len(can_receive_set)
        else:
            excluded_recv_count = 0

        # strict ロケを _blocked_dest_locs にも追加（退避チェーンからも完全ブロック）
        _blocked_dest_locs |= _multi_sku_dest_blocked

        logger.debug(
            f"[optimizer] 複数SKU混在ロケ除外: strict={len(multi_sku_locs_strict)}ロケ→完全ブロック, "
            f"receivable={len(multi_sku_locs_receivable)}ロケ→移動元のみ除外({excluded_inv_count}件)"
        )
        try:
            _publish_progress(get_current_trace_id(), {
                "type": "info", "phase": "filter",
                "message": (
                    f"複数SKU混在ロケ除外: strict={len(multi_sku_locs_strict)}ロケ完全ブロック, "
                    f"receivable={len(multi_sku_locs_receivable)}ロケ移動元{excluded_inv_count}件除外"
                )
            })
        except Exception:
            pass

        # 退避不可ロケ（非対象品質SKUが物理的に存在）を移動先から除外
        if can_receive_set and _blocked_dest_locs:
            _before_blocked = len(can_receive_set)
            can_receive_set = can_receive_set - _blocked_dest_locs
            _blocked_excluded = _before_blocked - len(can_receive_set)
            logger.debug(f"[optimizer] 退避不可ロケ除外: {_blocked_excluded}ロケを移動先から除外（非対象品質SKU存在）")
        # can_receive_set が無い場合（ロケマスタ無し）でもブロックを効かせるため
        # 各パスの候補ループで _blocked_dest_locs を直接チェックする必要がある
        # → Pass-2, Pass-3 の候補ループにもチェックを追加済み
    else:
        print("[optimizer] 複数SKU混在ロケなし")

    # --- 元々空きだったロケ集合を構築（複数SKU同居機能用）
    # _original_skus_by_loc に存在しないロケ、または存在するがSKU数=0のロケが対象
    # can_receive_set がある場合はそのロケ、ない場合は shelf_usage のロケを基準にする
    _original_empty_locs: Set[str] = set()
    _original_receivable_locs: Set[str] = set()  # 空き(0SKU) + 1-2SKU ロケ（受入可能集合）
    if getattr(cfg, "allow_empty_loc_multi_sku", True):
        _all_known_locs: Set[str] = set()
        if can_receive_set is not None:
            _all_known_locs = set(can_receive_set)
        # shelf_usage に含まれるロケも候補（in-memoryで管理されているもの）
        # _original_skus_by_loc に含まれないロケ = 元々空き
        _all_known_locs |= {str(loc).zfill(8) for loc in _original_skus_by_loc.keys()}
        for _el_loc, _el_skus in _original_skus_by_loc.items():
            if len(_el_skus) == 0:
                _original_empty_locs.add(_el_loc)
        # _original_skus_by_loc に含まれないロケはすべて元々空き
        # （can_receive_set から _original_skus_by_loc を引いた差分）
        if can_receive_set is not None:
            _original_empty_locs |= can_receive_set - set(_original_skus_by_loc.keys())
        # strict（3SKU以上）は除外
        _original_empty_locs -= multi_sku_locs_strict
        # 受入可能集合 = 空き(0SKU) + 1-2SKU ロケ
        _original_receivable_locs = _original_empty_locs | multi_sku_locs_receivable
        logger.debug(
            f"[optimizer] _original_empty_locs: {len(_original_empty_locs)}ロケ（元々空き）, "
            f"_original_receivable_locs: {len(_original_receivable_locs)}ロケ（空き+1-2SKU）"
        )

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
            logger.debug(f"[optimizer] qty_cases_move min={qmin} max={qmax}")
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

    # --- 別SKU混在チェック用ケース数マップ（パス前の完全な状態）---
    # enforce_constraints の _sim_skus_by_loc 更新で、移動元のSKUを除去する際に
    # 「その(loc,sku)の残ケース数」を正確に追跡する必要がある。
    # _sim_lots_by_loc_sku はパス後の変異invから構築されるため、
    # パスで移動済みの行は残ケース=0となり、不正にSKUが除去されてしまうバグがある。
    # ここでパス前の正確なケース数を保存し enforce_constraints に渡す。
    # NOTE: qty_cases_float は DB の cases カラム（floor 済み）を優先使用するため
    # 端数ケースが切り捨てられる。_orig_qty / _pack_qty で正確な float を使う。
    _original_qty_by_loc_sku: Dict[Tuple[str, str], float] = {}
    _exact_cases_series = (
        inv["_orig_qty"].astype(float) / inv["_pack_qty"].astype(float)
    )
    for _loc_v2, _sku_v2, _qty_v2 in zip(
        inv["ロケーション"].astype(str).str.replace('.0', '', regex=False).str.zfill(8),
        inv["商品ID"].astype(str),
        _exact_cases_series,
    ):
        _key2 = (_loc_v2, _sku_v2)
        _original_qty_by_loc_sku[_key2] = _original_qty_by_loc_sku.get(_key2, 0.0) + _qty_v2
    logger.debug(f"[optimizer] original_qty_by_loc_sku: {len(_original_qty_by_loc_sku)} entries")
    
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
    logger.debug(f"[optimizer] バラ除外: バラのみ={bara_only_count}件, 10ケース以下+バラ={low_case_bara_count}件, 移動対象={movable_count}件")
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
        inv = inv.reset_index(drop=True)

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

    # 列ごとの代表入数: _compute_rep_pack_by_col_for_inv に統一（DRY）
    _preserve_thr = getattr(cfg, "preserve_pack_mapping_threshold", 0.7)
    rep_pack_by_col: dict[int, float] = _compute_rep_pack_by_col_for_inv(inv, pack_map, _preserve_thr)
    logger.debug(f"[optimizer] rep_pack_by_col keys={len(rep_pack_by_col)} thr={_preserve_thr}")

    # 列ごとの"ミックス枠"初期化（±帯から外れても許容できる枠）
    try:
        unique_cols = sorted(int(x) for x in inv["col"].dropna().unique())
    except Exception:
        unique_cols = []
    mix_slots_left: dict[int, int] = {c: int(getattr(cfg, "mix_slots_per_col", 1)) for c in unique_cols}

    # 容積情報が無い行はスキップ
    inv = inv[inv["volume_each_case"] > 0].copy()
    inv = inv.reset_index(drop=True)  # フィルタ後のスパースインデックスをリセット

    # --- 棚毎の使用量 (m³)
    shelf_usage = inv.groupby("ロケーション")["volume_total"].sum().to_dict()
    # Ensure empty candidate locations exist in shelf_usage when loc_master provided
    if cap_by_loc:
        for loc in cap_by_loc.keys():
            shelf_usage.setdefault(loc, 0.0)
    logger.debug(f"[optimizer] shelf_usage locations={len(shelf_usage)} (cap={cap_limit})")
    moves: List[Move] = []

    # --- 実行順序: Pass-FIFO → Pass-1 → Pass-0 (列跨ぎFIFOを最優先、同列内整列を次に) ---
    import time
    def _log(msg):  # local log helper
        try:
            logger.debug(f"[optimizer] {msg}")
        except Exception:
            pass
    # Track Pass-wise statistics for summary report
    pass_stats = {"passC": 0, "pass0": 0, "pass1": 0, "pass2": 0, "pass3": 0}
    
    # --- 200件制限の中で最適な組み合わせを実現 ---
    # 全Pass候補を収集してから、優先度でソートして上位200件を選択
    
    all_candidate_moves: List[Move] = []
    
    # --- 別SKU混在チェック用マップ（全パスで共有・逐次更新される）---
    # _original_skus_by_loc（品質フィルタ前）を使用する。
    # _filtered_skus_by_loc だと出荷止め・外装不良・販促物SKUが見えず、
    # FIFOスワップの pick_loc に非対象品質SKUがいても検知できない。
    # blocked_dest_locs は evict_dest のみをブロックし pick_loc はブロックしないため、
    # 品質フィルタ前の完全なSKUマップで混在チェックする必要がある。
    _pass_skus_by_loc: Dict[str, Set[str]] = {
        k: set(v) for k, v in _original_skus_by_loc.items()
    }
    # --- ロケーション×SKU別の行数カウンタ（O(1)残存チェック用）---
    _p2_sku_count: Dict[Tuple[str, str], int] = {}
    if "商品ID" in inv.columns and "ロケーション" in inv.columns:
        for _cl2, _cs2 in zip(inv["ロケーション"].astype(str), inv["商品ID"].astype(str)):
            _p2_sku_count[(_cl2, _cs2)] = _p2_sku_count.get((_cl2, _cs2), 0) + 1
    # --- ロット混在チェック用マップ（全パスで共有・逐次更新される）---
    _pass_lots_by_loc_sku: Dict[Tuple[str, str], Set[int]] = {
        k: set(v) for k, v in _original_lots_by_loc_sku.items()
    }
    _pass_lot_strings_by_loc_sku: Dict[Tuple[str, str], Set[str]] = {
        k: set(v) for k, v in _original_lot_strings_by_loc_sku.items()
    }
    # --- 二重移動防止用セット（全パスで共有）---
    _moved_indices: Set[int] = set()

    # --- Pass-FIFO: 最古ロットをLv1-2へ配置するFIFO整列パス【最優先: Pass-Cより先に実行】
    # Pass-C が先に走ると moved_indices を汚染し、Pass-FIFO のスワップ候補が除外される問題を回避。
    # (例: D2342511 の Lv1-c26 が Pass-C で移動済みになると、Lv3-c14(OLDEST) のスワップ相手が消える)
    pf_t0 = time.perf_counter()
    if not getattr(cfg, "enable_pass_fifo", True):
        pf_moves = []
        _log("Pass-FIFO: SKIPPED (disabled)")
    else:
        pf_moves = _pass_fifo_to_pick(
            inv, shelf_usage, cap_limit, cfg,
            cap_by_loc=cap_by_loc or None,
            can_receive=can_receive_set if can_receive_set is not None else None,
            skus_by_loc=_pass_skus_by_loc,
            lots_by_loc_sku=_pass_lots_by_loc_sku,
            lot_strings_by_loc_sku=_pass_lot_strings_by_loc_sku,
            moved_indices=_moved_indices,
            blocked_dest_locs=_blocked_dest_locs,
            sku_vol_map=sku_vol_map,
            rep_pack_by_col=rep_pack_by_col,
            budget_left=None,
            trace_id=getattr(cfg, "trace_id", None),
        )
    all_candidate_moves.extend(pf_moves)
    _log(f"Pass-FIFO to_pick: candidates={len(pf_moves)} time={time.perf_counter()-pf_t0:.2f}s")

    # --- Pass-C: 集約（同じSKU×ロットの分散を1箇所にまとめる）【Pass-FIFOの後に実行】---
    pc_t0 = time.perf_counter()
    if not getattr(cfg, "enable_pass_consolidate", True):
        pc_moves = []
        _log("Pass-C consolidate: SKIPPED (disabled)")
    else:
        pc_moves = _pass_consolidate(
        inv=inv, shelf_usage=shelf_usage, cap_limit=cap_limit, cfg=cfg,
        sku_vol_map=sku_vol_map,
        cap_by_loc=cap_by_loc or None,
        can_receive=can_receive_set if can_receive_set is not None else None,
        skus_by_loc=_pass_skus_by_loc,
        lots_by_loc_sku=_pass_lots_by_loc_sku,
        moved_indices=_moved_indices,
        blocked_dest_locs=_blocked_dest_locs,
        blocked_source_locs=_multi_sku_source_blocked if _multi_sku_source_blocked else None,
    )
    all_candidate_moves.extend(pc_moves)
    _log(f"Pass-C consolidate: candidates={len(pc_moves)} time={time.perf_counter()-pc_t0:.2f}s")

    # (Post-Pass-1 FIFO直接スワップは _pass_fifo_to_pick に統合済みのため削除)
    _fifo_direct_bypass: List[Move] = []  # 後続コードとの互換性のために空リストを維持

    # --- Pass-0: エリア（列ゾーニング）の是正（列またぎ再配置）【第2優先】
    # SKUごとの最古ロットを算出（Pass-0で取口の最古ロットを保護するため）
    _p0_oldest_lot_by_sku: Optional[Dict[str, int]] = None
    if "lot_key" in inv.columns:
        _p0_valid = inv[inv["lot_key"] != UNKNOWN_LOT_KEY]
        if not _p0_valid.empty:
            _p0_oldest_lot_by_sku = _p0_valid.groupby("商品ID")["lot_key"].min().to_dict()

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
            can_receive=can_receive_set if can_receive_set is not None else None,
            budget_left=None,  # 制限なしで候補収集
            skus_by_loc=_pass_skus_by_loc,
            lots_by_loc_sku=_pass_lots_by_loc_sku,
            moved_indices=_moved_indices,
            blocked_dest_locs=_blocked_dest_locs,
            original_skus_by_loc=_original_skus_by_loc,
            oldest_lot_by_sku=_p0_oldest_lot_by_sku,
        )
        all_candidate_moves.extend(p0_moves)
        _log(f"Pass-0 area_rebalance: candidates={len(p0_moves)} time={time.perf_counter()-p0_t0:.2f}s")

    # --- グローバルインデックスをゾーンスワップ前にビルド（FIFOチェックで使用）
    with _OPTIMIZER_LOCK:
        _build_global_inv_indexes(inv)

    # --- Pass-0 zone swaps: ゾーン違反アイテム同士のスワップ（空きロケ不要）
    # エリア是正とは独立して動作する
    p0_swap_t0 = time.perf_counter()
    if not getattr(cfg, "enable_pass_zone_swap", True):
        p0_swap_moves = []
        _log("Pass-0 zone_swaps: SKIPPED (disabled)")
    else:
        p0_swap_moves = _pass0_zone_swaps(
            inv=inv,
            shelf_usage=shelf_usage,
            cap_limit=cap_limit,
            cfg=cfg,
            rep_pack_by_col=rep_pack_by_col,
            sku_vol_map=sku_vol_map,
            cap_by_loc=cap_by_loc or None,
            can_receive=can_receive_set if can_receive_set is not None else None,
            budget_left=None,
            skus_by_loc=_pass_skus_by_loc,
            lots_by_loc_sku=_pass_lots_by_loc_sku,
            moved_indices=_moved_indices,
            blocked_dest_locs=_blocked_dest_locs,
            oldest_lot_by_sku=_p0_oldest_lot_by_sku,
            original_skus_by_loc=_original_skus_by_loc,
        )
    all_candidate_moves.extend(p0_swap_moves)
    _log(f"Pass-0 zone_swaps: candidates={len(p0_swap_moves)} time={time.perf_counter()-p0_swap_t0:.2f}s")

    logger.debug(f"[optimizer] Total candidates collected: {len(all_candidate_moves)}")
    
    # --- 優先度ソート: Pass-1 > Pass-0 (reasonキーワードで判定) ---
    def _get_pass_priority(m: Move) -> int:
        reason = str(m.reason or "")
        # Pass-C: 集約（同一ロット統合）— 最優先
        if "同一ロット統合" in reason:
            return 0
        # Pass-1: FIFO / 取口保管整列
        if any(k in reason for k in (
            "古ロット", "新ロット", "先入先出", "取口配置", "取口内整列", "取口内再配置",
            "保管内整列", "保管内再配置", "FIFO是正", "最古ロット優先", "FIFO順序",
            "段下げ", "段上げ", "同列内移動", "手前化", "同一レベル内移動",
            "スワップ準備退避",
        )):
            return 1
        # Pass-0: エリア再配置 / ゾーンスワップ
        if any(k in reason for k in (
            "入数帯是正", "エリア移動", "エリア", "列移動", "入口に近づく",
            "入口接近", "適正エリア", "位置最適化", "ゾーンスワップ",
        )):
            return 2
        # Pass-2: 圧縮 / 集約
        if any(k in reason for k in (
            "圧縮", "集約", "SKU集約", "混載枠", "配置最適化", "スペース確保",
            "大幅段下げ", "容量確保退避",
        )):
            return 3
        # Pass-3: AI
        if "AI" in reason:
            return 4
        return 5
    
    def _calc_ease_gain(m: Move) -> float:
        # 移動効果: ease_key の改善量（from の取りにくさ - to の取りにくさ）
        # 正の値 = アクセスしやすい場所への移動（改善）
        # 負の値 = アクセスしにくい場所への移動（悪化）
        try:
            from_lv, from_col, from_dep = _parse_loc8(str(m.from_loc))
            to_lv, to_col, to_dep = _parse_loc8(str(m.to_loc))
            from_ease = _ease_key(from_lv, from_col, from_dep)
            to_ease = _ease_key(to_lv, to_col, to_dep)
            # from_ease > to_ease = 取りにくい場所から取りやすい場所へ = 正の改善量
            return float(from_ease - to_ease)
        except Exception:
            return 0.0
    
    logger.debug(f"[optimizer] About to sort {len(all_candidate_moves)} candidates...")
    # 優先度順にソート (Pass優先 → 効果大きい順)
    # ただし chain_group_id がある移動はグループ単位でまとめて並べる
    try:
        moves_with_meta = []
        for i, m in enumerate(all_candidate_moves):
            if i % 100 == 0:
                logger.debug(f"[optimizer] Processing candidate {i}/{len(all_candidate_moves)}...")
            try:
                priority = _get_pass_priority(m)
                gain = _calc_ease_gain(m)
                moves_with_meta.append((m, priority, gain))
            except Exception as e:
                logger.debug(f"[optimizer] ERROR processing candidate {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
        logger.debug(f"[optimizer] Created moves_with_meta with {len(moves_with_meta)} entries, now sorting...")
        
        # === chain_group_id 対応ソート ===
        # 1. chain_group_id でグループ化
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
        
        # 3. グループの代表優先度・効果・最古ロット日付を計算
        # Tuple: (chain_id, priority, gain, lot_key, group_moves)
        group_meta: List[Tuple[str, int, float, int, List[Tuple[Move, int, float]]]] = []
        for chain_id, group_moves in chain_groups.items():
            group_priority = min(m[1] for m in group_moves)  # 最高優先度（最小値）
            group_gain = sum(m[2] for m in group_moves)  # 合計効果
            # FIFO関連グループ（priority=1）の場合、最古ロット日付でソート
            group_lot_key = UNKNOWN_LOT_KEY
            if group_priority == 1:
                for gm, _, _ in group_moves:
                    lk = _datestr8_to_lot_key(gm.lot_date) if gm.lot_date else UNKNOWN_LOT_KEY
                    if lk < group_lot_key:
                        group_lot_key = lk
            # 退避チェーンのコスト: 補助移動数（execution_order > 1）をペナルティとして減算
            _chain_moves_count = len([m for m in group_moves if getattr(m[0], 'execution_order', 1) > 1])
            group_gain -= _chain_moves_count * 0.5
            group_meta.append((chain_id, group_priority, group_gain, group_lot_key, group_moves))

        # 4. グループを優先度→ロット日付（古い順）→効果順でソート
        #    FIFO関連（priority=1）は古いロット順を優先、それ以外は効果順
        group_meta.sort(key=lambda x: (x[1], x[3], -x[2]))  # priority ASC, lot_key ASC, gain DESC

        # 5. スタンドアロン移動もソート（ロット日付考慮）
        def _standalone_sort_key(item: Tuple[Move, int, float]) -> Tuple:
            m, pri, g = item
            lk = _datestr8_to_lot_key(m.lot_date) if pri == 1 and m.lot_date else UNKNOWN_LOT_KEY
            return (pri, lk, -g)
        standalone_moves.sort(key=_standalone_sort_key)  # priority ASC, lot_key ASC, gain DESC
        
        # 6. グループとスタンドアロンを統合（優先度→ロット日付→効果順でマージ）
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
                for m_tuple in group_meta[group_idx][4]:
                    sorted_moves_with_meta.append(m_tuple)
                group_idx += 1
            else:
                # 両方ある -> 優先度・ロット日付・効果で比較
                g_priority, g_gain, g_lot = group_meta[group_idx][1], group_meta[group_idx][2], group_meta[group_idx][3]
                s_m, s_priority, s_gain = standalone_moves[standalone_idx][0], standalone_moves[standalone_idx][1], standalone_moves[standalone_idx][2]
                s_lot = _datestr8_to_lot_key(s_m.lot_date) if s_priority == 1 and s_m.lot_date else UNKNOWN_LOT_KEY

                if (g_priority, g_lot, -g_gain) <= (s_priority, s_lot, -s_gain):
                    # グループの方が優先
                    for m_tuple in group_meta[group_idx][4]:
                        sorted_moves_with_meta.append(m_tuple)
                    group_idx += 1
                else:
                    # スタンドアロンの方が優先
                    sorted_moves_with_meta.append(standalone_moves[standalone_idx])
                    standalone_idx += 1
        
        moves_with_meta = sorted_moves_with_meta
        logger.debug(f"[optimizer] Sorting complete (chain_groups={len(chain_groups)}, standalone={len(standalone_moves)})")
    except Exception as e:
        logger.debug(f"[optimizer] FATAL ERROR in sorting: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # max_moves制限とSKU移動元ロケーション数制限を適用
    max_moves = getattr(cfg, "max_moves", None)
    max_source_locs_per_sku = getattr(cfg, "max_source_locs_per_sku", None)
    logger.debug(f"[optimizer] max_moves={max_moves}, max_source_locs_per_sku={max_source_locs_per_sku}, total_candidates={len(moves_with_meta)}")
    
    moves, skipped_by_limit = _select_candidate_moves(
        moves_with_meta,
        max_moves=max_moves,
        max_source_locs_per_sku=max_source_locs_per_sku,
    )

    if skipped_by_limit > 0:
        _log(f"SKU移動元ロケーション数制限により {skipped_by_limit} 件をスキップ")
        logger.debug(f"[optimizer] Skipped {skipped_by_limit} moves due to max_source_locs_per_sku={max_source_locs_per_sku}")
    
    if max_moves is not None and len(moves_with_meta) > len(moves):
        _log(f"Trimmed to {len(moves)} moves (from {len(moves_with_meta)} candidates)")
    
    logger.debug(f"[optimizer] After selection: len(moves)={len(moves)}")
    
    # Pass統計を更新
    pass_stats["passC"] = sum(1 for m in moves if _get_pass_priority(m) == 0)
    pass_stats["pass1"] = sum(1 for m in moves if _get_pass_priority(m) == 1)
    pass_stats["pass0"] = sum(1 for m in moves if _get_pass_priority(m) == 2)

    _log(f"Final move selection: Pass-C={pass_stats['passC']}, Pass-1={pass_stats['pass1']}, Pass-0={pass_stats['pass0']}, total={len(moves)}")
    
    # Debug: Log first few moves
    if moves:
        logger.debug(f"[optimizer] First 5 selected moves:")
        for i, m in enumerate(moves[:5]):
            priority = _get_pass_priority(m)
            logger.debug(f"  [{i}] priority={priority}, sku={m.sku_id}, from={m.from_loc}, to={m.to_loc}, reason={m.reason[:80] if m.reason else 'None'}")
    else:
        logger.debug(f"[optimizer] WARNING: No moves selected after sorting/filtering!")
    
    try:
        _publish_progress(get_current_trace_id(), {
            "type": "phase", "name": "pass_selection", 
            "moves": len(moves), "total_moves": len(moves),
            "passC": pass_stats["passC"], "pass1": pass_stats["pass1"], "pass0": pass_stats["pass0"],
            "message": f"最適な{len(moves)}件を選択 (Pass-C: {pass_stats['passC']}件, Pass-1: {pass_stats['pass1']}件, Pass-0: {pass_stats['pass0']}件)"
        })
    except Exception:
        pass
    
    moved = len(moves)
    # --- 出荷頻度マップの構築（高頻度SKUを優先処理するため）---
    _sku_velocity: Dict[str, float] = {}
    _velocity_col = None
    if "出荷数" in inv.columns:
        _velocity_col = "出荷数"
    elif "出荷ケース数" in inv.columns:
        _velocity_col = "出荷ケース数"
    if _velocity_col:
        _sku_velocity = inv.groupby("商品ID")[_velocity_col].sum().to_dict()
        _log(f"Shipment velocity map: {len(_sku_velocity)} SKUs (col={_velocity_col})")

    # Pass-0/Pass-1で全体が更新されたので並べ替えし直し
    # 出荷頻度が高いSKUを先に処理（取りやすい位置を優先確保）
    if _sku_velocity:
        inv["_velocity"] = inv["商品ID"].map(lambda s: -float(_sku_velocity.get(str(s), 0)))
        order_idx = inv.sort_values(
            by=["_velocity", "lot_key", "lv", "col", "dep"],
            ascending=[True, True, False, True, True],
            kind="mergesort",
        ).index
    else:
        order_idx = _sort_index_for_pick(inv)

    # --- strict pack-A gating flag ---
    hard_pack_A = bool(getattr(cfg, "strict_pack", None) and ("A" in str(getattr(cfg, "strict_pack"))))

    # 退避チェーン用: 空き容量上位ロケーションリスト
    _p2_base_eligible = [
        loc for loc in shelf_usage.keys()
        if loc not in PLACEHOLDER_LOCS and loc not in _blocked_dest_locs
        and (can_receive_set is None or loc in can_receive_set)
    ]
    _p2_max_prefilter = int(getattr(cfg, "eviction_dest_scan_limit", 200))
    _p2_prefiltered_locs: List[str] = sorted(
        _p2_base_eligible,
        key=lambda loc: (cap_by_loc.get(loc, cap_limit) if cap_by_loc else cap_limit) - shelf_usage.get(loc, 0.0),
        reverse=True
    )[:_p2_max_prefilter]

    # Track planned moves to same location for lot-mixing checks
    # Initialize with Pass-1 and Pass-0 moves
    main_loop_t0 = time.perf_counter()
    _skip_main_loop = not getattr(cfg, "enable_main_loop", True)
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
    # C3: Use lock to prevent race condition when multiple tenants optimize concurrently
    with _OPTIMIZER_LOCK:
        _build_global_inv_indexes(inv)
        _local_inv_indexes = copy.deepcopy(_GLOBAL_INV_INDEXES)

    # グローバルインデックスから再利用（二重構築を廃止）
    _log("Loading inventory indexes from global cache...")
    _idx_build_start = time.perf_counter()

    # Index 1: lots by (location, sku) -> set of lot_keys
    inv_lots_by_loc_sku = _local_inv_indexes["inv_lots_by_loc_sku"]
    # Index 2: columns where each (sku, lot_key) exists
    inv_cols_by_sku_lot = _local_inv_indexes["inv_cols_by_sku_lot"]
    # Index 4: lot_key -> level mapping by (sku, column) for FIFO rule check
    inv_lot_levels_by_sku_col = _local_inv_indexes["inv_lot_levels_by_sku_col"]

    # Index 3: inventory rows at each location for a SKU (for lot-level rule check)
    # このインデックスはグローバルキャッシュに含まれないため個別に構築する
    inv_rows_by_loc_sku: Dict[Tuple[str, str], List[Tuple[int, int, int]]] = {}  # list of (lot_key, level, idx)

    if _skip_main_loop:
        _log("Pass-2 main loop: SKIPPED (disabled)")
    elif "商品ID" in inv.columns and "lot_key" in inv.columns and "ロケーション" in inv.columns:
        for idx_row in inv.index:
            try:
                sku_v = str(inv.at[idx_row, "商品ID"])
                loc_v = str(inv.at[idx_row, "ロケーション"])
                lot_k = int(pd.to_numeric(inv.at[idx_row, "lot_key"], errors="coerce") or UNKNOWN_LOT_KEY)
                col_v = int(inv.at[idx_row, "col"]) if pd.notna(inv.at[idx_row, "col"]) else 0
                lv_v = int(inv.at[idx_row, "lv"]) if pd.notna(inv.at[idx_row, "lv"]) else 0

                # Index 3
                key1 = (loc_v, sku_v)
                if key1 not in inv_rows_by_loc_sku:
                    inv_rows_by_loc_sku[key1] = []
                inv_rows_by_loc_sku[key1].append((lot_k, lv_v, idx_row))
            except Exception:
                pass

    _log(f"Inventory indexes loaded in {time.perf_counter() - _idx_build_start:.2f}s: {len(inv_lots_by_loc_sku)} loc-sku pairs, {len(inv_cols_by_sku_lot)} sku-lot pairs, {len(inv_lot_levels_by_sku_col)} sku-col pairs")
    # ====== END PERFORMANCE OPTIMIZATION ======

    processed_rows = 0
    total_rows = len(order_idx)
    progress_mod = max(1, total_rows // 20)  # Show progress every 5% of total, at least every row
    cancellation_check_mod = max(1, total_rows // 10)  # Check cancellation every 10%
    
    # Time limit for main loop processing (default 5 minutes = 300 seconds)
    import time as _time_module
    _loop_start_time = _time_module.time()
    _loop_time_limit = float(getattr(cfg, "loop_time_limit", 300))  # seconds
    _time_check_mod = max(1, total_rows // 200)  # Check time every 0.5%
    
    def _log(msg):
        try:
            logger.debug(f"[optimizer] {msg}")
        except Exception:
            pass
    
    # Import cancellation check function
    try:
        from app.services.relocation_tasks import is_task_cancelled
    except ImportError:
        is_task_cancelled = lambda x: False  # Fallback if not available

    for idx in order_idx:
        if _skip_main_loop:
            break
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
        # 二重移動防止: 既に別パスで移動済みならスキップ
        if idx in _moved_indices:
            continue
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
        _p2_row_eviction_attempts = 0
        _p2_row_max_eviction = 3  # 1行あたり最大3回まで退避チェーン試行

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
        _p2_eviction_accepted = False  # 退避チェーン発動で即採用した場合のフラグ

        # --- Parallel or serial candidate evaluation ---
        use_parallel = getattr(cfg, "enable_parallel", False) and not getattr(cfg, "chain_depth", 0)
        
        if use_parallel:
            # Parallel evaluation: collect all candidates first, then evaluate in parallel
            all_candidates = []
            for target_level in levels:
                cand_locs = locs_by_level.get(target_level, [])
                if can_receive_set is not None:
                    cand_locs = [loc for loc in cand_locs if loc != from_loc and loc in can_receive_set]
                else:
                    cand_locs = [loc for loc in cand_locs if loc != from_loc and str(loc) not in _blocked_dest_locs]
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
                            hard_pack_A, depths_by_col_calc, cfg, _blocked_dest_locs,
                            _pass_skus_by_loc, _pass_lots_by_loc_sku
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
                if can_receive_set is not None:
                    cand_locs = [loc for loc in cand_locs if loc != from_loc and loc in can_receive_set]
                else:
                    cand_locs = [loc for loc in cand_locs if loc != from_loc and str(loc) not in _blocked_dest_locs]
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

                    # 別SKU混在チェック
                    _p2_existing_skus = _pass_skus_by_loc.get(str(to_loc), set())
                    _p2_has_foreign_sku = bool(_p2_existing_skus and not _p2_existing_skus <= {str(sku_val)})
                    if _p2_has_foreign_sku:
                        # 退避チェーンが有効な場合は外来SKUを退避して移動先を確保する
                        if not (getattr(cfg, "chain_depth", 0) and getattr(cfg, "eviction_budget", 0) and getattr(cfg, "touch_budget", 0)):
                            row_fail["forbidden"] += 1
                            continue
                        _p2_row_eviction_attempts += 1
                        if _p2_row_eviction_attempts > _p2_row_max_eviction:
                            row_fail["forbidden"] += 1
                            continue
                        # スナップショット保存（失敗時ロールバック用）
                        _snap_shelf_fsk = dict(shelf_usage)
                        _snap_skus_fsk = {k: set(v) for k, v in _pass_skus_by_loc.items()} if _pass_skus_by_loc is not None else None
                        _fsk_budget = _ChainBudget(
                            depth_left=int(getattr(cfg, "chain_depth", 1)),
                            evictions_left=int(getattr(cfg, "eviction_budget", 50)),
                            touch_left=int(getattr(cfg, "touch_budget", 100)),
                            touched=set([from_loc, str(to_loc)]),
                        )
                        _fsk_chain_group_id = f"evict_{secrets.token_hex(6)}"
                        _fsk_chain = _plan_eviction_chain(
                            need_vol=0.001,  # 外来SKU退避が目的（容量確保は別途処理）
                            target_loc=str(to_loc),
                            inv=inv,
                            shelf_usage=shelf_usage,
                            cap_limit=cap_limit,
                            sku_vol_map=sku_vol_map,
                            rep_pack_by_col=rep_pack_by_col,
                            pack_tolerance_ratio=getattr(cfg, "pack_tolerance_ratio", 0.10),
                            budget=_fsk_budget,
                            ai_col_hints=ai_col_hints,
                            cap_by_loc=cap_by_loc or None,
                            can_receive=can_receive_set if can_receive_set is not None else None,
                            hard_pack_A=hard_pack_A,
                            ease_weight=getattr(cfg, "ease_weight", 0.0001),
                            cfg=cfg,
                            chain_group_id=_fsk_chain_group_id,
                            execution_order_start=1,
                            evict_foreign_skus=str(sku_val),
                            blocked_dest_locs=_blocked_dest_locs,
                            skus_by_loc=_pass_skus_by_loc,
                            prefiltered_locs=_p2_prefiltered_locs,
                        )
                        if _fsk_chain is None:
                            # 退避失敗 → ロールバックしてこの移動先をスキップ
                            shelf_usage.clear()
                            shelf_usage.update(_snap_shelf_fsk)
                            if _pass_skus_by_loc is not None and _snap_skus_fsk is not None:
                                _pass_skus_by_loc.clear()
                                _pass_skus_by_loc.update(_snap_skus_fsk)
                            row_fail["forbidden"] += 1
                            logger.debug(f"[optimizer][Pass-2] foreign SKU eviction failed for {sku_val} → {to_loc}, skipping")
                            continue
                        # 退避成功 → 退避移動を候補チェーンに取り込み、採用して即ブレーク
                        logger.debug(f"[optimizer][Pass-2] foreign SKU eviction succeeded: {len(_fsk_chain)} moves for {to_loc}")
                        used = float(shelf_usage.get(to_loc, 0.0))
                        limit = cap_by_loc.get(to_loc, cap_limit) if cap_by_loc else cap_limit
                        if used + need_vol <= limit:
                            # 退避後に容量も足りた → このロケを即採用
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
                            best_score = score
                            best_choice = (to_loc, target_level, tcol, tdep, area_needs_mix)
                            best_ev_chain = _fsk_chain
                            best_ev_chain_group_id = _fsk_chain_group_id
                            _p2_eviction_accepted = True  # 外側ループも即抜ける
                        else:
                            # 退避後も容量不足 → ロールバックしてスキップ
                            shelf_usage.clear()
                            shelf_usage.update(_snap_shelf_fsk)
                            if _pass_skus_by_loc is not None and _snap_skus_fsk is not None:
                                _pass_skus_by_loc.clear()
                                _pass_skus_by_loc.update(_snap_skus_fsk)
                            row_fail["forbidden"] += 1
                        break  # eviction chain は状態変更済みのため to_loc ループを抜ける

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
                            _p2_row_eviction_attempts += 1
                            if _p2_row_eviction_attempts > _p2_row_max_eviction:
                                row_fail["capacity"] += 1
                                continue
                            # スナップショット保存（失敗時ロールバック用）
                            _snap_shelf_p2 = dict(shelf_usage)
                            _snap_skus_p2 = {k: set(v) for k, v in _pass_skus_by_loc.items()} if _pass_skus_by_loc is not None else None
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
                                can_receive=can_receive_set if can_receive_set is not None else None,
                                hard_pack_A=hard_pack_A,
                                ease_weight=getattr(cfg, "ease_weight", 0.0001),
                                cfg=cfg,
                                chain_group_id=candidate_ev_chain_group_id,
                                execution_order_start=1,
                                blocked_dest_locs=_blocked_dest_locs,
                                skus_by_loc=_pass_skus_by_loc,
                                prefiltered_locs=_p2_prefiltered_locs,
                            )
                            if candidate_ev_chain is None:
                                # ロールバック
                                shelf_usage.clear()
                                shelf_usage.update(_snap_shelf_p2)
                                if _pass_skus_by_loc is not None and _snap_skus_p2 is not None:
                                    _pass_skus_by_loc.clear()
                                    _pass_skus_by_loc.update(_snap_skus_p2)
                                row_fail["capacity"] += 1
                                continue
                            # Tentatively accept this candidate with the eviction chain
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
                    # 出荷頻度ボーナス: 高頻度SKUは低段・手前を強く優先
                    if _sku_velocity:
                        _vel = float(_sku_velocity.get(str(sku_val), 0))
                        if _vel > 0:
                            # 低段(lv=1)は高段(lv=4)より大幅にスコアが良くなる
                            score -= _vel * 0.001 * (4 - target_level)
                    
                    if score < best_score:
                        best_score = score
                        best_choice = (to_loc, target_level, tcol, tdep, area_needs_mix)
                        best_ev_chain = candidate_ev_chain
                        best_ev_chain_group_id = candidate_ev_chain_group_id
                if _p2_eviction_accepted:
                    break  # 外来SKU退避チェーン採用済み → 外側(target_level)ループも抜ける

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
                chain_group_id=main_chain_id or (
                    f"p2drop_{secrets.token_hex(6)}" if any(k in move_reason for k in ("段下げ", "大幅段下げ")) else
                    f"p2consol_{secrets.token_hex(6)}" if any(k in move_reason for k in ("集約", "SKU集約")) else
                    f"p2near_{secrets.token_hex(6)}" if any(k in move_reason for k in ("入口側移動", "手前化")) else
                    f"p2other_{secrets.token_hex(6)}"
                ),
                execution_order=main_exec_order or 1,
            )
            
            # Track this move for future lot-mixing checks
            lookup_key = (str(to_loc), str(sku_val))
            if lookup_key not in planned_lots_by_loc_sku:
                planned_lots_by_loc_sku[lookup_key] = set()
            planned_lots_by_loc_sku[lookup_key].add(int(lot_key))
            
            # Add eviction chain moves first
            for ev_mv in best_ev_chain:
                moves.append(ev_mv)
            # Then add the main move
            moves.append(move)
            
            # Update shelf usage
            shelf_usage[from_loc] = max(0, shelf_usage.get(from_loc, 0) - need_vol)
            shelf_usage[to_loc] = shelf_usage.get(to_loc, 0) + need_vol

            # inv を更新（後続の移動判定に反映するため）
            inv.at[idx, "ロケーション"] = str(to_loc)
            inv.at[idx, "lv"] = int(target_level)
            inv.at[idx, "col"] = int(tcol)
            inv.at[idx, "dep"] = int(tdep)

            # _pass_skus_by_loc / _p2_sku_count 更新（移動先追加 + 移動元除去）
            _pass_skus_by_loc.setdefault(str(to_loc), set()).add(str(sku_val))
            _p2ck_from = (str(from_loc), str(sku_val))
            _p2ck_to = (str(to_loc), str(sku_val))
            _p2_sku_count[_p2ck_from] = _p2_sku_count.get(_p2ck_from, 1) - 1
            _p2_sku_count[_p2ck_to] = _p2_sku_count.get(_p2ck_to, 0) + 1
            _from_skus_p2 = _pass_skus_by_loc.get(str(from_loc))
            if _from_skus_p2 and _p2_sku_count.get(_p2ck_from, 0) <= 0:
                _from_skus_p2.discard(str(sku_val))
                if not _from_skus_p2:
                    del _pass_skus_by_loc[str(from_loc)]

            # _pass_lots_by_loc_sku 更新
            _p2_to_key = (str(to_loc), str(sku_val))
            _pass_lots_by_loc_sku.setdefault(_p2_to_key, set()).add(int(lot_key))
            _p2_from_key = (str(from_loc), str(sku_val))
            if _p2_from_key in _pass_lots_by_loc_sku:
                _pass_lots_by_loc_sku[_p2_from_key].discard(int(lot_key))
                if not _pass_lots_by_loc_sku[_p2_from_key]:
                    del _pass_lots_by_loc_sku[_p2_from_key]

            # inv_lots_by_loc_sku も更新（Pass-2内の後続チェック用）
            inv_lots_by_loc_sku.setdefault(_p2_to_key, set()).add(int(lot_key))
            if _p2_from_key in inv_lots_by_loc_sku:
                inv_lots_by_loc_sku[_p2_from_key].discard(int(lot_key))
                if not inv_lots_by_loc_sku[_p2_from_key]:
                    del inv_lots_by_loc_sku[_p2_from_key]

            # 退避チェーンの移動もマップに反映
            for ev_mv in best_ev_chain:
                _ev_sku = str(ev_mv.sku_id)
                _ev_to = str(ev_mv.to_loc)
                _ev_from = str(ev_mv.from_loc)
                _pass_skus_by_loc.setdefault(_ev_to, set()).add(_ev_sku)
                # カウンタ更新 + from-loc側の残存チェック
                _p2ev_from_k = (_ev_from, _ev_sku)
                _p2ev_to_k = (_ev_to, _ev_sku)
                _p2_sku_count[_p2ev_from_k] = _p2_sku_count.get(_p2ev_from_k, 1) - 1
                _p2_sku_count[_p2ev_to_k] = _p2_sku_count.get(_p2ev_to_k, 0) + 1
                _ev_from_skus = _pass_skus_by_loc.get(_ev_from)
                if _ev_from_skus and _p2_sku_count.get(_p2ev_from_k, 0) <= 0:
                    _ev_from_skus.discard(_ev_sku)
                    if not _ev_from_skus:
                        del _pass_skus_by_loc[_ev_from]
                _ev_to_k = (_ev_to, _ev_sku)
                _ev_from_k = (_ev_from, _ev_sku)
                if ev_mv.lot_date:
                    try:
                        _ev_lk = int(ev_mv.lot_date)
                    except (ValueError, TypeError):
                        _ev_lk = UNKNOWN_LOT_KEY
                else:
                    _ev_lk = UNKNOWN_LOT_KEY
                _pass_lots_by_loc_sku.setdefault(_ev_to_k, set()).add(_ev_lk)
                inv_lots_by_loc_sku.setdefault(_ev_to_k, set()).add(_ev_lk)
                if _ev_from_k in _pass_lots_by_loc_sku:
                    _pass_lots_by_loc_sku[_ev_from_k].discard(_ev_lk)
                if _ev_from_k in inv_lots_by_loc_sku:
                    inv_lots_by_loc_sku[_ev_from_k].discard(_ev_lk)

            # 二重移動防止
            _moved_indices.add(idx)
    
    # メインループ(Pass-2)で生成された集約移動を、集約フラグOFF時にフィルタ
    if not getattr(cfg, "enable_pass_consolidate", True):
        _before_filter = len(moves)
        # Pass-C以外で生成された集約reason移動を除去（チェーン単位）
        _consol_chains = set()
        for m in moves:
            cg = getattr(m, 'chain_group_id', '') or ''
            reason = str(m.reason or '')
            if cg.startswith('p2consol_'):
                _consol_chains.add(cg)
        if _consol_chains:
            moves = [m for m in moves if (getattr(m, 'chain_group_id', '') or '') not in _consol_chains]
            _log(f"Consolidation filter: removed {_before_filter - len(moves)} moves ({len(_consol_chains)} chains)")

    # Record Pass-2 statistics (main loop moves after Pass-0 and Pass-1)
    main_loop_moves = len(moves) - pass_stats["passC"] - pass_stats["pass0"] - pass_stats["pass1"]
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

                # 二重移動防止
                if idx in _moved_indices:
                    continue

                from_loc = str(row["ロケーション"])
                lv, col, dep = row.get("lv"), row.get("col"), row.get("dep")
                if pd.isna(lv) or pd.isna(col) or pd.isna(dep):
                    continue
                lv, col, dep = int(lv), int(col), int(dep)

                if lv == 1:
                    continue

                sku_val = str(row["商品ID"])
                if not ai_col_hints or sku_val not in ai_col_hints:
                    continue

                qty_cases = int(row["qty_cases_move"]) or 0
                if qty_cases <= 0:
                    continue

                need_vol = float(row["volume_each_case"]) * qty_cases
                lot_key = int(row.get("lot_key") or UNKNOWN_LOT_KEY)

                ai_recommended_cols = ai_col_hints.get(sku_val, [])
                if not ai_recommended_cols:
                    continue

                best_choice = None
                best_score = math.inf

                for target_level in [1, 2]:
                    if target_level >= lv:
                        continue

                    cand_locs = locs_by_level.get(target_level, [])
                    if can_receive_set is not None:
                        cand_locs = [loc for loc in cand_locs if loc != from_loc and loc in can_receive_set]
                    else:
                        cand_locs = [loc for loc in cand_locs if loc != from_loc]

                    for to_loc in cand_locs:
                        tlv, tcol, tdep = _parse_loc8(str(to_loc))

                        # AI推奨列のみ
                        if int(tcol) not in ai_recommended_cols:
                            continue

                        # 容量チェック
                        used = float(shelf_usage.get(to_loc, 0.0))
                        limit = cap_by_loc.get(to_loc, cap_limit) if cap_by_loc else cap_limit
                        if used + need_vol > limit:
                            continue

                        # 別SKU混在チェック
                        _p3_existing_skus = _pass_skus_by_loc.get(str(to_loc), set())
                        if _p3_existing_skus and not _p3_existing_skus <= {sku_val}:
                            continue

                        # ロット混在チェック
                        _p3_lsk = (str(to_loc), sku_val)
                        _p3_existing_lots = _pass_lots_by_loc_sku.get(_p3_lsk, set())
                        if _p3_existing_lots:
                            _p3_known = _p3_existing_lots - {UNKNOWN_LOT_KEY}
                            if _p3_known and int(lot_key) not in _p3_known:
                                continue
                        # planned_lots_by_loc_sku もチェック
                        _p3_planned = planned_lots_by_loc_sku.get(_p3_lsk, set())
                        if _p3_planned and int(lot_key) not in _p3_planned:
                            continue

                        # FIFO段ルールチェック
                        if _violates_lot_level_rule_fast(sku_val, lot_key, target_level, tcol, inv_lot_levels_by_sku_col):
                            continue

                        # 入数帯チェック
                        if getattr(cfg, "strict_pack_area", True):
                            allowed_cols = _allowed_cols_for_row(row, cfg)
                            if tcol not in allowed_cols:
                                continue

                        col_priority = ai_recommended_cols.index(int(tcol)) if int(tcol) in ai_recommended_cols else 999
                        score = col_priority * 100 + tlv * 10 + tdep

                        if score < best_score:
                            best_score = score
                            best_choice = (to_loc, target_level, tcol, tdep)

                if best_choice is not None:
                    to_loc, target_level, tcol, tdep = best_choice

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
                    # planned_lots_by_loc_sku 更新
                    _p3_key = (str(to_loc), sku_val)
                    planned_lots_by_loc_sku.setdefault(_p3_key, set()).add(int(lot_key))
                    # _pass_skus_by_loc 更新
                    _pass_skus_by_loc.setdefault(str(to_loc), set()).add(sku_val)
                    # _pass_lots_by_loc_sku 更新
                    _pass_lots_by_loc_sku.setdefault(_p3_key, set()).add(int(lot_key))
                    _p3_from_key = (str(from_loc), sku_val)
                    if _p3_from_key in _pass_lots_by_loc_sku:
                        _pass_lots_by_loc_sku[_p3_from_key].discard(int(lot_key))
                    # 二重移動防止
                    _moved_indices.add(idx)
        
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
    
    # --- Pass-4: 有益スワップ（空きロケ不要の直接交換）---
    # NOTE: 異なるSKU間のスワップはFIFOスコアを悪化させるため、現在無効化
    # 将来的にはSKU優先度ベースの判定が必要
    _p4_t0 = time.perf_counter()
    _p4_start = len(moves)
    _p4_swaps = 0
    _p4_max = 0  # 無効化
    if _p4_max > 0:
        _log(f"Starting Pass-4 beneficial swaps, budget remaining: {_p4_max}")
        # 移動済みでない行のみ対象
        _p4_candidates = inv[~inv.index.isin(_moved_indices)].copy()
        _p4_candidates = _p4_candidates[_p4_candidates["qty_cases_move"].fillna(0).astype(int) > 0]
        if "volume_each_case" in _p4_candidates.columns:
            _p4_candidates = _p4_candidates[_p4_candidates["volume_each_case"].fillna(0).astype(float) > 0]

        # 段ごとにグループ化
        _p4_by_level: Dict[int, List[Tuple[int, pd.Series]]] = {}
        for _p4_idx, _p4_row in _p4_candidates.iterrows():
            _p4_lv = int(_p4_row.get("lv") or 0)
            if _p4_lv <= 0:
                continue
            _p4_by_level.setdefault(_p4_lv, []).append((_p4_idx, _p4_row))

        # 上段(Lv3-4)のアイテムと下段(Lv1-2)のアイテムをスワップ
        _p4_swaps = 0
        _p4_checked = 0
        _p4_upper = []
        for _lv in [4, 3]:
            _p4_upper.extend(_p4_by_level.get(_lv, []))
        _p4_lower = []
        for _lv in [1, 2]:
            _p4_lower.extend(_p4_by_level.get(_lv, []))

        _p4_used_indices: Set[int] = set()
        _p4_used_locs: Set[str] = set()

        for _p4_u_idx, _p4_u_row in _p4_upper:
            if _p4_swaps * 2 >= _p4_max:
                break
            if _p4_u_idx in _p4_used_indices:
                continue
            _p4_u_loc = str(_p4_u_row["ロケーション"])
            if _p4_u_loc in _p4_used_locs:
                continue
            _p4_u_sku = str(_p4_u_row["商品ID"])
            _p4_u_lv = int(_p4_u_row["lv"])
            _p4_u_vol = float(_p4_u_row.get("volume_each_case") or 0) * int(_p4_u_row.get("qty_cases_move") or 0)
            _p4_u_lot_key = int(_p4_u_row.get("lot_key") or UNKNOWN_LOT_KEY)
            if _p4_u_lot_key == UNKNOWN_LOT_KEY:
                continue

            # blocked チェック
            if _blocked_dest_locs and _p4_u_loc in _blocked_dest_locs:
                continue

            _p4_checked += 1
            if _p4_checked > 2000:  # 探索上限
                break

            for _p4_l_idx, _p4_l_row in _p4_lower:
                if _p4_l_idx in _p4_used_indices:
                    continue
                _p4_l_loc = str(_p4_l_row["ロケーション"])
                if _p4_l_loc in _p4_used_locs or _p4_l_loc == _p4_u_loc:
                    continue
                _p4_l_sku = str(_p4_l_row["商品ID"])
                _p4_l_lv = int(_p4_l_row["lv"])
                _p4_l_vol = float(_p4_l_row.get("volume_each_case") or 0) * int(_p4_l_row.get("qty_cases_move") or 0)
                _p4_l_lot_key = int(_p4_l_row.get("lot_key") or UNKNOWN_LOT_KEY)

                # 同じSKUならスキップ（FIFOダイレクトスワップで処理済み）
                if _p4_u_sku == _p4_l_sku:
                    continue
                if _p4_l_lot_key == UNKNOWN_LOT_KEY:
                    continue
                if _blocked_dest_locs and _p4_l_loc in _blocked_dest_locs:
                    continue
                if can_receive_set and (_p4_u_loc not in can_receive_set or _p4_l_loc not in can_receive_set):
                    continue

                # スワップが有益か判定: 上段アイテムが下段へ、下段アイテムが上段へ
                # 条件: 上段アイテムの方がロットが古い(FIFO改善)、または段下げで改善
                # 簡易判定: Lv差が1以上あればスワップ価値あり
                if _p4_u_lv <= _p4_l_lv:
                    continue  # 上段の方が低いLvなら改善にならない

                # 容量チェック（交換後）
                _p4_u_used = float(shelf_usage.get(_p4_u_loc, 0.0))
                _p4_u_limit = cap_by_loc.get(_p4_u_loc, cap_limit) if cap_by_loc else cap_limit
                _p4_l_used = float(shelf_usage.get(_p4_l_loc, 0.0))
                _p4_l_limit = cap_by_loc.get(_p4_l_loc, cap_limit) if cap_by_loc else cap_limit
                _p4_u_after = _p4_u_used - _p4_u_vol + _p4_l_vol
                _p4_l_after = _p4_l_used - _p4_l_vol + _p4_u_vol
                if _p4_u_after > _p4_u_limit * 1.01 or _p4_l_after > _p4_l_limit * 1.01:
                    continue

                # foreign SKUチェック: 交換後、各ロケに1SKUのみ
                _p4_u_skus = _pass_skus_by_loc.get(_p4_u_loc, set())
                _p4_l_skus = _pass_skus_by_loc.get(_p4_l_loc, set())
                # 上段ロケに下段SKUが入る: 元のSKU(上段)が出て下段SKUが入る
                _p4_u_remaining = (_p4_u_skus - {_p4_u_sku}) | {_p4_l_sku}
                _p4_l_remaining = (_p4_l_skus - {_p4_l_sku}) | {_p4_u_sku}
                if len(_p4_u_remaining) > 1 or len(_p4_l_remaining) > 1:
                    continue  # 混在発生

                # ロット混在チェック
                _p4_ok = True
                if _pass_lots_by_loc_sku:
                    # 上段ロケに下段SKUのロットが入る
                    _p4_ul_key = (_p4_u_loc, _p4_l_sku)
                    _p4_ul_lots = _pass_lots_by_loc_sku.get(_p4_ul_key, set())
                    if _p4_ul_lots:
                        _p4_ul_known = _p4_ul_lots - {UNKNOWN_LOT_KEY}
                        if _p4_ul_known and _p4_l_lot_key not in _p4_ul_known:
                            _p4_ok = False
                    # 下段ロケに上段SKUのロットが入る
                    _p4_lu_key = (_p4_l_loc, _p4_u_sku)
                    _p4_lu_lots = _pass_lots_by_loc_sku.get(_p4_lu_key, set())
                    if _p4_lu_lots:
                        _p4_lu_known = _p4_lu_lots - {UNKNOWN_LOT_KEY}
                        if _p4_lu_known and _p4_u_lot_key not in _p4_lu_known:
                            _p4_ok = False
                if not _p4_ok:
                    continue

                # === 有益スワップ実行 ===
                _p4_cg = f"swap_{secrets.token_hex(6)}"
                _p4_u_date = _lot_key_to_datestr8(_p4_u_lot_key)
                _p4_l_date = _lot_key_to_datestr8(_p4_l_lot_key)

                # Move 1: 下段アイテムを上段ロケへ
                moves.append(Move(
                    sku_id=_p4_l_sku,
                    lot=str(_p4_l_row.get("ロット") or ""),
                    qty=int(_p4_l_row.get("qty_cases_move") or 0),
                    from_loc=str(_p4_l_loc).zfill(8),
                    to_loc=str(_p4_u_loc).zfill(8),
                    lot_date=_p4_l_date,
                    reason=f"有益スワップ → Lv{_p4_l_lv}→{_p4_u_lv}(保管段へ移動、取口を高優先品に譲渡)",
                    chain_group_id=_p4_cg,
                    execution_order=1,
                ))
                # Move 2: 上段アイテムを下段ロケへ
                moves.append(Move(
                    sku_id=_p4_u_sku,
                    lot=str(_p4_u_row.get("ロット") or ""),
                    qty=int(_p4_u_row.get("qty_cases_move") or 0),
                    from_loc=str(_p4_u_loc).zfill(8),
                    to_loc=str(_p4_l_loc).zfill(8),
                    lot_date=_p4_u_date,
                    reason=f"有益スワップ → Lv{_p4_u_lv}→{_p4_l_lv}(段下げ、取口配置で効率化)",
                    chain_group_id=_p4_cg,
                    execution_order=2,
                ))

                # 状態更新
                shelf_usage[_p4_u_loc] = max(0.0, _p4_u_after)
                shelf_usage[_p4_l_loc] = max(0.0, _p4_l_after)
                _pass_skus_by_loc[_p4_u_loc] = _p4_u_remaining
                _pass_skus_by_loc[_p4_l_loc] = _p4_l_remaining
                _p4_used_indices.add(_p4_u_idx)
                _p4_used_indices.add(_p4_l_idx)
                _p4_used_locs.add(_p4_u_loc)
                _p4_used_locs.add(_p4_l_loc)
                _moved_indices.add(_p4_u_idx)
                _moved_indices.add(_p4_l_idx)
                _p4_swaps += 1
                break  # 次の上段アイテムへ

    _p4_moves = len(moves) - _p4_start
    pass_stats["pass4_swap"] = _p4_moves
    _log(f"Pass-4 beneficial swaps: {_p4_swaps} swaps ({_p4_moves} moves) in {time.perf_counter()-_p4_t0:.2f}s, total={len(moves)}")

    # --- Safety cap already applied during candidate selection ---
    # moves list is already sorted by Pass priority and limited to max_moves

    try:
        logger.debug(f"[optimizer] planned moves={len(moves)} (limit={getattr(cfg,'max_moves',None)})")
        _publish_progress(get_current_trace_id(), {
            "type": "planned", "count": len(moves),
            "message": f"移動案作成完了: {len(moves)}件（Pass優先度順ソート済み）"
        })
    except Exception:
        pass

    # Auto-apply hard constraints and record debug counters
    if getattr(cfg, "auto_enforce", True):
        logger.debug(f"[optimizer] Auto-enforce is enabled, starting constraint enforcement")
        try:
            lm_df = locals().get("lm_scoped", loc_master)
        except Exception as e:
            logger.debug(f"[optimizer] Failed to get lm_scoped: {e}")
            lm_df = loc_master
        try:
            _publish_progress(get_current_trace_id(), {
                "type": "info", "phase": "enforce",
                "message": f"制約チェック開始: {len(moves)}件を検証中..."
            })
        except Exception as e:
            logger.debug(f"[optimizer] Failed to publish enforce start progress: {e}")
            pass
        
        # --- 依存関係に基づくmoveの並び替え ---
        # move A が loc X から出発（SKUを空ける）し、move B が loc X に到着する場合、
        # A を B より先に処理しないと enforce が foreign_sku で却下する。
        # チェーングループ内は execution_order で正しく並んでいるが、
        # 異なるパス間の暗黙的依存は並びが保証されていない。
        _loc_clearers: Dict[str, List[int]] = {}  # loc → [moveのindex]
        for _si, _sm in enumerate(moves):
            _loc_clearers.setdefault(str(_sm.from_loc).zfill(8), []).append(_si)

        # 依存グラフ構築:
        # 1) move[j]のto_locを他のmove[i]がfrom_locで空ける場合、i→jの依存
        # 2) 同一chain_group内ではexecution_orderの小さい方が先（退避→配置の順序保証）
        _n = len(moves)
        _in_degree = [0] * _n
        _adj: Dict[int, List[int]] = {i: [] for i in range(_n)}
        _added_edges: Set[Tuple[int, int]] = set()

        def _add_edge(src: int, dst: int) -> None:
            if src != dst and (src, dst) not in _added_edges:
                _adj[src].append(dst)
                _in_degree[dst] += 1
                _added_edges.add((src, dst))

        # 1) from_loc/to_loc ベースの依存
        for j, _sm in enumerate(moves):
            _to = str(_sm.to_loc).zfill(8)
            for i in _loc_clearers.get(_to, []):
                _add_edge(i, j)

        # 2) chain_group_id 内の execution_order 依存
        # 退避チェーン/FIFOスワップでは退避(exec=1)が配置(exec=2)より先に実行される必要がある
        _chain_group_indices: Dict[str, List[Tuple[int, int]]] = {}  # chain_id → [(exec_order, index)]
        for _ci, _cm in enumerate(moves):
            _cg = getattr(_cm, 'chain_group_id', None)
            _eo = getattr(_cm, 'execution_order', None)
            if _cg and _eo is not None:
                _cg_str = str(_cg)
                if _cg_str and _cg_str != 'nan':
                    _chain_group_indices.setdefault(_cg_str, []).append((int(_eo), _ci))
        for _cg_id, _members in _chain_group_indices.items():
            _members_sorted = sorted(_members, key=lambda x: x[0])
            for _k in range(len(_members_sorted) - 1):
                _add_edge(_members_sorted[_k][1], _members_sorted[_k + 1][1])

        # トポロジカルソート（Kahn's algorithm）- 安定ソート（元の順序を維持）
        _queue = deque(i for i in range(_n) if _in_degree[i] == 0)
        _topo_order: List[int] = []
        while _queue:
            _node = _queue.popleft()
            _topo_order.append(_node)
            for _nb in _adj[_node]:
                _in_degree[_nb] -= 1
                if _in_degree[_nb] == 0:
                    _queue.append(_nb)
        # 循環がある場合は残りを末尾に追加
        if len(_topo_order) < _n:
            _remaining = [i for i in range(_n) if i not in set(_topo_order)]
            _topo_order.extend(_remaining)
        moves = [moves[i] for i in _topo_order]
        logger.debug(f"[optimizer] Topologically sorted {_n} moves for enforce (cycles={_n - len(_topo_order) + len([i for i in range(_n) if i not in set(_topo_order[:_n])])})")

        logger.debug(f"[optimizer] Calling enforce_constraints with {len(moves)} moves")
        # _original_skus_by_loc（品質フィルタ前）を使用する。
        # 品質フィルタ後（_filtered_skus_by_loc）では出荷止め・外装不良・販促物SKUや
        # 良品 qty=0 の SKU が見えず、移動先に別SKUが存在する違反が漏れる。
        # blocked_dest_locs で非対象品質ロケは line 1720 で先にブロックされるため、
        # _original_skus_by_loc を使っても追加の大量却下は発生しない。
        accepted = enforce_constraints(
            sku_master=sku_master,
            inventory=inv,
            moves=moves,
            cfg=cfg,
            loc_master=lm_df,
            original_skus_by_loc=_original_skus_by_loc,
            original_qty_by_loc_sku=_original_qty_by_loc_sku_full,
            blocked_dest_locs=_blocked_dest_locs,
            blocked_source_locs=_multi_sku_source_blocked if _multi_sku_source_blocked else None,
            original_inv_lot_levels=_original_inv_lot_levels,
            original_empty_locs=_original_receivable_locs if getattr(cfg, "allow_empty_loc_multi_sku", True) else None,
        )
        logger.debug(f"[optimizer] enforce_constraints completed: {len(accepted)} accepted out of {len(moves)}")

        # --- 最終リグレッションゲート: 移動前より悪化しないことを保証 ---
        try:
            _pre_gate = len(accepted)
            accepted = _final_regression_gate(
                accepted,
                _original_inv_lot_levels,
                _original_lot_strings_by_loc_sku,
                original_skus_by_loc=_original_skus_by_loc,
                original_qty_by_loc_sku=_original_qty_by_loc_sku_full,
            )
            if len(accepted) < _pre_gate:
                logger.debug(f"[optimizer] Regression gate removed {_pre_gate - len(accepted)} moves: {_pre_gate} -> {len(accepted)}")
            else:
                logger.debug(f"[optimizer] Regression gate: all {len(accepted)} moves passed")
        except Exception as e:
            logger.warning(f"[optimizer] WARNING: Regression gate failed: {e}")
            import traceback
            logger.warning(traceback.format_exc())

        # Resolve move dependencies (to_loc/from_loc conflicts)
        try:
            accepted = _resolve_move_dependencies(
                accepted,
                original_qty_by_loc_sku=_original_qty_by_loc_sku_full,
            )
            logger.warning(f"[optimizer] Dependency resolution completed: {len(accepted)} moves")
        except Exception as e:
            logger.debug(f"[optimizer] WARNING: Dependency resolution failed: {e}")
            import traceback
            traceback.print_exc()

        # --- dep_チェーン内の「集約(同一ロット統合)」reason再評価 ---
        # dep_チェーン結合後、先行移動が to_loc を空にするため実際には集約にならない
        # ケースがある。そのようなreasonを位置関係ベースの正確な記述に修正する。
        try:
            _reason_fix_count = 0
            # チェーンごとにグループ化
            _dep_chain_groups: Dict[str, List[int]] = defaultdict(list)
            for _ri, _rm in enumerate(accepted):
                _cg = getattr(_rm, 'chain_group_id', None) or ""
                if _cg.startswith("dep_"):
                    _dep_chain_groups[_cg].append(_ri)

            for _cg_id, _indices in _dep_chain_groups.items():
                if len(_indices) < 2:
                    continue
                # チェーン内の移動を execution_order でソート
                _chain_moves = sorted(_indices, key=lambda i: (accepted[i].execution_order or 0))
                # 同一SKU+ロットのペアを検出
                for _ci_a in range(len(_chain_moves)):
                    _ia = _chain_moves[_ci_a]
                    _ma = accepted[_ia]
                    for _ci_b in range(_ci_a + 1, len(_chain_moves)):
                        _ib = _chain_moves[_ci_b]
                        _mb = accepted[_ib]
                        # move_a(eo小).from_loc == move_b(eo大).to_loc
                        # かつ同一SKU+ロット
                        if (str(_ma.from_loc).zfill(8) == str(_mb.to_loc).zfill(8)
                                and _ma.sku_id == _mb.sku_id
                                and _ma.lot == _mb.lot
                                and _mb.reason
                                and ("集約" in _mb.reason)):
                            # move_a が先に実行されて from_loc を空にするため「集約」は不正確
                            # 位置関係から適切なreasonを再生成
                            _from8 = str(_mb.from_loc).zfill(8)
                            _to8 = str(_mb.to_loc).zfill(8)
                            _f_lv, _f_col, _f_dep = _parse_loc8(_from8)
                            _t_lv, _t_col, _t_dep = _parse_loc8(_to8)

                            _new_actions = []
                            _new_improvements = []

                            if _t_lv != _f_lv:
                                if _t_lv < _f_lv:
                                    _new_actions.append(f"段下げ(Lv{_f_lv}→{_t_lv})")
                                    _new_improvements.append("下段活用")
                                else:
                                    _new_actions.append(f"段上げ(Lv{_f_lv}→{_t_lv})")
                                    _new_improvements.append("空間効率化")

                            if _t_col != _f_col:
                                _new_actions.append(f"エリア移動(列{_f_col}→{_t_col})")
                                _new_improvements.append("エリア内バランス")

                            if _t_dep != _f_dep and _t_lv == _f_lv and _t_col == _f_col:
                                if _t_dep < _f_dep:
                                    _new_actions.append(f"手前化(奥{_f_dep}→{_t_dep})")
                                    _new_improvements.append("取出し容易化")
                                else:
                                    _new_actions.append(f"同段内移動")
                                    _new_improvements.append("配置最適化")

                            if not _new_actions:
                                _new_actions.append("ロケーション移動")
                                _new_improvements.append("配置最適化")

                            _new_reason = " & ".join(_new_actions)
                            if _new_improvements:
                                _new_reason += " → " + "、".join(_new_improvements)

                            # reasonフィールドを書き換え
                            accepted[_ib] = Move(
                                sku_id=_mb.sku_id,
                                lot=_mb.lot,
                                qty=_mb.qty,
                                from_loc=_mb.from_loc,
                                to_loc=_mb.to_loc,
                                lot_date=_mb.lot_date,
                                reason=_new_reason,
                                chain_group_id=_mb.chain_group_id,
                                execution_order=_mb.execution_order,
                                distance=_mb.distance,
                            )
                            _reason_fix_count += 1
                            logger.debug(f"[optimizer] dep_chain reason fix: chain={_cg_id} "
                                         f"move eo={_mb.execution_order} '{_mb.reason}' -> '{_new_reason}'")

            if _reason_fix_count > 0:
                logger.warning(f"[optimizer] dep_chain reason re-evaluation: fixed {_reason_fix_count} moves")
        except Exception as e:
            logger.warning(f"[optimizer] WARNING: dep_chain reason re-evaluation failed: {e}")
            import traceback
            logger.warning(traceback.format_exc())

        # 最終実行順序でSKU/ロット混在を再検証
        # _original_skus_by_loc（品質フィルタ前）を使用する。
        # enforce_constraints では _filtered_skus_by_loc（品質フィルタ後）を使い、
        # blocked_dest_locs で非対象品質ロケをブロックしているが、
        # _resolve_move_dependencies で並び替え後の実行順序では、
        # 退避moveが先行しない場合に良品SKU混在が漏れる。
        # _post_resolution_validate は最終安全ネットなので、
        # 品質フィルタ前の完全なSKUマップで検証する。
        # _post_resolution_validate はリグレッションゲートと重複するため無効化。
        # enforce_constraints + リグレッションゲート + safety-net で品質を保証する。
        logger.warning(f"[optimizer] Post-resolution validation skipped (handled by regression gate): {len(accepted)} moves")

        # --- 混在ロケ移動の最終除去 ---
        # _resolve_move_dependencies で依存チェーン再構成後に混在ロケからの移動が
        # 残っている場合があるため、最終チェックで除去する
        # from_loc: 2SKU以上（_multi_sku_source_blocked）からの移動は全て除去
        # to_loc: 3SKU以上（multi_sku_locs_strict）への移動は全て除去
        _final_blocked_src = _multi_sku_source_blocked
        _final_blocked_dst = multi_sku_locs_strict
        if _final_blocked_src or _final_blocked_dst:
            _pre_blocked = len(accepted)
            _blocked_chains: Set[str] = set()
            _blocked_indices: Set[int] = set()  # チェーンなし移動のインデックス追跡
            # 1st pass: 混在ロケ移動を持つチェーン/インデックスを特定
            for _idx_m, m in enumerate(accepted):
                _fl = str(m.from_loc).zfill(8)
                _tl = str(m.to_loc).zfill(8)
                if _fl in _final_blocked_src or _tl in _final_blocked_dst:
                    _cg = getattr(m, 'chain_group_id', None)
                    if _cg:
                        _blocked_chains.add(_cg)
                    else:
                        _blocked_indices.add(_idx_m)
            # 2nd pass: ブロックされたチェーン全体 + チェーンなし移動を除去
            if _blocked_chains or _blocked_indices:
                accepted = [m for _idx_m, m in enumerate(accepted)
                            if _idx_m not in _blocked_indices and
                            not (getattr(m, 'chain_group_id', None) and
                                 m.chain_group_id in _blocked_chains)]
                logger.warning(f"[optimizer] 混在ロケ最終除去: {_pre_blocked - len(accepted)}件除去 ({len(_blocked_chains)}チェーン, {len(_blocked_indices)}チェーンなし)")

        # --- 最終安全ネット: リグレッションゲートで品質保証済みのため無効化 ---
        logger.warning(f"[optimizer] Safety-net skipped: {len(accepted)} moves")
        try:
            pass  # Safety-net enabled
            _pre_count = len(accepted)
            _sn_sim_skus: Dict[str, Set[str]] = {k: set(v) for k, v in _original_skus_by_loc.items()}
            _sn_sim_qty: Dict[Tuple[str, str], float] = dict(_original_qty_by_loc_sku_full)
            _sn_sim_lots: Dict[Tuple[str, str], Set[str]] = {k: set(v) for k, v in _original_lot_strings_by_loc_sku.items()}
            _sn_sim_vol: Dict[str, float] = dict(shelf_usage)
            _sn_rejected_chains: Set[str] = set()
            _sn_accepted: List[Move] = []

            # 入力順（依存解決済み）で処理: スワップはメンバーが揃った時点でアトミック処理
            _sn_swap_members: Dict[str, List[Move]] = {}
            _sn_swap_ids: Set[str] = set()
            for m in accepted:
                _cg_raw = getattr(m, 'chain_group_id', None)
                _cg_str = str(_cg_raw) if _cg_raw is not None and not (isinstance(_cg_raw, float) and _cg_raw != _cg_raw) else ""
                if _cg_str.startswith("swap_") or _cg_str.startswith("fifo_direct_") or _cg_str.startswith("zone_swap_"):
                    _sn_swap_members.setdefault(_cg_str, []).append(m)
                    _sn_swap_ids.add(_cg_str)

            _sn_processed_swaps: Set[str] = set()
            for m in accepted:
                _cg_raw = getattr(m, 'chain_group_id', None)
                cg = str(_cg_raw) if _cg_raw is not None and not (isinstance(_cg_raw, float) and _cg_raw != _cg_raw) else ""

                if cg in _sn_swap_ids:
                    # スワップグループ: 既に処理済みならスキップ
                    if cg in _sn_processed_swaps:
                        continue
                    _sn_processed_swaps.add(cg)

                    if cg in _sn_rejected_chains:
                        continue

                    _sn_sg_members = _sn_swap_members.get(cg, [])
                    if len(_sn_sg_members) != 2:
                        _sn_rejected_chains.add(cg)
                        continue
                    _sn_m1, _sn_m2 = _sn_sg_members[0], _sn_sg_members[1]
                    _sn_sku1, _sn_sku2 = str(_sn_m1.sku_id), str(_sn_m2.sku_id)
                    _sn_to1, _sn_to2 = str(_sn_m1.to_loc).zfill(8), str(_sn_m2.to_loc).zfill(8)
                    _sn_from1, _sn_from2 = str(_sn_m1.from_loc).zfill(8), str(_sn_m2.from_loc).zfill(8)

                    # 移動元在庫チェック（スワップ）
                    _sn_avail1 = _sn_sim_qty.get((_sn_from1, _sn_sku1), 0.0)
                    _sn_avail2 = _sn_sim_qty.get((_sn_from2, _sn_sku2), 0.0)
                    if _sn_avail1 < float(_sn_m1.qty) - 1e-9 or _sn_avail2 < float(_sn_m2.qty) - 1e-9:
                        logger.debug(f"[safety-net] reject swap no_source: {cg} avail1={_sn_avail1:.1f} avail2={_sn_avail2:.1f}")
                        _sn_rejected_chains.add(cg)
                        continue

                    # アトミックforeign SKUチェック（enforce_constraintsと同じ無条件discard方式）
                    _sn_swap_skus = {_sn_sku1, _sn_sku2}
                    _sn_at_to1 = set(_sn_sim_skus.get(_sn_to1, set()))
                    _sn_at_to2 = set(_sn_sim_skus.get(_sn_to2, set()))

                    # 第三のSKU（スワップ対象でないSKU）がいたら却下
                    _sn_third1 = _sn_at_to1 - _sn_swap_skus
                    _sn_third2 = _sn_at_to2 - _sn_swap_skus
                    if _sn_third1 or _sn_third2:
                        _sn_rejected_chains.add(cg)
                        continue

                    # スワップ後: 交換元SKUを無条件discard、交換先SKUを追加
                    _sn_after1 = set(_sn_at_to1)
                    _sn_after1.discard(_sn_sku2)
                    _sn_after1.add(_sn_sku1)
                    _sn_after2 = set(_sn_at_to2)
                    _sn_after2.discard(_sn_sku1)
                    _sn_after2.add(_sn_sku2)

                    if len(_sn_after1) > 1 or len(_sn_after2) > 1:
                        _sn_rejected_chains.add(cg)
                        continue

                    # ロット混在チェック
                    # 同一SKUスワップは双方向交換のため相手ロットが同時退出する → チェックスキップ
                    _sn_lot_ok = True
                    if _sn_sku1 != _sn_sku2:
                        for _sn_m in [_sn_m1, _sn_m2]:
                            _sn_lot_v = str(_sn_m.lot) if _sn_m.lot else ""
                            if _sn_lot_v and _sn_lot_v not in ("nan", "None"):
                                _sn_lsk = (str(_sn_m.to_loc).zfill(8), str(_sn_m.sku_id))
                                _sn_ex_lots = _sn_sim_lots.get(_sn_lsk, set())
                                if _sn_ex_lots and _sn_lot_v not in _sn_ex_lots:
                                    _sn_lot_ok = False; break
                    if not _sn_lot_ok:
                        _sn_rejected_chains.add(cg)
                        continue

                    # 受理 & 状態更新
                    for _sn_m in [_sn_m1, _sn_m2]:
                        _sn_accepted.append(_sn_m)
                        _sn_s = str(_sn_m.sku_id)
                        _sn_t = str(_sn_m.to_loc).zfill(8)
                        _sn_f = str(_sn_m.from_loc).zfill(8)
                        _sn_q = float(_sn_m.qty)
                        _sn_l = str(_sn_m.lot) if _sn_m.lot else ""
                        _sn_sim_skus.setdefault(_sn_t, set()).add(_sn_s)
                        _sn_sim_qty[(_sn_t, _sn_s)] = _sn_sim_qty.get((_sn_t, _sn_s), 0.0) + _sn_q
                        _sn_sim_qty[(_sn_f, _sn_s)] = _sn_sim_qty.get((_sn_f, _sn_s), 0.0) - _sn_q
                        if _sn_sim_qty.get((_sn_f, _sn_s), 0.0) <= 0:
                            _sn_fs = _sn_sim_skus.get(_sn_f)
                            if _sn_fs:
                                _sn_fs.discard(_sn_s)
                                if not _sn_fs:
                                    del _sn_sim_skus[_sn_f]
                        if _sn_l and _sn_l not in ("nan", "None"):
                            _sn_sim_lots.setdefault((_sn_t, _sn_s), set()).add(_sn_l)
                            _sn_fk = (_sn_f, _sn_s)
                            if _sn_fk in _sn_sim_lots:
                                _sn_sim_lots[_sn_fk].discard(_sn_l)
                    # アトミックSKU更新
                    _sn_sim_skus[_sn_to1] = _sn_after1
                    _sn_sim_skus[_sn_to2] = _sn_after2
                    continue

                # 非スワップmoveの処理
                to_loc = str(m.to_loc).zfill(8)
                from_loc = str(m.from_loc).zfill(8)
                sku = str(m.sku_id)
                qty = float(m.qty)
                lot = str(m.lot) if m.lot else ""
                if lot == "nan" or lot == "None":
                    lot = ""

                if cg and cg in _sn_rejected_chains:
                    continue

                # 移動元在庫チェック: inv変更がロールバックされずに残った幻の在庫を検出
                _sn_avail = _sn_sim_qty.get((from_loc, sku), 0.0)
                if _sn_avail < qty - 1e-9:
                    logger.debug(f"[safety-net] reject no_source: {sku}@{from_loc} need={qty} have={_sn_avail:.1f} chain={cg}")
                    if cg:
                        _sn_rejected_chains.add(cg)
                    continue

                # foreign SKU チェック
                _existing = _sn_sim_skus.get(to_loc, set())
                if _existing and not _existing <= {sku}:
                    # 複数SKU同居が許可された移動は安全ネットでも通過させる
                    # 条件: フィーチャー有効 AND 元々空きロケ AND 許可プレフィックス一致
                    _sn_is_allowed_pass = False
                    if getattr(cfg, "allow_empty_loc_multi_sku", True):
                        _sn_allowed_prefixes = tuple(getattr(cfg, "multi_sku_allowed_chain_prefixes",
                            ("p1fifo", "swap_fifo_", "fifo_direct_", "p0rebal_", "p2consol_")))
                        _sn_is_originally_empty = to_loc in _original_receivable_locs
                        _sn_is_allowed_pass = bool(
                            _sn_is_originally_empty and cg and any(cg.startswith(p) for p in _sn_allowed_prefixes)
                        )
                    if not _sn_is_allowed_pass:
                        logger.debug(f"[safety-net] reject foreign_sku: {sku}->{to_loc} existing={_existing - {sku}} chain={cg}")
                        if cg:
                            _sn_rejected_chains.add(cg)
                        continue

                # ロット混在チェック
                if lot:
                    _lsk = (to_loc, sku)
                    _existing_lots = _sn_sim_lots.get(_lsk, set())
                    if _existing_lots and lot not in _existing_lots:
                        logger.debug(f"[safety-net] reject lot_mix: {sku}->{to_loc} lot={lot} existing={_existing_lots} chain={cg}")
                        if cg:
                            _sn_rejected_chains.add(cg)
                        continue

                # level volume cap チェック
                _sn_tlv = _parse_loc8(to_loc)[0]
                _sn_add_vol = float(sku_vol_map.get(sku, 0.0) or 0.0) * qty
                _sn_level_cap = _get_level_vol_cap(_sn_tlv, cfg)
                if _sn_level_cap is not None and _sn_sim_vol.get(to_loc, 0.0) + _sn_add_vol > _sn_level_cap:
                    logger.debug(f"[safety-net] reject level_vol_cap: {sku}->{to_loc} vol={_sn_sim_vol.get(to_loc, 0.0):.3f}+{_sn_add_vol:.3f} > {_sn_level_cap:.3f} chain={cg}")
                    if cg:
                        _sn_rejected_chains.add(cg)
                    continue

                _sn_accepted.append(m)
                _sn_sim_skus.setdefault(to_loc, set()).add(sku)
                _sn_sim_qty[(to_loc, sku)] = _sn_sim_qty.get((to_loc, sku), 0.0) + qty
                _sn_sim_qty[(from_loc, sku)] = _sn_sim_qty.get((from_loc, sku), 0.0) - qty
                if _sn_sim_qty.get((from_loc, sku), 0.0) <= 0:
                    _from_set = _sn_sim_skus.get(from_loc)
                    if _from_set:
                        _from_set.discard(sku)
                        if not _from_set:
                            del _sn_sim_skus[from_loc]
                if lot:
                    _sn_sim_lots.setdefault((to_loc, sku), set()).add(lot)
                    _from_lsk = (from_loc, sku)
                    if _from_lsk in _sn_sim_lots:
                        _sn_sim_lots[_from_lsk].discard(lot)
                _sn_sim_vol[to_loc] = _sn_sim_vol.get(to_loc, 0.0) + _sn_add_vol
                _sn_sim_vol[from_loc] = max(0.0, _sn_sim_vol.get(from_loc, 0.0) - _sn_add_vol)

            accepted = _sn_accepted
            _sn_rejected = _pre_count - len(accepted)
            logger.warning(f"[optimizer] Safety-net validation: {_pre_count} -> {len(accepted)} moves (rejected {_sn_rejected}, chains={len(_sn_rejected_chains)})")
        except StopIteration:
            pass  # Safety-net disabled
        except Exception as e:
            logger.debug(f"[optimizer] WARNING: Safety-net validation failed: {e}")
            import traceback
            traceback.print_exc()

        # Post-Pass-1 FIFO直接スワップを最終追加
        # enforce_constraints/混在ロケ除去/safety-netの全ゲート通過後に追加。
        # ミニパス内で容量・ロット混在・blocked・can_receive チェック済みのため安全。
        if _fifo_direct_bypass:
            accepted.extend(_fifo_direct_bypass)
            logger.warning(f"[optimizer] Added {len(_fifo_direct_bypass)} FIFO direct swap moves (post-gate bypass)")

        # Generate comprehensive summary report
        rejected_count = len(moves) - len(accepted)
        try:
            logger.debug(f"[optimizer] Generating summary report: accepted={len(accepted)}, rejected={rejected_count}")
            summary_report = _generate_relocation_summary(
                inv_before=inv,
                moves=accepted,
                rejected_count=rejected_count,
                pass_stats=pass_stats,
            )
            logger.debug(f"[optimizer] Summary report generated: {list(summary_report.keys())}")
            
            # Format report as readable text
            report_lines = []
            report_lines.append("=" * 70)
            report_lines.append("📊 リロケーション結果の総合評価")
            report_lines.append("=" * 70)
            report_lines.append("")
            report_lines.append("【実施結果】")
            report_lines.append(f"  計画移動数: {summary_report.get('total_planned', 0):,} 件")
            report_lines.append(f"  承認移動数: {summary_report.get('total_accepted', 0):,} 件")
            final_mc = summary_report.get('final_move_count')
            if final_mc is not None and final_mc != summary_report.get('total_accepted', 0):
                report_lines.append(f"  最終移動数: {final_mc:,} 件（退避チェーン統合後）")
            report_lines.append(f"  却下移動数: {summary_report.get('total_rejected', 0):,} 件")
            report_lines.append(f"  移動率: {summary_report.get('move_rate_percent', 0):.1f}%")
            report_lines.append(f"  影響SKU数: {summary_report.get('affected_skus', 0):,} 種類")
            report_lines.append(f"  総ケース数: {summary_report.get('total_cases', 0):,} ケース")
            report_lines.append("")
            
            # === New evaluation metrics ===
            report_lines.append("【最適化効果】")
            
            # Pass-wise breakdown (実行順: Pass-C → Pass-1 → Pass-0 → Pass-2 → Pass-3)
            pass_stats_from_summary = summary_report.get('pass_stats', {})
            if pass_stats_from_summary and any(pass_stats_from_summary.values()):
                report_lines.append("  ▶ Pass毎の改善 (実行順):")
                if pass_stats_from_summary.get("passC", 0) > 0:
                    report_lines.append(f"    • Pass-C (同一ロット集約) 【最優先】: {pass_stats_from_summary['passC']:,}件")
                    report_lines.append(f"      → 同一SKU×ロットの分散在庫を1箇所に集約")
                if pass_stats_from_summary.get("pass1", 0) > 0:
                    report_lines.append(f"    • Pass-1 (取口/保管整列) 【第2優先】: {pass_stats_from_summary['pass1']:,}件")
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
            logger.debug(f"[optimizer] Report formatted, {len(report_lines)} lines")
            
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
            logger.debug(f"[optimizer] Failed to generate summary report: {e}")
            logger.debug(f"[optimizer] Traceback: {traceback.format_exc()}")
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
                logger.debug(f"[optimizer] Failed to publish summary report: {e}")
        
        # Cache moves for timeout recovery
        current_trace = get_current_trace_id()
        if current_trace and accepted:
            moves_data = [m.to_dict() if hasattr(m, 'to_dict') else m.__dict__ for m in accepted]
            cache_moves(current_trace, moves_data)
            logger.debug(f"[optimizer] Cached {len(moves_data)} moves for trace_id={current_trace}")

        return accepted


# ---------------------------------------------------------------------------
# score_all_inventory: 全在庫スコアリング（移動なし・現状課題の可視化）
# ---------------------------------------------------------------------------

def score_all_inventory(
    sku_master: pd.DataFrame,
    inventory: pd.DataFrame,
    *,
    cfg: OptimizerConfig | None = None,
    loc_master: Optional[pd.DataFrame] = None,
    block_filter: Iterable[str] | None = None,
    quality_filter: Iterable[str] | None = None,
) -> dict:
    """
    全在庫をスコアリングして現状の最適化課題を可視化する（移動は行わない）。

    課題の種別:
      - fifo_issue: 最古ロット(lot_rank=1)が取り口（段1-2）にない
      - swap_needed: 取り口にいるが最古ロットではない（同SKUの最古が保管にある）
      - packband_issue: 入数帯エリア（列ゾーン）が間違っている

    Returns
    -------
    dict
        inventory_scores: 在庫ごとのスコア・課題フラグ（total_issue_score降順）
        summary: 問題件数の集計
    """
    cfg = cfg or OptimizerConfig()
    pick_levels = tuple(getattr(cfg, "pick_levels", (1, 2)))
    pack_mismatch_penalty = float(getattr(cfg, "pack_mismatch_penalty", 400.0))

    # --- 在庫データ正規化 ---
    inv = inventory.copy()
    if "ブロック略称" not in inv.columns and "block_code" in inv.columns:
        inv["ブロック略称"] = inv["block_code"]
    if "quality_name" not in inv.columns and "品質区分名" in inv.columns:
        inv["quality_name"] = inv["品質区分名"].astype(str)

    # フィルタ適用
    if block_filter is not None:
        inv = inv[inv["ブロック略称"].isin(list(block_filter))].copy()
        inv = inv.reset_index(drop=True)
    if quality_filter is not None:
        q_col = "quality_name" if "quality_name" in inv.columns else "品質区分名"
        if q_col in inv.columns:
            inv = inv[inv[q_col].isin(list(quality_filter))].copy()
            inv = inv.reset_index(drop=True)

    # プレースホルダロケは除外
    inv = inv[~inv["ロケーション"].astype(str).isin(PLACEHOLDER_LOCS)].copy()
    inv = inv.reset_index(drop=True)

    # ロケーションを8桁ゼロパディング
    inv["ロケーション"] = inv["ロケーション"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)

    # ロケマスタフィルタ
    if loc_master is not None and not loc_master.empty:
        lm_scoped = _filter_loc_master_by_block_quality(loc_master, block_filter, quality_filter)
        cap_by_loc, _ = _cap_map_from_master(lm_scoped, getattr(cfg, "fill_rate", DEFAULT_FILL_RATE))
        valid_locs_set = set(cap_by_loc.keys())
        inv = inv[inv["ロケーション"].isin(valid_locs_set)].copy()
        inv = inv.reset_index(drop=True)

    # 引当在庫は除外
    # B4: fillna(1) で変換不能値を安全側（引当あり=除外）として扱う
    if "引当数" in inv.columns:
        inv = inv[pd.to_numeric(inv["引当数"], errors="coerce").fillna(1) <= 0].copy()
        inv = inv.reset_index(drop=True)

    if inv.empty:
        return {
            "inventory_scores": [],
            "summary": {
                "total_items": 0,
                "fifo_issues": 0,
                "swap_needed": 0,
                "packband_issues": 0,
                "items_with_issues": 0,
                "skus_with_oldest_not_at_pick": 0,
                "skus_needing_swap": 0,
                "skus_with_packband_issue": 0,
            },
        }

    # lot_key（ロット日付を整数キーに変換）
    inv["lot_key"] = inv["ロット"].map(_parse_lot_date_key)

    # lv / col / dep をロケーション文字列から解析
    loc_str = inv["ロケーション"].astype(str).str.zfill(8)
    inv["lv"]  = pd.to_numeric(loc_str.str[0:3], errors="coerce").fillna(0).astype(int)
    inv["col"] = pd.to_numeric(loc_str.str[3:6], errors="coerce").fillna(0).astype(int)
    inv["dep"] = pd.to_numeric(loc_str.str[6:8],  errors="coerce").fillna(0).astype(int)
    inv["ease_key"] = (
        inv["lv"] * 10000
        + (42 - inv["col"]) * 100
        + inv["dep"]
    )

    # 入数（pack_est）を付与
    pack_map: Optional[pd.Series] = None
    if "入数" in sku_master.columns:
        sku_key = "sku_id" if "sku_id" in sku_master.columns else "商品ID"
        # SKUマスタに重複がある場合は最初の値を使用
        pack_map = sku_master.drop_duplicates(subset=[sku_key]).set_index(sku_key)["入数"].astype(float)
    if pack_map is not None:
        inv["pack_est"] = inv["商品ID"].astype(str).map(pack_map).fillna(0.0)
    elif "pack_qty" in inv.columns:
        inv["pack_est"] = pd.to_numeric(inv["pack_qty"], errors="coerce").fillna(0.0)
    else:
        inv["pack_est"] = 0.0

    # ケース数
    if "cases" in inv.columns:
        inv["qty_cases"] = pd.to_numeric(inv["cases"], errors="coerce").fillna(0.0)
    elif "ケース" in inv.columns:
        inv["qty_cases"] = pd.to_numeric(inv["ケース"], errors="coerce").fillna(0.0)
    else:
        inv["qty_cases"] = 0.0

    # --- FIFO ランキング（SKU全体での古い順: 1=最古）---
    # 同一SKUの全ロケーションを横断して lot_key を昇順ランク付け
    inv["lot_rank"] = (
        inv.groupby("商品ID")["lot_key"]
        .rank(method="dense", ascending=True)
        .astype(int)
    )

    # 取り口フラグ
    inv["is_at_pick"] = inv["lv"].isin(pick_levels)

    # 各SKU で「最古ロット(rank=1)が取り口にあるか」を集計
    oldest_at_pick = (
        inv[inv["lot_rank"] == 1]
        .groupby("商品ID")["is_at_pick"]
        .any()
    )
    inv["oldest_at_pick"] = inv["商品ID"].map(oldest_at_pick).fillna(False)

    # --- 課題フラグ ---
    # 1) FIFO課題: 最古ロット(rank=1)が取り口にない
    inv["fifo_issue"] = (inv["lot_rank"] == 1) & (~inv["is_at_pick"])

    # 2) スワップ課題: 取り口にいるが最古ではない（同SKUの最古が保管にある）
    inv["swap_needed"] = (
        inv["is_at_pick"]
        & (inv["lot_rank"] > 1)
        & (~inv["oldest_at_pick"])
    )

    # 3) 入数帯課題: 正しい列ゾーンにいない
    allowed_cols_series = inv.apply(lambda row: _allowed_cols_for_row(row, cfg), axis=1)
    inv["packband_issue"] = inv.apply(
        lambda row: int(row["col"]) not in allowed_cols_series[row.name], axis=1
    )

    # --- スコア計算 ---
    # fifo_score: (現在の段-1) × 10000（段が高いほどペナルティ大）
    inv["fifo_score"] = (
        inv["fifo_issue"].astype(int) * (inv["lv"] - 1) * 10000
        + inv["swap_needed"].astype(int) * 5000
    )
    inv["packband_score"] = inv["packband_issue"].astype(int) * pack_mismatch_penalty
    inv["total_issue_score"] = inv["fifo_score"] + inv["packband_score"]

    # --- サマリー ---
    summary = {
        "total_items": len(inv),
        "fifo_issues": int(inv["fifo_issue"].sum()),
        "swap_needed": int(inv["swap_needed"].sum()),
        "packband_issues": int(inv["packband_issue"].sum()),
        "items_with_issues": int((inv["total_issue_score"] > 0).sum()),
        "skus_with_oldest_not_at_pick": int(
            inv[(inv["lot_rank"] == 1) & (~inv["is_at_pick"])]["商品ID"].nunique()
        ),
        "skus_needing_swap": int(inv[inv["swap_needed"]]["商品ID"].nunique()),
        "skus_with_packband_issue": int(inv[inv["packband_issue"]]["商品ID"].nunique()),
    }

    print(
        f"[score_all_inventory] 合計={summary['total_items']}件 "
        f"FIFO課題={summary['fifo_issues']} "
        f"スワップ必要={summary['swap_needed']} "
        f"入数帯課題={summary['packband_issues']}"
    )

    # --- 出力 ---
    output_cols = [
        "商品ID", "ロット", "ロケーション", "lv", "col", "dep",
        "ease_key", "pack_est", "qty_cases", "lot_key", "lot_rank",
        "is_at_pick", "oldest_at_pick",
        "fifo_issue", "swap_needed", "packband_issue",
        "fifo_score", "packband_score", "total_issue_score",
    ]
    available_cols = [c for c in output_cols if c in inv.columns]
    records = (
        inv[available_cols]
        .sort_values("total_issue_score", ascending=False)
        .to_dict(orient="records")
    )

    return {
        "inventory_scores": records,
        "summary": summary,
    }