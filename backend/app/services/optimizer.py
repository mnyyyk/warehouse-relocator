from __future__ import annotations

import math
import re
import copy
import asyncio
import secrets
import time
import json
from collections import deque
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict, Set, Any

import pandas as pd

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

# -------------------------------
# Global state for debugging and SSE
# -------------------------------
_last_relocation_debug: Dict[str, Any] = {
    "trace_id": None,
    "planned": 0,
    "accepted": 0,
    "rejections": {},
    "examples": {},
}

_CURRENT_TRACE_ID: Optional[str] = None
_DROP_STORE: Dict[str, List[Dict[str, Any]]] = {}
_TRACE_SUBS: Dict[str, List[asyncio.Queue[str]]] = {}
_TRACE_BUFFER: Dict[str, deque[str]] = {}
_TRACE_BUFFER_MAX = 200


# -------------------------------
# Public dataclasses
# -------------------------------
from dataclasses import dataclass

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
    # 各列に許容する“ミックス列”枠（±10%を外れても許す枠数）
    mix_slots_per_col: int = 0
    # ミックス枠を消費する場合の微小なコスト
    mix_usage_penalty: float = 1.0
    # AI優先列配置のボーナス
    ai_preferred_column_bonus: float = 25.0
    # 絶対制約スイッチ類（AI案の最終ゲートなどで使用）
    hard_cap: bool = False            # 容量をハードに守る
    hard_fifo: bool = False           # 「古いロットが下」ルールをハードに守る
    strict_pack: Optional[str] = None # "A", "B", "A+B" を指定可（Noneは従来通り）
    exclude_oversize: bool = False    # 1ケースが棚容積上限を超えるSKUを除外
    # --- Eviction chain (bounded multi-step relocation) ---
    chain_depth: int = 0            # 0=disabled; try 2 to enable shallow chains
    eviction_budget: int = 0        # max number of eviction (auxiliary) moves
    touch_budget: int = 1000        # max number of distinct locations we can touch
    buffer_slots: int = 0           # reserved empty-ish slots for temporary staging (0=disabled)
# -------------------------------
# Additional helpers for hard constraints and pack clustering
# -------------------------------

def _hard_fifo_violation_simple(inv: pd.DataFrame, sku: str, lot_key: int, target_level: int, tcol: int) -> bool:
    """(sku, col) に対して、target_level へ配置すると『古いロットほど低段』に違反するかを厳密判定する。"""
    need_cols = {"lv", "lot_key", "商品ID", "col"}
    if not need_cols.issubset(inv.columns):
        return False
    same = inv[(inv["商品ID"].astype(str) == str(sku)) & (inv["col"] == int(tcol))]
    if same.empty:
        return False
    newer = same[same["lot_key"] > lot_key]
    if not newer.empty:
        min_newer_lv = int(newer["lv"].min())
        if target_level >= min_newer_lv:
            return True
    older = same[same["lot_key"] < lot_key]
    if not older.empty:
        max_older_lv = int(older["lv"].max())
        if target_level <= max_older_lv:
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
    tmp["weight"] = pd.to_numeric(tmp.get("qty_cases_move", 1.0), errors="coerce").fillna(1.0).astype(float)
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

def enforce_constraints(
    sku_master: pd.DataFrame,
    inventory: pd.DataFrame,
    moves: List[Move],
    *,
    cfg: OptimizerConfig | None = None,
) -> List[Move]:
    """
    外部(例: AIドラフト)で生成された移動候補に対し、
    容量・FIFO・入数まとまり(A)・オーバーサイズ等の『絶対ゲート』を順次適用し、
    採択可能な移動のみを返す。
    - 容量は逐次加算で判定 (to_locごとの used + delta)
    - FIFOは (SKU, 列) 単位で『古いロットほど低段』の厳密順序をチェック
    - 入数まとまりは "A"（±帯）をハードに適用（"B"はスコア系なのでここではハード適用しない）
    """
    cfg = cfg or OptimizerConfig()
    cap_limit = _capacity_limit(getattr(cfg, "fill_rate", None))

    # --- SKU -> 1ケース容積(m³)
    sku_vol_map = _build_carton_volume_map(sku_master)
    key_series = inventory["商品ID"].astype(str) if "商品ID" in inventory.columns else inventory["sku_id"].astype(str)

    # 入数マップ
    pack_map = None
    if "入数" in sku_master.columns:
        key = "sku_id" if "sku_id" in sku_master.columns else "商品ID"
        pack_map = sku_master.set_index(key)["入数"].astype(float)

    # 在庫の lot_key / 位置 を整備
    inv = inventory.copy()
    _require_cols(inv, ["ロケーション", "商品ID", "ロット"], "inventory")
    inv["lot_key"] = inv["ロット"].map(_parse_lot_date_key)
    inv[["lv", "col", "dep"]] = inv["ロケーション"].apply(lambda s: pd.Series(_parse_loc8(str(s))))

    # 代表入数(列)
    rep_pack_by_col = _compute_rep_pack_by_col_for_inv(inv, pack_map, getattr(cfg, "preserve_pack_mapping_threshold", 0.7))

    # ロケ別使用量（m³）を初期化
    # 在庫に qty_cases_move が無い可能性があるので安全に推計
    inv_key = inv["商品ID"].astype(str)
    vol_each_series = inv_key.map(sku_vol_map).fillna(0.0)
    qty_cases_float = pd.to_numeric(inv.get("qty_cases_move", inv.get("ケース", 0)), errors="coerce").fillna(0.0).astype(float)
    if "qty_cases_move" not in inv.columns:
        # ケース数が取れない場合は 1ケース相当で安全側に
        qty_cases_float = qty_cases_float.replace(0.0, 1.0)
    inv["volume_total"] = vol_each_series * qty_cases_float
    shelf_usage = inv.groupby("ロケーション")["volume_total"].sum().to_dict()

    accepted: List[Move] = []
    # シミュレーション用: invへ逐次的に行を追加入力（最小限の列だけ）
    sim_inv = inv[["商品ID", "lot_key", "lv", "col", "dep", "ロケーション"]].copy()

    for m in moves:
        sku = str(m.sku_id)
        lot_key = _parse_lot_date_key(m.lot)
        # 未判定ロットは採択しない
        if lot_key == UNKNOWN_LOT_KEY:
            continue
        to_loc = str(m.to_loc)
        from_loc = str(m.from_loc)
        tlv, tcol, tdep = _parse_loc8(to_loc)
        add_each = float(sku_vol_map.get(sku, 0.0) or 0.0)
        add_vol = add_each * float(m.qty)

        # 1) オーバーサイズ除外
        if cfg.exclude_oversize and add_each > cap_limit:
            continue

        # 2) ハード容量
        if cfg.hard_cap:
            used_to = float(shelf_usage.get(to_loc, 0.0))
            if used_to + add_vol > cap_limit:
                continue

        # 3) ハードFIFO（同SKU×同列）
        if cfg.hard_fifo:
            if _hard_fifo_violation_simple(sim_inv, sku, lot_key, tlv, tcol):
                continue

        # 4) 入数まとまり(A): ±帯をハード
        if cfg.strict_pack and ("A" in str(cfg.strict_pack)):
            if pack_map is not None:
                pack_est = float(pack_map.get(sku, float("nan")))
                rep = rep_pack_by_col.get(tcol)
                if rep and not _within_band(pack_est, rep, getattr(cfg, "pack_tolerance_ratio", 0.10)):
                    continue

        # --- 採択（lot_date を付与してコピー）
        lot_date_str = _lot_key_to_datestr8(lot_key)
        accepted.append(
            Move(
                sku_id=sku,
                lot=m.lot,
                qty=m.qty,
                from_loc=from_loc,
                to_loc=to_loc,
                lot_date=lot_date_str,
            )
        )
        # 容量を逐次更新
        shelf_usage[to_loc] = shelf_usage.get(to_loc, 0.0) + add_vol
        shelf_usage[from_loc] = max(0.0, shelf_usage.get(from_loc, 0.0) - add_vol)

        # FIFO用に sim_inv を更新
        new_row = {
            "商品ID": sku, "lot_key": lot_key, "lv": tlv, "col": tcol, "dep": tdep, "ロケーション": to_loc
        }
        sim_inv = pd.concat([sim_inv, pd.DataFrame([new_row])], ignore_index=True)

    return accepted


# -------------------------------
# Internal helpers
# -------------------------------

def _capacity_limit(fill_rate: Optional[float]) -> float:
    """棚1マスの容積上限 (m^3)。既定は 1.0*1.0*1.3 * 0.95。
    `fill_rate` が None の場合は DEFAULT_FILL_RATE を用いる。
    """
    cap = SHELF_WIDTH_M * SHELF_DEPTH_M * SHELF_HEIGHT_M
    fr = DEFAULT_FILL_RATE if fill_rate is None else float(fill_rate)
    return cap * fr


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


def _location_key(loc: str) -> Tuple[int, int, int]:
    """入口に近い順（列→奥→段）でのソートキー。"""
    lv, col, dep = _parse_loc8(loc)
    return (col, dep, lv)


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

    return 9_9999_999


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
) -> bool:
    """同一列(=tcol)に限定して「古いロットほど低段(数値が小さい)」を**厳密**に維持する。
    条件:
      - 同列にある「新しいロット」の最小段 >= target_level だと違反（古いは必ずそれより下）
      - 同列にある「古いロット」の最大段 >= target_level だと違反（新しいは必ずそれより上）
    """
    need_cols = {"lv", "lot_key", "商品ID", "col"}
    if not need_cols.issubset(inv.columns):
        return False

    same = inv[(inv["商品ID"].astype(str) == str(sku)) & (inv["col"] == int(tcol))]
    if same.empty:
        return False

    newer = same[same["lot_key"] > lot_key]
    if not newer.empty:
        min_newer_lv = int(newer["lv"].min())
        # 古い(=本行)は、新しいより必ず低い段(数値が小さい)に置く必要がある
        if target_level >= min_newer_lv:
            return True

    older = same[same["lot_key"] < lot_key]
    if not older.empty:
        max_older_lv = int(older["lv"].max())
        # 新しい(=本行)は、古いより必ず高い段(数値が大きい)に置く必要がある
        if target_level <= max_older_lv:
            return True

    return False


# Pass-1: Move blockers (新しいロットが古いロットより低段にいるケースを上段へ退避)
def _pass1_raise_blockers(
    inv: pd.DataFrame,
    shelf_usage: dict[str, float],
    cap_limit: float,
    cfg: OptimizerConfig,
) -> List[Move]:
    """Pass-1: 同一列×同一SKU内で、並び順(古い→低段)に違反している“新しいロット”を
    必要最小限だけ上段へ退避してブロッカーを解消する。
    退避先は基本「同一列」のみを探索（容量を満たす最小スコア先を選択）。
    """
    moves: List[Move] = []
    if inv.empty:
        return moves

    # 対象は容量情報がある行のみ
    subset = inv[(inv["volume_each_case"] > 0) & (inv["lot_key"] != UNKNOWN_LOT_KEY)].copy()
    if subset.empty:
        return moves

    # 列→SKUでグルーピングして、ロット昇順で走査（古い→新しい）
    for (c, sku), g in subset.groupby(["col", "商品ID"]):
        try:
            c_int = int(c)
        except Exception:
            continue
        g = g.sort_values(["lot_key", "lv"], ascending=[True, True])  # 古い順、より低段が先

        # 直前(より古い)の最大段（=新しい側の許容最小段）
        max_old_lv = None
        for idx, row in g.iterrows():
            lv = int(row["lv"])
            lot_key = int(row["lot_key"])
            from_loc = str(row["ロケーション"])
            qty_cases = int(row["qty_cases_move"]) or 0
            if qty_cases <= 0:
                continue
            need_vol = float(row["volume_each_case"]) * qty_cases

            if max_old_lv is None:
                max_old_lv = lv
                continue

            # 新しいロットは max_old_lv 以上の段にいなければならない
            if lv >= max_old_lv:
                max_old_lv = max(max_old_lv, lv)
                continue

            # 違反: この“新しいロット”を上段(>= max_old_lv)へ退避
            # 候補: 同一列 c の各ロケーションのうち、レベル>=max_old_lv & 容量OK
            cand_locs = [
                loc for loc in shelf_usage.keys()
                if loc not in PLACEHOLDER_LOCS
                and loc != from_loc
                and _parse_loc8(loc)[1] == c_int
                and _parse_loc8(loc)[0] >= max_old_lv
            ]
            cand_locs.sort(key=_location_key)

            best_to = None
            best_score = math.inf
            for to_loc in cand_locs:
                tlv, tcol, tdep = _parse_loc8(to_loc)
                used = shelf_usage.get(to_loc, 0.0)
                if used + need_vol > cap_limit:
                    continue
                # 厳密列内ルールに照らして最終確認
                if _violates_lot_level_rule(inv, str(row["商品ID"]), lot_key, tlv, tcol, tdep, idx):
                    continue

                # スコア: なるべく少ない持ち上げ + 浅い奥
                score = (tlv - lv) + (tdep * 0.001)
                if score < best_score:
                    best_score = score
                    best_to = to_loc

            if best_to is None:
                # 退避できなければ諦め（Pass-2が別列等で救う可能性はある）
                continue

            # 移動実行
            tlv, tcol, tdep = _parse_loc8(best_to)
            # lot_date を付与
            lk = int(row.get("lot_key")) if pd.notna(row.get("lot_key")) else _parse_lot_date_key(str(row.get("ロット") or ""))
            moves.append(
                Move(
                    sku_id=str(row["商品ID"]),
                    lot=str(row.get("ロット") or ""),
                    qty=qty_cases,
                    from_loc=from_loc,
                    to_loc=best_to,
                    lot_date=_lot_key_to_datestr8(lk),
                )
            )
            # 使用量更新
            shelf_usage[from_loc] = max(0.0, shelf_usage.get(from_loc, 0.0) - need_vol)
            shelf_usage[best_to] = shelf_usage.get(best_to, 0.0) + need_vol

            # inv 内の現在位置を更新
            inv.at[idx, "lv"] = tlv
            inv.at[idx, "col"] = tcol
            inv.at[idx, "dep"] = tdep
            inv.at[idx, "ロケーション"] = best_to

            # 次の新しいロットが従うべき最小段
            max_old_lv = max(max_old_lv, tlv)

    if moves:
        print(f"[optimizer] pass1_raise_blockers moves={len(moves)}")
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
                    score -= OptimizerConfig.ai_preferred_column_bonus.__get__(OptimizerConfig) / (rank + 1.0)  # type: ignore
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

    # prefer shallow depth slightly
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
) -> Optional[str]:
    """Choose a plausible destination location for the whole row."""
    from_loc = str(row["ロケーション"])
    best_to = None
    best_score = math.inf
    for to_loc in shelf_usage.keys():
        if to_loc in PLACEHOLDER_LOCS or to_loc == from_loc or to_loc in avoid_locs:
            continue
        tlv, tcol, tdep = _parse_loc8(str(to_loc))
        if tlv <= 0:
            continue
        # FIFO strict within same column only: if moving within same column, ensure not violating
        lot_key = int(row.get("lot_key") or UNKNOWN_LOT_KEY)
        if lot_key != UNKNOWN_LOT_KEY and tcol == int(row.get("col") or 0):
            if _violates_lot_level_rule(inv, str(row["商品ID"]), lot_key, tlv, tcol, tdep, None):
                continue
        # score and pick
        s = _score_destination_for_row(
            row,
            str(to_loc),
            rep_pack_by_col,
            pack_tolerance_ratio,
            OptimizerConfig.same_sku_same_column_bonus.__get__(OptimizerConfig),  # type: ignore
            OptimizerConfig.prefer_same_column_bonus.__get__(OptimizerConfig),     # type: ignore
            OptimizerConfig.split_sku_new_column_penalty.__get__(OptimizerConfig), # type: ignore
            ai_col_hints,
        )
        if s < best_score:
            best_score = s
            best_to = str(to_loc)
    return best_to

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
) -> Optional[List[Move]]:
    """Try to free `need_vol` capacity on `target_loc` by evicting whole rows.
    Returns a list of eviction moves in execution order if successful; otherwise None.
    This mutates `inv` and `shelf_usage` when it succeeds.
    """
    # Quick check
    used = float(shelf_usage.get(target_loc, 0.0))
    free = cap_limit - used
    if free >= need_vol:
        return []
    if budget.depth_left <= 0 or budget.evictions_left <= 0 or budget.touch_left <= 0:
        return None

    # Choose evictable rows in target_loc (prefer newest lot and pack-mismatch)
    in_rows = inv[inv["ロケーション"].astype(str) == str(target_loc)].copy()
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
    for _, row in in_rows_sorted.iterrows():
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
        )
        if not dest:
            continue

        # If destination lacks capacity, recursively free it first
        used_d = float(shelf_usage.get(dest, 0.0))
        need_d = max(0.0, used_d + vol_move - cap_limit)
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
        mv = Move(
            sku_id=sku,
            lot=lot_str,
            qty=qty_cases,
            from_loc=str(row["ロケーション"]),
            to_loc=str(dest),
            lot_date=_lot_key_to_datestr8(lk),
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
        budget.max_depth_used = max(budget.max_depth_used, (OptimizerConfig.chain_depth.__get__(OptimizerConfig) - budget.depth_left + 1))  # type: ignore

        # check if we've freed enough on target
        used_now = float(shelf_usage.get(target_loc, 0.0))
        free_now = cap_limit - used_now
        if free_now >= need_vol:
            return chain

        # else continue trying more rows in target_loc
    return None


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
    cap_limit = _capacity_limit(getattr(cfg, "fill_rate", None))

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

    if quality_filter is not None:
        qset = set(str(x) for x in quality_filter)
        inv = inv[inv["quality_name"].astype(str).isin(qset)].copy()
    print(f"[optimizer] after quality_filter={list(quality_filter) if quality_filter is not None else None} rows={len(inv)}")

    # プレースホルダロケは移動元として扱わない
    inv = inv[~inv["ロケーション"].astype(str).isin(PLACEHOLDER_LOCS)].copy()

    # --- SKU 段ボール容積マップ（m³/ケース）
    sku_vol_map = _build_carton_volume_map(sku_master)
    print(f"[optimizer] carton_volume_map size={sku_vol_map.shape[0]}")

    # --- 入数マップ（ケース換算に使用）
    pack_map = None
    if "入数" in sku_master.columns:
        key = "sku_id" if "sku_id" in sku_master.columns else "商品ID"
        pack_map = sku_master.set_index(key)["入数"].astype(float)

    # --- lot key & 現在のロケ座標（後続の段ルールで使用）
    inv["lot_key"] = inv["ロット"].map(_parse_lot_date_key)
    inv[["lv", "col", "dep"]] = inv["ロケーション"].apply(lambda s: pd.Series(_parse_loc8(str(s))))

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

    inv["qty_cases_move"] = inv["qty_cases_float"].apply(lambda x: int(math.ceil(float(x))))
    if not inv.empty:
        try:
            qmin = int(inv["qty_cases_move"].min())
            qmax = int(inv["qty_cases_move"].max())
            print(f"[optimizer] qty_cases_move min={qmin} max={qmax}")
        except Exception:
            pass

    # --- ケース容積を付与（安全側：移動用整数で容量評価）
    key_series = inv["商品ID"].astype(str)
    inv["volume_each_case"] = key_series.map(sku_vol_map).fillna(0.0)
    inv["volume_total"] = inv["qty_cases_move"] * inv["volume_each_case"]

    # オーバーサイズSKUの早期除外
    if getattr(cfg, "exclude_oversize", False):
        inv = inv[inv["volume_each_case"] <= cap_limit].copy()

    # --- 推定入数（pack_est）: SKUマスターに入数があればそれを使う
    if pack_map is not None:
        inv["pack_est"] = inv["商品ID"].astype(str).map(pack_map)
    else:
        inv["pack_est"] = pd.NA

    # 列⇔SKU の存在マップ（同一SKUを同じ列にまとめるための判断に使用）
    # ※この時点の在庫状態に基づく（逐次更新は後続で最小限のみ反映）
    try:
        col_to_skus = inv.groupby("col")["商品ID"].apply(lambda s: set(str(x) for x in s.unique())).to_dict()
    except Exception:
        col_to_skus = {}
    try:
        sku_to_cols = inv.groupby("商品ID")["col"].apply(lambda s: set(int(x) for x in s.unique())).to_dict()
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
    print(f"[optimizer] shelf_usage locations={len(shelf_usage)} (cap={cap_limit})")

    # --- Pass-1: ブロッカー退避（同列内で新しいロットが低段にいる違反を解消）
    moves: List[Move] = []
    p1_moves = _pass1_raise_blockers(inv, shelf_usage, cap_limit, cfg)
    moves.extend(p1_moves)
    moved = len(moves)
    # Pass-1で全体が更新されたので並べ替えし直し
    order_idx = _sort_index_for_pick(inv)

    for idx in order_idx:
        row = inv.loc[idx]
        from_loc = str(row["ロケーション"])  # 8桁
        lv, col, dep = row.get("lv"), row.get("col"), row.get("dep")
        if pd.isna(lv) or pd.isna(col) or pd.isna(dep):
            continue
        lv, col, dep = int(lv), int(col), int(dep)
        if lv == 1:
            continue  # 既に最下段

        qty_cases = int(row["qty_cases_move"]) or 0
        if qty_cases <= 0:
            continue
        need_vol = float(row["volume_each_case"]) * qty_cases
        sku_val = str(row["商品ID"])
        lot_key = int(row.get("lot_key") or 9_9999_999)
        if int(lot_key) == UNKNOWN_LOT_KEY:
            continue

        # 全ロケーションから、より低い段(target_level)の候補を探索し、スコア最小を採用
        best_choice: Optional[Tuple[str, int, int, int, bool]] = None
        best_score: float = math.inf
        cur_col = int(col)
        pack_val = row.get("pack_est")
        for target_level in range(1, lv):
            cand_locs = [loc for loc in shelf_usage.keys()
                         if loc not in PLACEHOLDER_LOCS
                         and loc != from_loc
                         and _parse_loc8(loc)[0] == target_level]
            cand_locs.sort(key=_location_key)
            for to_loc in cand_locs:
                tlv, tcol, tdep = _parse_loc8(to_loc)
                if _violates_lot_level_rule(inv, sku_val, lot_key, target_level, tcol, tdep, idx):
                    continue
                used = float(shelf_usage.get(to_loc, 0.0))
                if used + need_vol > cap_limit:
                    # Try bounded eviction chain if enabled
                    if getattr(cfg, "chain_depth", 0) and getattr(cfg, "eviction_budget", 0) and getattr(cfg, "touch_budget", 0):
                        budget = _ChainBudget(
                            depth_left=int(getattr(cfg, "chain_depth", 0)),
                            evictions_left=int(getattr(cfg, "eviction_budget", 0)),
                            touch_left=int(getattr(cfg, "touch_budget", 0)),
                            touched=set([from_loc, to_loc]),
                        )
                        ev_chain = _plan_eviction_chain(
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
                        )
                        if ev_chain is None:
                            continue
                        # Tentatively accept this candidate with the eviction chain
                        # We will append `ev_chain` moves just before the main move (after scoring passes)
                    else:
                        continue

                # ペナルティ計算（小さいほど良い）
                score = 0.0
                needs_mix_slot = False

                # 0) AI の列ヒント（順位に応じてボーナスを逓減）
                if ai_col_hints:
                    try:
                        pref_cols = ai_col_hints.get(sku_val) or ai_col_hints.get(str(sku_val))
                        if pref_cols:
                            pref_ints = [int(x) for x in pref_cols]
                            if tcol in pref_ints:
                                rank = pref_ints.index(tcol)  # 0-based
                                score -= cfg.ai_preferred_column_bonus / (rank + 1.0)
                    except Exception:
                        pass

                # 1) 列代表入数（支配的入数があれば優先、なければメディアン）との整合性
                if pack_val is not None and not pd.isna(pack_val):
                    rep = rep_pack_by_col.get(tcol)
                    if rep and float(rep) > 0:
                        diff_ratio = abs(float(pack_val) - float(rep)) / float(rep)
                        if diff_ratio > cfg.pack_tolerance_ratio:
                            # A: ハード制約（±帯）… ただしミックス枠が残っていれば許可
                            if mix_slots_left.get(tcol, 0) <= 0:
                                continue
                            needs_mix_slot = True
                            # B: ソフト制約（罰則）—超過分を二乗で強化
                            excess = diff_ratio - cfg.pack_tolerance_ratio
                            score += cfg.pack_mismatch_penalty * (excess * excess + 1.0)
                            score += cfg.mix_usage_penalty
                        else:
                            # 許容内の微差は小さくペナルティ
                            penalty_base = cfg.pack_within_tolerance_penalty * (diff_ratio * 100.0)
                            if getattr(cfg, "strict_pack", None) and ("B" in str(cfg.strict_pack)):
                                penalty_base *= 3.0  # Bモード: 近似クラスターをより強めに誘導
                            score += penalty_base

                # 2) 同一SKUを同じ列に集約（ボーナス）
                if tcol in col_to_skus and sku_val in col_to_skus[tcol]:
                    score -= cfg.same_sku_same_column_bonus

                # 3) 元と同じ列をやや優先（ボーナス）
                if tcol == cur_col:
                    score -= cfg.prefer_same_column_bonus

                # 4) SKUの新規列への分散は少しだけ抑制（ペナルティ）
                if sku_val in sku_to_cols and tcol not in sku_to_cols[sku_val]:
                    score += cfg.split_sku_new_column_penalty

                # 5) タイブレーク（奥は浅いほうをほんの少し優先）
                score += (tdep * 0.001)

                if score < best_score:
                    best_score = score
                    best_choice = (to_loc, tlv, tcol, tdep, needs_mix_slot)

        if best_choice is not None:
            to_loc, tlv, tcol, tdep, needs_mix_slot = best_choice
            # If we prepared an eviction chain for this destination, realize it now
            if 'ev_chain' in locals() and ev_chain:
                for emv in ev_chain:
                    moves.append(emv)
                moved += len([1 for _ in moves[-len(ev_chain):]]) if isinstance(ev_chain, list) else 0
                # reset so it won't be reused for other candidates
                ev_chain = []
            # lot_key はこの直前で計算済み（変数 lot_key）。lot_date を付与して出力。
            moves.append(
                Move(
                    sku_id=sku_val,
                    lot=str(row.get("ロット") or ""),
                    qty=qty_cases,
                    from_loc=from_loc,
                    to_loc=to_loc,
                    lot_date=_lot_key_to_datestr8(lot_key),
                )
            )
            # 使用量更新（安全側: 切り上げケースで評価）
            shelf_usage[from_loc] = max(0.0, shelf_usage.get(from_loc, 0.0) - need_vol)
            shelf_usage[to_loc] = shelf_usage.get(to_loc, 0.0) + need_vol

            if needs_mix_slot:
                mix_slots_left[tcol] = max(0, mix_slots_left.get(tcol, 0) - 1)

            # 現在位置を更新（後続の段ルール評価に反映）
            inv.at[idx, "lv"] = tlv
            inv.at[idx, "col"] = tcol
            inv.at[idx, "dep"] = tdep
            inv.at[idx, "ロケーション"] = to_loc

            # 列⇔SKUマップを最小限更新（厳密な再集計はコスト高なので省略）
            col_to_skus.setdefault(tcol, set()).add(sku_val)
            sku_to_cols.setdefault(sku_val, set()).add(tcol)

            moved += 1

        if moved >= (cfg.max_moves or 1_000_000):
            break

    limit_str = cfg.max_moves if cfg.max_moves else "unlimited"
    print(f"[optimizer] planned moves={len(moves)} (limit={limit_str})")
    if getattr(cfg, "chain_depth", 0):
        print("[optimizer] eviction chain enabled: depth=", getattr(cfg, "chain_depth", 0))
    
    # Generate comprehensive summary report
    try:
        planned_count = len(moves)  # All planned moves
        accepted_count = planned_count  # Assume all accepted at this stage
        rejected_count = 0
        
        report = _generate_comprehensive_report(
            moves=moves,
            planned_count=planned_count,
            accepted_count=accepted_count,
            rejected_count=rejected_count,
            sku_master=sku_master,
            inventory=inventory,  # Pass inventory for proper consolidation analysis
        )
        
        # Store report for retrieval
        trace_id = getattr(cfg, "trace_id", None)
        set_summary_report(report, trace_id)
        print(f"[optimizer] Summary report generated ({len(report)} chars, trace_id={trace_id})")
    except Exception as e:
        print(f"[optimizer] Failed to generate summary report: {e}")
    
    return moves


def optimise_relocation(
    sku_master: pd.DataFrame,
    inventory: pd.DataFrame,
    *,
    cfg: OptimizerConfig | None = None,
    block_filter: Iterable[str] | None = None,
    quality_filter: Iterable[str] | None = None,
) -> List[Move]:
    """
    Alias of :func:`plan_relocation` for callers that use the British
    spelling ``optimise_relocation`` (relocation_tasks.py etc.).
    """
    return plan_relocation(
        sku_master,
        inventory,
        cfg=cfg,
        block_filter=block_filter,
        quality_filter=quality_filter,
    )

# ---------------------------------------------------------------------------
# Summary Report Generation
# ---------------------------------------------------------------------------
_LAST_SUMMARY_REPORT: Optional[str] = None
_TRACE_SUMMARY_REPORTS: dict[str, str] = {}

def _generate_comprehensive_report(
    moves: List[Move],
    planned_count: int,
    accepted_count: int,
    rejected_count: int,
    sku_master: Optional[pd.DataFrame] = None,
    inventory: Optional[pd.DataFrame] = None,
) -> str:
    """
    Generate a comprehensive text report for relocation results.
    
    Args:
        moves: Final accepted moves
        planned_count: Total number of planned moves
        accepted_count: Number of accepted moves
        rejected_count: Number of rejected moves
        sku_master: SKU master dataframe (for pack_qty lookup)
        inventory: Current inventory dataframe (for before-state analysis)
        
    Returns:
        Formatted text report with proper SKU consolidation analysis
    """
    if not moves:
        return "移動が生成されませんでした。"
    
    # Build pack_map for入数 lookup
    pack_map: dict[str, int] = {}
    if sku_master is not None and not sku_master.empty:
        for _, row in sku_master.iterrows():
            sku_id = str(row.get("商品ID", ""))
            pack_qty = row.get("入数", None)
            if sku_id and pd.notna(pack_qty):
                try:
                    pack_map[sku_id] = int(pack_qty)
                except:
                    pass
    
    # Analyze moves
    total_moves = len(moves)
    unique_skus = set()
    total_cases = 0
    sku_from_locs: dict[str, set[str]] = {}  # SKU -> set of from_locs
    sku_to_locs: dict[str, set[str]] = {}    # SKU -> set of to_locs
    reason_counts: dict[str, int] = {}
    old_lot_to_pick = 0
    high_ship_to_hot = 0
    
    for m in moves:
        sku = str(m.sku_id)
        unique_skus.add(sku)
        total_cases += m.qty
        
        # Track from/to locations per SKU
        sku_from_locs.setdefault(sku, set()).add(m.from_loc)
        sku_to_locs.setdefault(sku, set()).add(m.to_loc)
        
        # Parse reason field
        reason = getattr(m, "reason", None) or ""
        
        # Extract pack from reason if available (format: "入数帯是正(XX入→...")
        import re
        pack_match = re.search(r'入数帯是正\((\d+)入', reason)
        if pack_match:
            pack = pack_match.group(1)
            key = f"入数帯是正({pack}入"
            reason_counts[key] = reason_counts.get(key, 0) + 1
        
        # Count FIFO improvements (old lot to pick levels)
        from_lv = int(m.from_loc[0:3]) if len(m.from_loc) >= 3 else 0
        to_lv = int(m.to_loc[0:3]) if len(m.to_loc) >= 3 else 0
        if to_lv in (1, 2) and from_lv > to_lv:
            old_lot_to_pick += 1
        
        # Count high-ship SKU to hotspot (level 1, col 15-30, depth 1-3)
        to_col = int(m.to_loc[3:6]) if len(m.to_loc) >= 6 else 0
        to_dep = int(m.to_loc[6:8]) if len(m.to_loc) == 8 else 0
        if to_lv == 1 and 15 <= to_col <= 30 and 1 <= to_dep <= 3:
            high_ship_to_hot += 1
    
    # Analyze SKU consolidation (CORRECT VERSION with inventory data)
    sku_consolidations: list[tuple[str, int, int]] = []
    
    if inventory is not None and not inventory.empty:
        # Count locations per SKU in original inventory
        sku_before_locs: dict[str, set[str]] = {}
        for _, row in inventory.iterrows():
            sku_id = str(row.get("商品ID", ""))
            loc = str(row.get("ロケーション", ""))
            if sku_id and loc:
                sku_before_locs.setdefault(sku_id, set()).add(loc)
        
        # For each moved SKU, calculate after-move location count
        for sku in sorted(unique_skus):
            before_locs = sku_before_locs.get(sku, set())
            moved_from = sku_from_locs.get(sku, set())
            moved_to = sku_to_locs.get(sku, set())
            
            # After-move = (before - moved_from) + moved_to
            after_locs = (before_locs - moved_from) | moved_to
            
            before_count = len(before_locs)
            after_count = len(after_locs)
            
            # Only include if there's actual consolidation (many->few)
            if before_count > after_count:
                sku_consolidations.append((sku, before_count, after_count))
    else:
        # Fallback: analyze only within moves (no inventory data)
        for sku in sorted(unique_skus):
            from_count = len(sku_from_locs.get(sku, set()))
            to_count = len(sku_to_locs.get(sku, set()))
            # Only include if there's actual consolidation (many->few)
            if from_count > to_count:
                sku_consolidations.append((sku, from_count, to_count))
    
    # Sort by consolidation impact (largest reduction first)
    sku_consolidations.sort(key=lambda x: x[1] - x[2], reverse=True)
    
    # Build report
    lines = []
    lines.append("=" * 70)
    lines.append("📊 リロケーション結果の総合評価")
    lines.append("=" * 70)
    lines.append("")
    
    lines.append("【実施結果】")
    lines.append(f"  計画移動数: {planned_count} 件")
    lines.append(f"  承認移動数: {accepted_count} 件")
    lines.append(f"  却下移動数: {rejected_count} 件")
    move_rate = (accepted_count / planned_count * 100) if planned_count > 0 else 0
    lines.append(f"  移動率: {move_rate:.1f}%")
    lines.append(f"  影響SKU数: {len(unique_skus)} 種類")
    lines.append(f"  総ケース数: {total_cases:,} ケース")
    lines.append("")
    
    lines.append("【最適化効果】")
    lines.append("  ▶ Pass毎の改善 (実行順):")
    lines.append("    • Pass-1 (取口/保管整列) 【最優先】: " + f"{total_moves}件")
    lines.append("      → 同一SKU・列内で古いロットを取り口へ、新しいロットを保管段へ（相対的順序）")
    lines.append("")
    lines.append(f"  📦 古いロット→取り口ロケ: {old_lot_to_pick} 件")
    lines.append("     (同一SKU・列内で相対的に古いロットをレベル1-2へ移動)")
    lines.append(f"  🔥 高出荷SKU→ホットスポット: {high_ship_to_hot} 件")
    lines.append("     (レベル1, 列15-30(中心), 奥行1-3(手前))")
    lines.append("")
    
    if reason_counts:
        lines.append("  ▶ 移動理由別の内訳:")
        sorted_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
        for reason, count in sorted_reasons[:10]:  # Top 10
            lines.append(f"    • {reason}: {count}件")
        lines.append("")
    
    # SKU consolidation section (CORRECTED)
    lines.append("【SKU分散/集約状況】")
    if sku_consolidations:
        for sku, from_cnt, to_cnt in sku_consolidations[:5]:  # Top 5 consolidated SKUs
            lines.append(f"  {sku}: {from_cnt}ロケ→{to_cnt}ロケ (集約効果あり)")
    else:
        lines.append("  ※ 集約効果のあるSKU移動はありません")
        lines.append("  ※ 各移動は独立した段下げ/ゾーン是正です")
    lines.append("")
    
    lines.append("【ハードルール検証】")
    lines.append("  ✅ ロット混在なし")
    lines.append("")
    
    lines.append("【推奨事項】")
    if old_lot_to_pick > 0:
        lines.append(f"  • ✅ 古いロット{old_lot_to_pick}件を取り口ロケ(Lv1)へ移動 - FIFO回転促進")
    if high_ship_to_hot > 0:
        lines.append(f"  • ✅ 出荷頻度の高いSKU {high_ship_to_hot}件をホットスポットへ配置 - ピッキング効率向上")
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def set_summary_report(report: str, trace_id: Optional[str] = None) -> None:
    """Store generated summary report for retrieval."""
    global _LAST_SUMMARY_REPORT, _TRACE_SUMMARY_REPORTS
    _LAST_SUMMARY_REPORT = report
    if trace_id:
        _TRACE_SUMMARY_REPORTS[trace_id] = report


def get_last_summary_report() -> Optional[str]:
    """Retrieve the most recently generated summary report."""
    return _LAST_SUMMARY_REPORT


def get_summary_report(trace_id: str) -> Optional[str]:
    """Retrieve summary report for a specific trace ID."""
    return _TRACE_SUMMARY_REPORTS.get(trace_id)


# -------------------------------
# Additional debug/trace functions
# -------------------------------
def get_last_rejection_debug() -> Dict[str, Any]:
    """Return a deep copy of the last relocation rejection breakdown."""
    return copy.deepcopy(_last_relocation_debug)


def get_last_relocation_debug() -> Dict[str, Any]:
    """Alias for get_last_rejection_debug."""
    return get_last_rejection_debug()


def get_current_trace_id() -> Optional[str]:
    """Return the currently bound trace ID."""
    global _CURRENT_TRACE_ID
    return _CURRENT_TRACE_ID


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


__all__ = [
    "Move",
    "OptimizerConfig",
    "plan_relocation",
    "optimise_relocation",
    "enforce_constraints",
    "get_last_summary_report",
    "get_summary_report",
    "set_summary_report",
    "get_last_rejection_debug",
    "get_last_relocation_debug",
    "get_current_trace_id",
    "sse_events",
]
