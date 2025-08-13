# app/services/ai_planner.py
from __future__ import annotations
from typing import Dict, List, Any, Optional
import os, json
import re
from datetime import datetime, timedelta
import pandas as pd

import uuid, time

# ---- JSONL logger (best effort; do not break main flow) ----
AI_LOG_PATH = os.getenv(
    "AI_LOG_JSONL",
    os.path.join(os.path.dirname(__file__), "../../logs/ai_planner.jsonl"),
)

def _jsonl_log(event: dict) -> None:
    try:
        path = os.path.abspath(AI_LOG_PATH)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        base = {
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "app": "warehouse-optimizer",
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({**base, **event}, ensure_ascii=False) + "\n")
    except Exception:
        pass

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # ライブラリ未導入でもインポートエラーで落ちないように

# --- OpenAI client compat (new SDK and legacy) -----------------------------
try:  # new SDK (>=1.0)
    from openai import OpenAI as _OpenAIClient  # type: ignore
except Exception:  # pragma: no cover
    _OpenAIClient = None  # type: ignore
try:  # legacy SDK (<1.0)
    import openai as _openai_legacy  # type: ignore
except Exception:  # pragma: no cover
    _openai_legacy = None  # type: ignore

from app.services.optimizer import _parse_loc8, _parse_lot_date_key, _capacity_limit, OptimizerConfig


def _representative_pack_by_col(inv: pd.DataFrame, pack_map: Optional[pd.Series]) -> Dict[int, float]:
    """列（column）ごとの代表入数（中央値）を推定する。"""
    if pack_map is None or inv.empty:
        return {}
    tmp = inv.copy()
    tmp["pack_est"] = tmp["商品ID"].astype(str).map(pack_map)
    tmp = tmp.dropna(subset=["pack_est"])  # 入数不明SKUは除外
    if tmp.empty:
        return {}
    # 列（col）はロケーションから抽出
    tmp[["lv", "col", "dep"]] = tmp["ロケーション"].apply(lambda s: pd.Series(_parse_loc8(str(s))))
    med = tmp.groupby("col", dropna=True)["pack_est"].median()
    return {int(c): float(v) for c, v in med.items() if pd.notna(v)}


def _shelf_usage(inv: pd.DataFrame, sku_vol_map: pd.Series) -> Dict[str, float]:
    """ロケーション（L/C/Dの8桁キー）ごとの概算使用容積(m3)。"""
    if inv.empty:
        return {}
    key_series = inv["商品ID"].astype(str)
    vol_each = key_series.map(sku_vol_map).fillna(0.0)
    qty_cases = pd.to_numeric(inv.get("cases", inv.get("ケース", 0)), errors="coerce").fillna(0.0)
    # cases 列がないときは入数でケース換算…は optimizer 側でやるのでここはある前提
    vol_total = vol_each * qty_cases
    return inv.assign(_vol=vol_total).groupby("ロケーション")["_vol"].sum().to_dict()


def _pack_map_from_master(sku_master: pd.DataFrame) -> Optional[pd.Series]:
    """SKUマスタから入数マップ（sku_id -> 入数）を作る。"""
    if "入数" not in sku_master.columns:
        return None
    key = "sku_id" if "sku_id" in sku_master.columns else "商品ID"
    s = pd.to_numeric(sku_master["入数"], errors="coerce")
    return s.astype(float).set_axis(sku_master[key].astype(str))


def _carton_volume_map(sku_master: pd.DataFrame) -> pd.Series:
    """SKUマスタから外箱容積(m3)のマップを作る。
    候補列は複数（環境差吸収のため）。未検出時は0.0。
    """
    key = "sku_id" if "sku_id" in sku_master.columns else "商品ID"
    candidates = (
        "carton_volume_m3",
        "volume_m3",
        "商品予備項目００６",
        "商品予備項目006",
        "容積m3",
        "容積m^3",
    )
    vol_col = next((c for c in candidates if c in sku_master.columns), None)
    if vol_col is None:
        return pd.Series(0.0, index=sku_master[key].astype(str))
    s = pd.to_numeric(sku_master[vol_col], errors="coerce").fillna(0.0).astype(float)
    s.index = sku_master[key].astype(str)
    return s


def build_ai_summary(sku_master: pd.DataFrame, inventory: pd.DataFrame, cfg: OptimizerConfig) -> dict:
    """AIに渡す軽量サマリ（列代表入数・列空き容量・SKU別の現状配列）。
    - 空き容量は列内のスロット数（distinct (level, depth)）× 棚容量（fill rate適用）から、
      在庫の概算使用量を引いた“粗い”推定。列内全スロット同容量という近似。
    """
    inv = inventory.copy()
    # ロットキー付与（AIが順序理解しやすいよう簡略）
    inv["lot_key"] = inv["ロット"].map(_parse_lot_date_key)
    inv[["lv", "col", "dep"]] = inv["ロケーション"].apply(lambda s: pd.Series(_parse_loc8(str(s))))

    pack_map = _pack_map_from_master(sku_master)
    vol_map = _carton_volume_map(sku_master)
    cap_per_slot = _capacity_limit(getattr(cfg, "fill_rate", None))  # 1スロットあたりの上限(m3)

    # 列代表入数
    rep_pack_by_col = _representative_pack_by_col(inv, pack_map)

    # 列ごとの概算使用量(m3)とスロット数
    shelf_use = _shelf_usage(inv, vol_map)  # key: ロケーション(L/C/D)
    col_use_approx: Dict[int, float] = {}
    col_slots: Dict[int, set] = {}
    for loc, used in shelf_use.items():
        lv, col, dep = _parse_loc8(loc)
        col_use_approx[col] = col_use_approx.get(col, 0.0) + float(used)
        col_slots.setdefault(col, set()).add((lv, dep))
    col_slots_count = {c: len(s) for c, s in col_slots.items()}

    # SKUごとの現状（列→推定ケース数）と入数
    qty_cases = pd.to_numeric(inv.get("cases", inv.get("ケース", 1)), errors="coerce").fillna(1.0)
    inv = inv.assign(_cases=qty_cases)

    sku_rows: List[Dict[str, Any]] = []
    for sku, g in inv.groupby("商品ID"):
        cols = g.groupby("col")["_cases"].sum().to_dict()
        # lot_key の None を除外して安全に整数化
        lots = sorted({int(x) for x in g["lot_key"].tolist() if pd.notna(x)})
        pack_est = None
        if pack_map is not None:
            v = pack_map.get(str(sku))
            if pd.notna(v):
                pack_est = float(v)
        sku_rows.append({
            "sku_id": str(sku),
            "pack_est": pack_est,
            "current_cols": {int(c): float(v) for c, v in cols.items()},
            "lots": lots,
        })

    # 列側のサマリ（代表入数が未推定の列も含め、現存する列は全て載せる）
    all_cols = sorted({int(c) for c in inv["col"].dropna().astype(int).unique().tolist()})
    columns: List[Dict[str, Any]] = []
    for c in all_cols:
        used = float(col_use_approx.get(c, 0.0))
        slots = int(col_slots_count.get(c, max(1, inv.loc[inv["col"] == c, ["lv", "dep"]].drop_duplicates().shape[0])))
        cap_total = float(cap_per_slot) * max(1, slots)
        rep_pack = rep_pack_by_col.get(c)
        columns.append({
            "col": int(c),
            "rep_pack": (float(rep_pack) if rep_pack is not None else None),
            "rough_used_m3": used,
            "rough_free_m3": max(0.0, cap_total - used),
            "slots": max(1, slots),
        })

    return {
        "config": {
            "pack_tolerance_ratio": getattr(cfg, "pack_tolerance_ratio", 0.10),
            "prefer_same_column_bonus": getattr(cfg, "prefer_same_column_bonus", 5.0),
            "same_sku_same_column_bonus": getattr(cfg, "same_sku_same_column_bonus", 20.0),
        },
        "columns": columns,
        "skus": sku_rows,
    }


def draft_relocation_with_ai(
    sku_master: pd.DataFrame,
    inventory: pd.DataFrame,
    cfg: OptimizerConfig,
    *,
    model: Optional[str] = None,
    temperature: float = 0.1,
    topk: int = 3,
) -> Dict[str, List[int]]:
    """
    AIに列の優先リスト（SKU -> [col,...]）を出してもらう。
    失敗時は {} を返す（ロジック単独で進行）。
    """
    try:
        summary = build_ai_summary(sku_master, inventory, cfg)
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            print("[ai_planner] OPENAI_API_KEY missing; returning empty column preferences.")
            return {}

        mdl = (model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
        temp = float(temperature)
        if mdl.startswith("gpt-5"):
            # gpt-5 は temperature パラメータを送らない（デフォルト適用）
            temp = None

        system = (
            "You are a warehouse slotting planner. "
            "Group SKUs by similar case pack (±10%), preserve current column mapping if it already clusters well, "
            "and output preferred columns per SKU. Do NOT violate FIFO-by-level: older lots must be on lower levels."
        )
        user = (
            "Return JSON only with the shape: "
            '{"preferred_columns": {"<sku_id>":[<col>, ... up to %d]}}. '
            "Rank columns by pack similarity to each column's representative pack and by keeping SKUs in existing columns if reasonable. "
            "Do not invent new columns.\n\n"
            "INPUT:\n%s" % (topk, json.dumps(summary, ensure_ascii=False)[:180000])  # 念のため上限
        )

        # gpt-5 系は response_format 未対応の可能性があるため、フォーマット指示はプロンプトに埋め込む
        resp_fmt = {"type": "json_object"}
        if mdl.startswith("gpt-5"):
            resp_fmt = None
            user = user + "\n\nIMPORTANT: Respond with a single JSON object only. No markdown, no explanations."

        print(f"[ai_planner] columns: model={mdl}, temp={'None' if temp is None else temp}, user_len={len(user)}")

        trace_id_columns = uuid.uuid4().hex[:12]
        _jsonl_log({
            "event": "ai.call",
            "phase": "columns",
            "trace_id": trace_id_columns,
            "model": mdl,
            "temperature": (None if temp is None else float(temp)),
            "user_len": len(user),
            "topk": int(topk),
        })

        try:
            t0 = time.time()
            text = _chat_complete_compat(
                api_key=api_key,
                model=mdl,
                temperature=temp,
                system=system,
                user=user,
                response_format=resp_fmt,
            )
            lat_ms = int((time.time() - t0) * 1000)
            _jsonl_log({
                "event": "ai.response",
                "phase": "columns",
                "trace_id": trace_id_columns,
                "lat_ms": lat_ms,
                "resp_len": (len(text) if isinstance(text, str) else None),
                "resp_head": (text[:800] if isinstance(text, str) else None),
            })
        except Exception as e:
            print(f"[ai_planner] chat completion failed (columns): {e}")
            _jsonl_log({
                "event": "ai.error",
                "phase": "columns",
                "error": str(e)[:400],
            })
            return {}

        if not text or not str(text).strip():
            print("[ai_planner] columns: empty completion text")
            return {}

        data, cleaned, err = _parse_json_relaxed(text)
        if data is None:
            _jsonl_log({
                "event": "ai.parse_error",
                "phase": "columns",
                "trace_id": trace_id_columns,
                "error": str(err)[:400] if err else "unknown",
            })
            print(f"[ai_planner] columns: JSON parse fail: {err}")
            return {}
        else:
            cnt = 0
            if isinstance(data, dict):
                pref = data.get("preferred_columns")
                if isinstance(pref, dict):
                    cnt = sum(1 for _ in pref.keys())
            _jsonl_log({
                "event": "ai.parse_ok",
                "phase": "columns",
                "trace_id": trace_id_columns,
                "count": int(cnt),
            })

        pref = data.get("preferred_columns") if isinstance(data, dict) else None
        out: Dict[str, List[int]] = {}
        if isinstance(pref, dict):
            for sku, cols in pref.items():
                try:
                    arr = [int(c) for c in cols][:topk]
                    if arr:
                        out[str(sku)] = arr
                except Exception:
                    continue
            return out
        else:
            if isinstance(data, dict):
                print(f"[ai_planner] columns: no 'preferred_columns' key; keys={list(data.keys())}")
            else:
                print(f"[ai_planner] columns: unexpected top-level type: {type(data).__name__}")
            return {}
    except Exception as e:
        print(f"[ai_planner] columns: unexpected error: {e}")
        _jsonl_log({
            "event": "ai.error",
            "phase": "columns",
            "error": str(e)[:400],
        })
        return {}


# Wrapper for _maybe_call_ai_planner in upload.py
def ai_rank_columns_for_skus(
    sku_df: pd.DataFrame,
    inv_df: pd.DataFrame,
    block_codes: list[str] | None = None,
    quality_names: list[str] | None = None,
    max_candidates: int = 3,
) -> Dict[str, List[int]]:
    """_maybe_call_ai_planner から呼ばれる想定のラッパー。
    - 引数は upload.py 側の呼び出しシグネチャに合わせる。
    - block_codes / quality_names は inv_df 側で既に適用済み想定のためここでは使用しない。
    - OPENAI_MODEL が設定されていればそれを優先してモデル指定する。
    """
    cfg = OptimizerConfig()

    # モデル指定は環境変数を優先（未設定なら draft_relocation_with_ai のデフォルトを使用）
    kwargs: Dict[str, Any] = {"topk": int(max_candidates)}
    env_model = os.getenv("OPENAI_MODEL")
    if env_model:
        kwargs["model"] = env_model

    try:
        return draft_relocation_with_ai(
            sku_master=sku_df,
            inventory=inv_df,
            cfg=cfg,
            **kwargs,
        )
    except Exception:
        # 失敗時はヒューリスティック側にフォールバックさせる
        return {}


# === AI Main Planner: summary for moves, draft & revise =====================

ANCHOR_HOMOGENEITY_THRESHOLD = 0.7  # 入数帯域の同質性
ANCHOR_UTILIZATION_THRESHOLD = 0.6  # 使用率
MAX_INPUT_JSON_LEN = 180_000

# --- tolerant JSON parsing helpers -----------------------------------------
def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl != -1:
            s = s[nl+1:]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()

def _extract_first_json_blob(s: str) -> Optional[str]:
    """Return the first balanced top-level JSON object/array substring, if any."""
    if not s:
        return None
    start_obj = s.find('{')
    start_arr = s.find('[')
    starts = [i for i in (start_obj, start_arr) if i != -1]
    if not starts:
        return None
    i0 = min(starts)
    depth = 0
    in_str = False
    quote = ''
    esc = False
    for i, ch in enumerate(s[i0:], start=i0):
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == quote:
                in_str = False
        else:
            if ch == '"' or ch == "'":
                in_str = True
                quote = ch
            elif ch in '{[':
                depth += 1
            elif ch in '}]':
                depth -= 1
                if depth == 0:
                    return s[i0:i+1]
    return None

def _parse_json_relaxed(text: str):
    """Try strict json.loads, then code-fence stripping, then blob extraction.
    Returns (data, cleaned_text, error_msg). error_msg is None on success.
    """
    if text is None:
        return None, None, 'none'
    s = str(text)
    try:
        return json.loads(s), s, None
    except Exception:
        pass
    s2 = _strip_code_fences(s)
    if s2 != s:
        try:
            return json.loads(s2), s2, None
        except Exception:
            pass
    blob = _extract_first_json_blob(s2)
    if blob:
        try:
            return json.loads(blob), blob, None
        except Exception as e3:
            return None, None, f'parse error after extract: {e3}; head={s2[:200]}'
    return None, None, f'parse error: head={s[:200]}'

def _chat_complete_compat(*, api_key: str, model: str, temperature: float | None,
                          system: str, user: str, response_format: dict | None = None) -> str:
    """Create a chat completion using either the new or legacy OpenAI SDK.
    Returns the assistant text content. Raises RuntimeError on failure.
    """
    last_err: Exception | None = None

    # New SDK path
    if _OpenAIClient is not None:
        try:
            client = _OpenAIClient(api_key=api_key)
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
            if temperature is not None:
                kwargs["temperature"] = float(temperature)
            if response_format:
                kwargs["response_format"] = response_format
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content  # type: ignore[attr-defined]
        except Exception as e:  # pragma: no cover
            print(f"[ai_planner] new-sdk chat failed: {e}")
            last_err = e

    # Legacy SDK path
    if _openai_legacy is not None:
        try:
            _openai_legacy.api_key = api_key
            legacy_kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
            if temperature is not None:
                legacy_kwargs["temperature"] = float(temperature)
            resp = _openai_legacy.ChatCompletion.create(**legacy_kwargs)
            return resp["choices"][0]["message"]["content"]  # type: ignore[index]
        except Exception as e:  # pragma: no cover
            print(f"[ai_planner] legacy-sdk chat failed: {e}")
            last_err = e

    raise RuntimeError(f"No OpenAI client available or both clients failed: {last_err}")


def _calc_turnover_scores(ship: pd.DataFrame, rotation_window_days: int) -> Dict[str, float]:
    """出荷データからSKUの回転スコア(0..1 正規化)を計算。"""
    if ship is None or ship.empty or "trandate" not in ship.columns:
        return {}
    df = ship.copy()
    df["trandate"] = pd.to_datetime(df["trandate"], errors="coerce")
    df = df.dropna(subset=["trandate"])  # 不正日付は除外
    if df.empty:
        return {}
    end = df["trandate"].max()
    start = end - timedelta(days=int(rotation_window_days or 90))
    df = df[df["trandate"] >= start]
    if df.empty:
        return {}
    qty_col = "item_shipquantity" if "item_shipquantity" in df.columns else "qty"
    df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0.0)
    grp = df.groupby(df["item_internalid"].astype(str))[qty_col].sum()
    # 1日あたり換算
    days = max(1, int(rotation_window_days or 90))
    rate = grp / float(days)
    if (rate.max() - rate.min()) <= 0:
        return {k: float(0.0) for k in rate.index.astype(str)}
    norm = (rate - rate.min()) / (rate.max() - rate.min())
    return {str(k): float(v) for k, v in norm.items()}


def _compute_column_features(inv: pd.DataFrame, sku_master: pd.DataFrame, cfg: OptimizerConfig) -> List[Dict[str, Any]]:
    """列ごとの代表入数/使用量/スロット/同質性/使用率/アンカー/入口ランク等を算出。"""
    if inv.empty:
        return []
    inv = inv.copy()
    inv["lot_key"] = inv["ロット"].map(_parse_lot_date_key)
    inv[["lv", "col", "dep"]] = inv["ロケーション"].apply(lambda s: pd.Series(_parse_loc8(str(s))))

    pack_map = _pack_map_from_master(sku_master)
    vol_map = _carton_volume_map(sku_master)
    cap_per_slot = _capacity_limit(getattr(cfg, "fill_rate", None))

    # 列代表入数
    rep_pack_by_col = _representative_pack_by_col(inv, pack_map)

    # 使用量(m3)とスロット数
    shelf_use = _shelf_usage(inv, vol_map)  # loc -> used m3
    col_use_approx: Dict[int, float] = {}
    col_slots: Dict[int, set] = {}
    col_min_depth: Dict[int, int] = {}
    for loc, used in shelf_use.items():
        lv, col, dep = _parse_loc8(loc)
        col_use_approx[col] = col_use_approx.get(col, 0.0) + float(used)
        col_slots.setdefault(col, set()).add((lv, dep))
        col_min_depth[col] = min(dep, col_min_depth.get(col, dep))
    col_slots_count = {c: len(s) for c, s in col_slots.items()}

    # 入数同質性（代表値±tolの帯域内の割合：cases重みは簡易に均等扱い）
    tol = getattr(cfg, "pack_tolerance_ratio", 0.10)
    packs: Dict[str, float] = {}
    if pack_map is not None:
        packs = {str(k): float(v) for k, v in pack_map.items() if pd.notna(v)}

    inv_pack = inv.copy()
    inv_pack["pack"] = inv_pack["商品ID"].astype(str).map(packs).fillna(0.0)
    columns: List[Dict[str, Any]] = []
    all_cols = sorted({int(c) for c in inv_pack["col"].dropna().astype(int).unique().tolist()})
    for c in all_cols:
        used = float(col_use_approx.get(c, 0.0))
        slots = int(col_slots_count.get(c, max(1, inv_pack.loc[inv_pack["col"] == c, ["lv", "dep"]].drop_duplicates().shape[0])))
        cap_total = float(cap_per_slot) * max(1, slots)
        rep_pack = rep_pack_by_col.get(c, None)
        sub = inv_pack[inv_pack["col"] == c]
        if rep_pack is not None and not sub.empty:
            band = rep_pack * tol
            in_band = ((sub["pack"] >= (rep_pack - band)) & (sub["pack"] <= (rep_pack + band)))
            homogeneity = float(in_band.mean()) if len(sub) else 0.0
        else:
            homogeneity = 0.0
        utilization = float(used / cap_total) if cap_total > 0 else 0.0
        is_anchor = (homogeneity >= ANCHOR_HOMOGENEITY_THRESHOLD) and (utilization >= ANCHOR_UTILIZATION_THRESHOLD)
        columns.append({
            "col": int(c),
            "rep_pack": (float(rep_pack) if rep_pack is not None else None),
            "rough_used_m3": used,
            "rough_free_m3": max(0.0, cap_total - used),
            "slots": max(1, slots),
            "homogeneity_ratio": homogeneity,
            "utilization_ratio": utilization,
            "is_anchor": bool(is_anchor),
            "stay_put_weight": float(homogeneity * utilization),
            "entrance_rank": int(c),  # 列番号が若いほど入口に近いという前提
            "min_depth": int(col_min_depth.get(c, 0)),
        })
    return columns


def _sku_cohesion_stats(inv: pd.DataFrame, pack_map: Optional[pd.Series]) -> List[Dict[str, Any]]:
    """SKUごとの列分布・凝集度・lot順等を構築。"""
    if inv.empty:
        return []
    inv = inv.copy()
    inv["lot_key"] = inv["ロット"].map(_parse_lot_date_key)
    inv[["lv", "col", "dep"]] = inv["ロケーション"].apply(lambda s: pd.Series(_parse_loc8(str(s))))
    qty_cases = pd.to_numeric(inv.get("cases", inv.get("ケース", 1)), errors="coerce").fillna(1.0)
    inv = inv.assign(_cases=qty_cases)
    rows: List[Dict[str, Any]] = []
    for sku, g in inv.groupby("商品ID"):
        by_col = g.groupby("col")["_cases"].sum().to_dict()
        total = float(sum(by_col.values()) or 0.0)
        primary_col = None
        cohesion = 0.0
        if total > 0 and by_col:
            primary_col = int(max(by_col, key=by_col.get))
            cohesion = float(max(by_col.values()) / total)
        lots = sorted({int(x) for x in g["lot_key"].tolist() if pd.notna(x)})
        pack_est = None
        if pack_map is not None:
            v = pack_map.get(str(sku))
            if pd.notna(v):
                pack_est = float(v)
        rows.append({
            "sku_id": str(sku),
            "pack_est": pack_est,
            "current_cols": {int(k): float(v) for k, v in by_col.items()},
            "current_primary_col": primary_col,
            "cohesion_score": cohesion,
            "lots_sorted": lots,
        })
    return rows


def build_ai_summary_for_moves(
    sku_master: pd.DataFrame,
    inventory: pd.DataFrame,
    ship: Optional[pd.DataFrame],
    cfg: OptimizerConfig,
    *,
    rotation_window_days: int = 90,
) -> dict:
    """AIメイン用の詳細サマリを生成。列特徴/アンカー/回転/凝集を含む。"""
    inv = inventory.copy()
    pack_map = _pack_map_from_master(sku_master)
    turnover = _calc_turnover_scores(ship, rotation_window_days)
    columns = _compute_column_features(inv, sku_master, cfg)
    skus = _sku_cohesion_stats(inv, pack_map)

    # SKUの体積（1ケース）も同梱
    vol_map = _carton_volume_map(sku_master)
    vol_case = {str(k): float(v) for k, v in vol_map.items()}

    # turnover を紐付け
    for row in skus:
        row["turnover_score"] = float(turnover.get(row["sku_id"], 0.0))
        row["volume_case_m3"] = float(vol_case.get(row["sku_id"], 0.0))

    cap_per_slot = _capacity_limit(getattr(cfg, "fill_rate", None))

    # --- lots_full per SKU (original lot strings and from_locs) -------------
    inv2 = inventory.copy()
    inv2["lot_key"] = inv2["ロット"].map(_parse_lot_date_key)
    inv2[["lv", "col", "dep"]] = inv2["ロケーション"].apply(lambda s: pd.Series(_parse_loc8(str(s))))

    lot_agg = (
        inv2.groupby(["商品ID", "ロット"], dropna=False)
        .agg(lot_key=("lot_key", "max"),
             from_locs=("ロケーション", lambda s: sorted({str(x) for x in s})))
        .reset_index()
    )

    def _key_to_datestr(k):
        try:
            ik = int(k)
            return f"{ik:08d}" if 20000101 <= ik <= 20991231 else None
        except Exception:
            return None

    lots_map: Dict[str, List[Dict[str, Any]]] = {}
    for _, r in lot_agg.iterrows():
        sku = str(r["商品ID"])
        lots_map.setdefault(sku, []).append({
            "lot": str(r["ロット"]),
            "lot_date": _key_to_datestr(r["lot_key"]),
            "from_locs": [str(x) for x in r["from_locs"]],
        })

    # attach lots_full to each SKU row
    for row in skus:
        row["lots_full"] = lots_map.get(row["sku_id"], [])

    # placeholder locations to avoid (exact list + prefixes)
    placeholders_exact = sorted({
        str(x) for x in inv["ロケーション"].astype(str).unique()
        if str(x).startswith("000000") or str(x).startswith("222222")
    })

    return {
        "constraints": {
            "slot_capacity_m3": float(cap_per_slot),
            "fill_rate": float(getattr(cfg, "fill_rate", 0.90) or 0.90),
            "pack_tolerance_ratio": float(getattr(cfg, "pack_tolerance_ratio", 0.10) or 0.10),
            "anchor_homogeneity_threshold": float(ANCHOR_HOMOGENEITY_THRESHOLD),
            "anchor_utilization_threshold": float(ANCHOR_UTILIZATION_THRESHOLD),
        },
        "columns": columns,
        "skus": skus,
        "server_notes": {
            "to_col_ok": True,
            "slot_capacity_enforced_server_side": True,
            "notes": "When only to_col is provided, the backend will choose a concrete to_loc that respects per-slot capacity and fill_rate. Prefer columns with rough_free_m3 > 0 and matching rep_pack."
        },
        "placeholders": {
            "exact": placeholders_exact,
            "deny_prefix": ["000000", "222222"]
        },
    }


def draft_moves_with_ai(
    sku_master: pd.DataFrame,
    inventory: pd.DataFrame,
    ship: Optional[pd.DataFrame],
    recv: Optional[pd.DataFrame],
    cfg: OptimizerConfig,
    *,
    block_codes: Optional[List[str]] = None,
    quality_names: Optional[List[str]] = None,
    rotation_window_days: int = 90,
    max_moves: int = 50,
    model: Optional[str] = None,
    temperature: float = 0.1,
) -> List[Dict[str, Any]]:
    """AIが直接 Move 配列を返す（列まで指定OK、to_locはサーバ側で補完可能）。
    返却スキーマ: [{sku_id, lot, qty_cases, from_loc, to_col|to_loc}, ...]
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            print("[ai_planner] OPENAI_API_KEY missing; returning empty plan (fallback to Greedy).")
            return []
        summary = build_ai_summary_for_moves(
            sku_master=sku_master,
            inventory=inventory,
            ship=ship,
            cfg=cfg,
            rotation_window_days=int(rotation_window_days or 90),
        )
        mdl = (model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
        temp = float(temperature)
        if mdl.startswith("gpt-5"):
            temp = None

        schema_note = (
            "Return ONLY JSON with a single top-level object containing key 'moves' whose value is an array of objects.\n"
            "Each move object MUST have: sku_id (string), lot (string), qty_cases (integer > 0), from_loc (8-digit string), and to_col (integer). Do NOT output to_loc; the server will choose a concrete slot.\n"
            "The 'lot' MUST be exactly one of summary.skus[].lots_full[].lot for that sku_id. The 'from_loc' MUST be one of the corresponding lots_full[].from_locs. Never use placeholder locations listed under summary.placeholders.\n"
            "Total number of move objects MUST be <= max_moves. Never return an empty 'moves' array when any SKU appears in multiple columns or when FIFO violations exist."
        )

        system = (
            "You are a warehouse slotting planner. Primary goals: (1) co-locate the same SKU within a single column and enforce FIFO by levels (older lots on lower levels), (2) preserve anchor columns (high pack homogeneity and utilization), (3) minimize work cost (touched locations / evictions / moved cases).\n"
            "Secondary: within those constraints, place higher-turnover SKUs closer to the entrance (smaller column numbers and smaller depth values).\n"
            "Hard constraints: Do not exceed max_moves. Output to_col ONLY (no to_loc). Use only real lots and from_locs listed in summary.skus[].lots_full, and avoid placeholder locations (summary.placeholders). Prefer columns with rough_free_m3 > 0 and representative pack within ±pack_tolerance_ratio.\n"
            "Do not move SKUs with unknown lot dates. Do not invent columns or locations. Always return at least a few moves when consolidation or FIFO fixes are possible."
        )

        user_payload = {
            "block_filter": list(block_codes or []),
            "quality_filter": list(quality_names or []),
            "max_moves": int(max_moves),
            "rotation_window_days": int(rotation_window_days or 90),
            "summary": summary,
        }
        user = (
            f"{schema_note}\n\nINPUT:\n" + json.dumps(user_payload, ensure_ascii=False)[:MAX_INPUT_JSON_LEN]
        )

        resp_fmt = {"type": "json_object"}
        if mdl.startswith("gpt-5"):
            resp_fmt = None
            user = user + "\n\nIMPORTANT: Respond with a single JSON object only. No markdown, no explanations."

        print(f"[ai_planner] moves: model={mdl}, temp={'None' if temp is None else temp}, user_len={len(user)}")

        trace_id_moves = uuid.uuid4().hex[:12]
        _jsonl_log({
            "event": "ai.call",
            "phase": "moves",
            "trace_id": trace_id_moves,
            "model": mdl,
            "temperature": (None if temp is None else float(temp)),
            "user_len": len(user),
            "max_moves": int(max_moves),
            "rotation_window_days": int(rotation_window_days or 90),
        })

        try:
            t0 = time.time()
            text = _chat_complete_compat(
                api_key=api_key,
                model=mdl,
                temperature=temp,
                system=system,
                user=user,
                response_format=resp_fmt,
            )
            lat_ms = int((time.time() - t0) * 1000)
            _jsonl_log({
                "event": "ai.response",
                "phase": "moves",
                "trace_id": trace_id_moves,
                "lat_ms": lat_ms,
                "resp_len": (len(text) if isinstance(text, str) else None),
                "resp_head": (text[:800] if isinstance(text, str) else None),
            })
        except Exception as e:
            print(f"[ai_planner] chat completion failed (moves): {e}")
            _jsonl_log({
                "event": "ai.error",
                "phase": "moves",
                "error": str(e)[:400],
            })
            return []

        if not text or not str(text).strip():
            print("[ai_planner] moves: empty completion text")
            return []
        data, cleaned, err = _parse_json_relaxed(text)
        if data is None:
            head = (text[:800] + '...') if isinstance(text, str) and len(text) > 800 else (text if isinstance(text, str) else None)
            _jsonl_log({
                "event": "ai.parse_error",
                "phase": "moves",
                "trace_id": trace_id_moves,
                "error": (str(err)[:400] if err else "unknown"),
                "resp_head": head,
            })
            print(f"[ai_planner] moves: JSON parse fail: {err}; head={head}")
            return []
        else:
            moves_len = 0
            if isinstance(data, dict) and isinstance(data.get("moves"), list):
                moves_len = len(data.get("moves"))
            _jsonl_log({
                "event": "ai.parse_ok",
                "phase": "moves",
                "trace_id": trace_id_moves,
                "count": int(moves_len),
            })
        moves = data.get("moves") if isinstance(data, dict) else None
        if isinstance(moves, list) and len(moves) > 0:
            normalized: List[Dict[str, Any]] = []
            for m in moves:
                if not isinstance(m, dict):
                    continue
                mm = dict(m)
                # allow 'qty' fallback
                if "qty_cases" not in mm and "qty" in mm:
                    try:
                        mm["qty_cases"] = int(mm.pop("qty"))
                    except Exception:
                        pass
                # prefer to_col only
                if "to_col" in mm and "to_loc" in mm:
                    mm.pop("to_loc", None)
                normalized.append(mm)
            return normalized
        if isinstance(moves, list) and len(moves) == 0:
            print("[ai_planner] moves: 'moves' present but empty (model returned zero).")
            return []
        if isinstance(data, list) and len(data) > 0:
            return data
        if isinstance(data, list) and len(data) == 0:
            print("[ai_planner] moves: top-level array but empty")
            return []
        if isinstance(data, dict):
            print(f"[ai_planner] moves: no 'moves' key; keys={list(data.keys())}")
        else:
            print(f"[ai_planner] moves: unexpected top-level type: {type(data).__name__}")
        return []
    except Exception as e:
        print(f"[ai_planner] moves: unexpected error: {e}")
        _jsonl_log({
            "event": "ai.error",
            "phase": "moves",
            "error": str(e)[:400],
        })
        return []


def revise_moves_with_feedback(
    prev_moves: List[Dict[str, Any]],
    violations: Dict[str, Any],
    remaining_budget: int,
    context: Dict[str, Any],
    *,
    model: Optional[str] = None,
    temperature: float = 0.1,
) -> List[Dict[str, Any]]:
    """監査の違反サマリを与えて再提案を得る。返却は Move 配列。"""
    try:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            print("[ai_planner] OPENAI_API_KEY missing; cannot revise.")
            return []
        mdl = (model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
        temp = float(temperature)
        if mdl.startswith("gpt-5"):
            temp = None

        system = (
            "You are revising a relocation plan for a warehouse. Fix violations strictly while keeping moves minimal. Respect FIFO by levels, preserve anchor columns, and minimize work cost. Output JSON only with key 'moves' as described."
        )
        user_payload = {
            "previous_moves": prev_moves,
            "violations": violations,
            "remaining_budget": int(remaining_budget),
            "context": context,
        }
        schema_note = (
            "Return ONLY JSON with a top-level object containing key 'moves' (array of move objects with fields: sku_id, lot, qty_cases, from_loc, and to_col|to_loc). The number of moves MUST be <= remaining_budget."
        )
        user = schema_note + "\n\nINPUT:\n" + json.dumps(user_payload, ensure_ascii=False)[:MAX_INPUT_JSON_LEN]

        resp_fmt = {"type": "json_object"}
        if mdl.startswith("gpt-5"):
            resp_fmt = None
            user = user + "\n\nIMPORTANT: Respond with a single JSON object only. No markdown, no explanations."

        print(f"[ai_planner] revise: model={mdl}, temp={'None' if temp is None else temp}, user_len={len(user)}")

        try:
            text = _chat_complete_compat(
                api_key=api_key,
                model=mdl,
                temperature=temp,
                system=system,
                user=user,
                response_format=resp_fmt,
            )
        except Exception as e:
            print(f"[ai_planner] chat completion failed (revise): {e}")
            return []

        if not text or not str(text).strip():
            print("[ai_planner] moves: empty completion text")
            return []
        data, cleaned, err = _parse_json_relaxed(text)
        if data is None:
            print(f"[ai_planner] moves: JSON parse fail: {err}")
            return []
        moves = data.get("moves") if isinstance(data, dict) else None
        if isinstance(moves, list) and len(moves) > 0:
            return moves
        if isinstance(moves, list) and len(moves) == 0:
            print("[ai_planner] moves: 'moves' present but empty")
            return []
        if isinstance(data, list) and len(data) > 0:
            return data
        if isinstance(data, list) and len(data) == 0:
            print("[ai_planner] moves: top-level array but empty")
            return []
        if isinstance(data, dict):
            print(f"[ai_planner] moves: no 'moves' key; keys={list(data.keys())}")
        else:
            print(f"[ai_planner] moves: unexpected top-level type: {type(data).__name__}")
        return []
    except Exception as e:
        print(f"[ai_planner] revise: unexpected error: {e}")
        return []