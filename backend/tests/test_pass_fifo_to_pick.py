"""Unit tests for _pass_fifo_to_pick (新統合FIFOパス)."""
import pytest
import pandas as pd
from typing import Dict, Set, Tuple, Optional


def _make_cfg(**kwargs):
    from app.services.optimizer import OptimizerConfig
    cfg = OptimizerConfig()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def _make_inv(rows):
    """在庫 DataFrame を構築するヘルパー。

    各 row は dict で以下のキーを持つ:
      sku, lot, lot_key, loc, lv, col, dep, qty, vol_each, is_movable(optional)
    """
    records = []
    for r in rows:
        loc = str(r["loc"]).zfill(8)
        lv, col, dep = int(loc[0:3]), int(loc[3:6]), int(loc[6:8])
        records.append({
            "商品ID": str(r["sku"]),
            "ロット": str(r.get("lot", "")),
            "lot_key": int(r["lot_key"]),
            "ロケーション": loc,
            "lv": lv,
            "col": col,
            "dep": dep,
            "qty_cases_move": int(r.get("qty", 1)),
            "volume_each_case": float(r.get("vol_each", 1.0)),
            "is_movable": bool(r.get("is_movable", True)),
        })
    return pd.DataFrame(records)


def _build_shelf_usage(inv: pd.DataFrame, cap_limit: float = 100.0) -> Dict[str, float]:
    """在庫から shelf_usage を構築するヘルパー。"""
    usage: Dict[str, float] = {}
    for _, row in inv.iterrows():
        loc = str(row["ロケーション"]).zfill(8)
        vol = float(row["volume_each_case"]) * int(row["qty_cases_move"])
        usage[loc] = usage.get(loc, 0.0) + vol
    # 全ロケに cap_limit 以上のエントリを持たせる（空きスロット候補として必要）
    return usage


def _add_empty_locs(shelf_usage: Dict[str, float], locs):
    """空き(usage=0)のロケーションをshelf_usageに追加する。"""
    for loc in locs:
        loc_z = str(loc).zfill(8)
        if loc_z not in shelf_usage:
            shelf_usage[loc_z] = 0.0


def _call(inv, shelf_usage, cfg=None, **kwargs):
    """_pass_fifo_to_pick を呼び出すヘルパー。"""
    from app.services.optimizer import _pass_fifo_to_pick
    if cfg is None:
        cfg = _make_cfg()
    return _pass_fifo_to_pick(inv, shelf_usage, 100.0, cfg, **kwargs)


# ============================================================================
# テストケース 1: 基本 - 同列スワップ1件
# ============================================================================
def test_case1_same_column_swap():
    """SKU 1つ、2ロット、Lv3古 / Lv1新 / 同列 → 同列スワップ 1件"""
    inv = _make_inv([
        {"sku": "A", "lot": "L1", "lot_key": 20250101, "loc": "00300101", "qty": 2, "vol_each": 1.0},  # Lv3, col1 古
        {"sku": "A", "lot": "L2", "lot_key": 20250601, "loc": "00100101", "qty": 2, "vol_each": 1.0},  # Lv1, col1 新
    ])
    shelf_usage = _build_shelf_usage(inv)
    moves = _call(inv, shelf_usage)

    assert len(moves) == 2
    # chain_group_id が同じこと (スワップペア)
    assert moves[0].chain_group_id == moves[1].chain_group_id
    assert moves[0].chain_group_id is not None
    # execution_order
    orders = sorted(m.execution_order for m in moves)
    assert orders == [1, 2]
    # 古ロット(L1)が Lv1へ、新ロット(L2)が Lv3へ
    move_by_lot = {m.lot: m for m in moves}
    assert move_by_lot["L1"].to_loc.startswith("001")  # Lv1
    assert move_by_lot["L2"].to_loc.startswith("003")  # Lv3


# ============================================================================
# テストケース 2: 空きスロット配置
# ============================================================================
def test_case2_empty_slot_placement():
    """全行Lv3-4、Lv1-2に空きあり → 古行移動"""
    inv = _make_inv([
        {"sku": "B", "lot": "L1", "lot_key": 20250101, "loc": "00300201", "qty": 1, "vol_each": 2.0},  # Lv3 古
        {"sku": "B", "lot": "L2", "lot_key": 20250901, "loc": "00400201", "qty": 1, "vol_each": 2.0},  # Lv4 新
    ])
    shelf_usage = _build_shelf_usage(inv)
    # Lv1の空きスロットを追加
    empty_loc = "00100201"  # Lv1, col2
    _add_empty_locs(shelf_usage, [empty_loc])

    moves = _call(inv, shelf_usage)

    # 少なくとも1件の移動が生成されること
    assert len(moves) >= 1
    # 古ロット(L1)がLv1-2に移動していること
    ancient_moves = [m for m in moves if m.lot == "L1"]
    assert len(ancient_moves) >= 1
    assert ancient_moves[0].to_loc.startswith("001") or ancient_moves[0].to_loc.startswith("002")


# ============================================================================
# テストケース 3: pack帯soft - 帯内優先
# ============================================================================
def test_case3_pack_band_soft_prefers_in_band():
    """候補列2つ、1つはpack帯内 → 帯内が優先選択"""
    inv = _make_inv([
        # pack=10 の SKU (SKUのpack_per_caseカラムが必要なので、inv作成後に追加)
        {"sku": "C", "lot": "L1", "lot_key": 20250101, "loc": "00300501", "qty": 1, "vol_each": 1.0},
        {"sku": "C", "lot": "L2", "lot_key": 20251201, "loc": "00400501", "qty": 1, "vol_each": 1.0},
    ])
    inv["pack_per_case"] = 10.0  # 入数 = 10

    shelf_usage = _build_shelf_usage(inv)
    # 2つの空きLv1スロット: col5 (帯内: rep=10), col20 (帯外: rep=50)
    _add_empty_locs(shelf_usage, ["00100501", "00102001"])

    # rep_pack_by_col: col5=10(帯内), col20=50(帯外)
    rep_pack_by_col = {5: 10.0, 20: 50.0}
    cfg = _make_cfg(pack_tolerance_ratio=0.10)

    moves = _call(inv, shelf_usage, cfg=cfg, rep_pack_by_col=rep_pack_by_col)

    # 生成された移動がある
    assert len(moves) >= 1
    # 古ロット(L1)がcol5(帯内)に移動していること
    l1_moves = [m for m in moves if m.lot == "L1"]
    assert len(l1_moves) >= 1
    # to_loc の col部分が '005' であること
    assert l1_moves[0].to_loc[3:6] == "005"


# ============================================================================
# テストケース 4: pack帯soft - 帯外でも選択
# ============================================================================
def test_case4_pack_band_soft_out_of_band_ok():
    """全候補がpack帯外 → それでも帯外選択(soft)"""
    inv = _make_inv([
        {"sku": "D", "lot": "L1", "lot_key": 20250101, "loc": "00300501", "qty": 1, "vol_each": 1.0},
        {"sku": "D", "lot": "L2", "lot_key": 20251201, "loc": "00400501", "qty": 1, "vol_each": 1.0},
    ])
    inv["pack_per_case"] = 10.0

    shelf_usage = _build_shelf_usage(inv)
    # 帯外の空きスロットのみ
    _add_empty_locs(shelf_usage, ["00102001"])
    rep_pack_by_col = {20: 50.0}  # col20はpack=50 → SKUのpack=10は帯外
    cfg = _make_cfg(pack_tolerance_ratio=0.10)

    moves = _call(inv, shelf_usage, cfg=cfg, rep_pack_by_col=rep_pack_by_col)

    # 帯外でも移動が生成されること (soft制約)
    assert len(moves) >= 1


# ============================================================================
# テストケース 5: 同列優先
# ============================================================================
def test_case5_same_column_priority():
    """同列空きと別列空き → 同列が優先"""
    inv = _make_inv([
        {"sku": "E", "lot": "L1", "lot_key": 20250101, "loc": "00300301", "qty": 1, "vol_each": 1.0},  # Lv3 col3 古
        {"sku": "E", "lot": "L2", "lot_key": 20251201, "loc": "00400301", "qty": 1, "vol_each": 1.0},  # Lv4 col3 新
    ])
    shelf_usage = _build_shelf_usage(inv)
    # 同列(col3)のLv1と別列(col10)のLv1
    _add_empty_locs(shelf_usage, ["00100301", "00101001"])

    moves = _call(inv, shelf_usage)

    # 古ロット(L1)が同列(col3)に移動していること
    l1_moves = [m for m in moves if m.lot == "L1"]
    assert len(l1_moves) >= 1
    assert l1_moves[0].to_loc[3:6] == "003"  # col=003


# ============================================================================
# テストケース 6: 自SKU列間スワップ
# ============================================================================
def test_case6_cross_column_swap():
    """空きなし + Lv1-2に自SKU新ロットあり → 列間スワップ"""
    # vol_each=100.0 にして cap_limit(100.0)と同じにし、空き容量を0にする
    inv = _make_inv([
        {"sku": "F", "lot": "L1", "lot_key": 20250101, "loc": "00300501", "qty": 1, "vol_each": 2.0},  # Lv3 col5 古
        {"sku": "F", "lot": "L2", "lot_key": 20251201, "loc": "00100801", "qty": 1, "vol_each": 100.0},  # Lv1 col8 新 (容量満杯)
    ])
    shelf_usage = _build_shelf_usage(inv)
    # Lv1(col8)の usage=100.0 = cap_limit → 空き容量 0 → 空きスロットなし
    # lots_by_loc_sku でロット混在も示す（念のため）
    lots_by_loc_sku = {
        ("00100801", "F"): {20251201},  # L2が居る
    }

    moves = _call(inv, shelf_usage, lots_by_loc_sku=lots_by_loc_sku)

    # スワップが生成されること
    assert len(moves) == 2
    assert moves[0].chain_group_id == moves[1].chain_group_id
    # 古ロット(L1)がLv1-2へ
    l1_move = next(m for m in moves if m.lot == "L1")
    assert l1_move.to_loc.startswith("001") or l1_move.to_loc.startswith("002")
    # 新ロット(L2)がLv3-4へ
    l2_move = next(m for m in moves if m.lot == "L2")
    assert l2_move.to_loc.startswith("003") or l2_move.to_loc.startswith("004")


# ============================================================================
# テストケース 7: 優先度順 - 古SKUが先に空きスロット取得
# ============================================================================
def test_case7_priority_order():
    """2SKU同時、古SKU優先 → 古SKUが先に空きスロット取得"""
    inv = _make_inv([
        # SKU-A: 最古ロット=20240101 (より古い)
        {"sku": "A", "lot": "LA1", "lot_key": 20240101, "loc": "00300101", "qty": 1, "vol_each": 50.0},
        {"sku": "A", "lot": "LA2", "lot_key": 20241201, "loc": "00400101", "qty": 1, "vol_each": 50.0},
        # SKU-B: 最古ロット=20250101 (より新しい)
        {"sku": "B", "lot": "LB1", "lot_key": 20250101, "loc": "00300201", "qty": 1, "vol_each": 50.0},
        {"sku": "B", "lot": "LB2", "lot_key": 20251201, "loc": "00400201", "qty": 1, "vol_each": 50.0},
    ])
    shelf_usage = _build_shelf_usage(inv)
    # Lv1に空きスロット1つだけ (容量50)
    shelf_usage["00100101"] = 0.0  # Lv1 col1 空き

    # budget_left=2 (スワップ2件分) で古SKU(A)の1スワップのみ実行できるようにする
    moves = _call(inv, shelf_usage, budget_left=2)  # スワップ2件=1SKU分

    # SKU-Aが移動していること (古い順優先)
    sku_a_moves = [m for m in moves if m.sku_id == "A"]
    assert len(sku_a_moves) >= 1


# ============================================================================
# テストケース 8: len(sku_df) <= 1 はスキップ
# ============================================================================
def test_case8_skip_single_row_sku():
    """len(sku_df) <= 1 → スキップ"""
    inv = _make_inv([
        {"sku": "G", "lot": "L1", "lot_key": 20250101, "loc": "00300101", "qty": 1, "vol_each": 1.0},
    ])
    shelf_usage = _build_shelf_usage(inv)
    _add_empty_locs(shelf_usage, ["00100101"])

    moves = _call(inv, shelf_usage)
    assert len(moves) == 0


# ============================================================================
# テストケース 9: lot_key.nunique <= 1 はスキップ
# ============================================================================
def test_case9_skip_single_lot():
    """lot_key.nunique <= 1 → スキップ"""
    inv = _make_inv([
        {"sku": "H", "lot": "L1", "lot_key": 20250101, "loc": "00300101", "qty": 1, "vol_each": 1.0},
        {"sku": "H", "lot": "L1", "lot_key": 20250101, "loc": "00300201", "qty": 1, "vol_each": 1.0},
    ])
    shelf_usage = _build_shelf_usage(inv)
    _add_empty_locs(shelf_usage, ["00100101"])

    moves = _call(inv, shelf_usage)
    assert len(moves) == 0


# ============================================================================
# テストケース 10: UNKNOWN_LOT_KEY 行は通過しない
# ============================================================================
def test_case10_unknown_lot_excluded():
    """UNKNOWN_LOT_KEY 行 → 通過しない"""
    from app.services.optimizer import UNKNOWN_LOT_KEY
    inv = _make_inv([
        {"sku": "I", "lot": "JU2025", "lot_key": UNKNOWN_LOT_KEY, "loc": "00300101", "qty": 2, "vol_each": 1.0},
        {"sku": "I", "lot": "L2", "lot_key": 20251201, "loc": "00100101", "qty": 2, "vol_each": 1.0},
    ])
    shelf_usage = _build_shelf_usage(inv)

    moves = _call(inv, shelf_usage)
    assert len(moves) == 0


# ============================================================================
# テストケース 11: is_movable=False は通過しない
# ============================================================================
def test_case11_is_movable_false_excluded():
    """is_movable=False → 通過しない"""
    inv = _make_inv([
        {"sku": "J", "lot": "L1", "lot_key": 20250101, "loc": "00300101", "qty": 1, "vol_each": 1.0, "is_movable": False},
        {"sku": "J", "lot": "L2", "lot_key": 20251201, "loc": "00100101", "qty": 1, "vol_each": 1.0, "is_movable": False},
    ])
    shelf_usage = _build_shelf_usage(inv)

    moves = _call(inv, shelf_usage)
    assert len(moves) == 0


# ============================================================================
# テストケース 12: budget_left 枯渇で途中 break
# ============================================================================
def test_case12_budget_left_exhaustion():
    """budget_left 枯渇 → 途中で break"""
    # SKU K: Lv3(古) / Lv1(新) 同列 → 同列スワップ候補
    # SKU M: Lv3(古) / Lv1(新) 同列 → 同列スワップ候補 (M の方が oldest が古い)
    inv = _make_inv([
        {"sku": "K", "lot": "L1", "lot_key": 20250101, "loc": "00300101", "qty": 1, "vol_each": 1.0},
        {"sku": "K", "lot": "L2", "lot_key": 20251201, "loc": "00100101", "qty": 1, "vol_each": 1.0},
        {"sku": "M", "lot": "M1", "lot_key": 20240101, "loc": "00300201", "qty": 1, "vol_each": 1.0},
        {"sku": "M", "lot": "M2", "lot_key": 20241201, "loc": "00100201", "qty": 1, "vol_each": 1.0},
    ])
    shelf_usage = _build_shelf_usage(inv)

    # budget_left=0 → 移動なし
    moves = _call(inv, shelf_usage, budget_left=0)
    assert len(moves) == 0

    # budget_left=2 (スワップ2件分) で古SKU(A)の1スワップのみ実行できるようにする
    # M の oldest(20240101) < K の oldest(20250101) → M が先に処理され 2 moves 生成
    moves = _call(inv, shelf_usage, budget_left=2)
    assert len(moves) == 2


# ============================================================================
# テストケース 13: ロット混在拒否 - 候補スロットに自SKU異ロット
# ============================================================================
def test_case13_lot_mix_rejected():
    """候補スロットに自SKU異ロットあり → 除外"""
    inv = _make_inv([
        {"sku": "N", "lot": "L1", "lot_key": 20250101, "loc": "00300101", "qty": 1, "vol_each": 2.0},  # 古
        {"sku": "N", "lot": "L2", "lot_key": 20251201, "loc": "00400101", "qty": 1, "vol_each": 2.0},  # 新
    ])
    shelf_usage = _build_shelf_usage(inv)
    # Lv1の空きスロットを追加
    empty_loc = "00100101"
    _add_empty_locs(shelf_usage, [empty_loc])

    # lots_by_loc_sku: その空きスロットに既に別ロット(L3)がある
    lots_by_loc_sku: Dict[Tuple[str, str], Set[int]] = {
        ("00100101", "N"): {20250601},  # 異なるロット(L3)
    }

    moves = _call(inv, shelf_usage, lots_by_loc_sku=lots_by_loc_sku)

    # L1(古ロット)は空きスロットに入れないのでスキップまたはスワップになる
    # スロット混在で拒否された場合、空き移動ではなくスワップ(0件か2件)
    l1_to_lv1_moves = [m for m in moves if m.lot == "L1" and m.to_loc.startswith("001") and m.chain_group_id is None]
    assert len(l1_to_lv1_moves) == 0


# ============================================================================
# テストケース 14: 他SKU混在拒否
# ============================================================================
def test_case14_foreign_sku_rejected():
    """候補スロットに他SKUあり → 除外"""
    inv = _make_inv([
        {"sku": "P", "lot": "L1", "lot_key": 20250101, "loc": "00300101", "qty": 1, "vol_each": 2.0},
        {"sku": "P", "lot": "L2", "lot_key": 20251201, "loc": "00400101", "qty": 1, "vol_each": 2.0},
    ])
    shelf_usage = _build_shelf_usage(inv)
    empty_loc = "00100101"
    _add_empty_locs(shelf_usage, [empty_loc])

    # skus_by_loc: その空きスロットに他SKU(Q)がいる
    skus_by_loc: Dict[str, Set[str]] = {
        "00100101": {"Q"},
    }

    moves = _call(inv, shelf_usage, skus_by_loc=skus_by_loc)

    # 他SKU混在で拒否されるため、空き移動(chain_group_id=None)はない
    solo_moves = [m for m in moves if m.to_loc == "00100101" and m.chain_group_id is None]
    assert len(solo_moves) == 0


# ============================================================================
# テストケース 15: 同列スワップ - from_loc/to_loc に他SKU混在 → 生成されない
# ============================================================================
def test_case15_same_column_swap_rejected_by_foreign_sku():
    """同列スワップ: to_loc or from_loc に他SKUが居たら生成しない"""
    # SKU_A: old at Lv3-col5-dep1 (00300501), new at Lv1-col5-dep1 (00100501) 同列
    # 他SKU_B が old_loc (00300501) に同居
    inv = _make_inv([
        {"sku": "A", "lot": "L1", "lot_key": 20240101,
         "loc": "00300501", "qty": 5, "vol_each": 1.0},
        {"sku": "A", "lot": "L2", "lot_key": 20250101,
         "loc": "00100501", "qty": 5, "vol_each": 1.0},
        {"sku": "B", "lot": "LB", "lot_key": 20250101,
         "loc": "00300501", "qty": 3, "vol_each": 1.0},
    ])
    shelf_usage = _build_shelf_usage(inv)
    skus_by_loc: Dict[str, Set[str]] = {
        "00300501": {"A", "B"},  # 他SKU混在
        "00100501": {"A"},
    }
    lots_by_loc_sku: Dict[Tuple[str, str], Set[int]] = {
        ("00300501", "A"): {20240101},
        ("00300501", "B"): {20250101},
        ("00100501", "A"): {20250101},
    }

    moves = _call(inv, shelf_usage, skus_by_loc=skus_by_loc, lots_by_loc_sku=lots_by_loc_sku)

    # 同列スワップは他SKUが居るので生成されない
    swap_moves = [m for m in moves if m.chain_group_id and m.chain_group_id.startswith("swap_fifo_")]
    assert len(swap_moves) == 0


# ============================================================================
# テストケース 16: 列間スワップ - target_row の from_loc に他SKU混在 → スキップ
# ============================================================================
def test_case16_cross_column_swap_rejected_by_foreign_sku():
    """列間スワップ: target_row の from_loc に他SKUが居たらスキップ"""
    # SKU_A: old at Lv4-col10-dep1 (00401001), new at Lv1-col20-dep1 (00102001)
    # 他SKU_B が new_loc (00102001) に同居 → 列間スワップ不可
    # Lv1-2 に他の空きスロットもなし → 移動 0件
    inv = _make_inv([
        {"sku": "A", "lot": "L1", "lot_key": 20240101,
         "loc": "00401001", "qty": 5, "vol_each": 1.0},
        {"sku": "A", "lot": "L2", "lot_key": 20250101,
         "loc": "00102001", "qty": 5, "vol_each": 1.0},
        {"sku": "B", "lot": "LB", "lot_key": 20250101,
         "loc": "00102001", "qty": 3, "vol_each": 1.0},
    ])
    shelf_usage = _build_shelf_usage(inv)
    # Lv1-2 に空きスロットなし (他の空きを追加しない)
    skus_by_loc: Dict[str, Set[str]] = {
        "00401001": {"A"},
        "00102001": {"A", "B"},  # 他SKU混在
    }
    lots_by_loc_sku: Dict[Tuple[str, str], Set[int]] = {
        ("00401001", "A"): {20240101},
        ("00102001", "A"): {20250101},
        ("00102001", "B"): {20250101},
    }

    moves = _call(inv, shelf_usage, skus_by_loc=skus_by_loc, lots_by_loc_sku=lots_by_loc_sku)

    # 列間スワップは他SKUが居るのでスキップ、空きスロットもないので 0 moves
    assert len(moves) == 0


# ============================================================================
# テストケース 17: クロスSKUスワップ成功
# ============================================================================
def test_case17_cross_sku_swap_success():
    """クロスSKUスワップ成功: X(Lv3) ↔ Y(Lv1) で両方クリーン"""
    inv = _make_inv([
        # X: oldest at Lv3-c5-dep1 (clean)
        {"sku": "X", "lot": "20240101", "lot_key": 20240101, "loc": "00300501",
         "qty": 5, "vol_each": 1.0},
        {"sku": "X", "lot": "20250101", "lot_key": 20250101, "loc": "00300502",
         "qty": 5, "vol_each": 1.0},
        # Y: Lv1-c10 clean (Y's newer lot); Y has another Lv1 elsewhere for FIFO safety
        {"sku": "Y", "lot": "20260101", "lot_key": 20260101, "loc": "00101001",
         "qty": 3, "vol_each": 1.0},  # Y の非 OLDEST @ Lv1 ← スワップ対象
        {"sku": "Y", "lot": "20250101", "lot_key": 20250101, "loc": "00101501",
         "qty": 3, "vol_each": 1.0},  # Y の OLDEST @ Lv1 (FIFO safe)
    ])
    skus_by_loc = {
        "00300501": {"X"},
        "00300502": {"X"},
        "00101001": {"Y"},
        "00101501": {"Y"},
    }
    lots_by_loc_sku = {
        ("00300501", "X"): {20240101},
        ("00300502", "X"): {20250101},
        ("00101001", "Y"): {20260101},
        ("00101501", "Y"): {20250101},
    }
    shelf_usage = {"00300501": 5.0, "00300502": 5.0, "00101001": 3.0, "00101501": 3.0}

    moves = _call(inv, shelf_usage, skus_by_loc=skus_by_loc, lots_by_loc_sku=lots_by_loc_sku)

    # クロスSKU スワップが発生: X Lv3-c5-dep1 ↔ Y Lv1-c10-dep1
    cross_swaps = [m for m in moves if (m.chain_group_id or "").startswith("swap_cross_sku_")]
    assert len(cross_swaps) == 2  # 1 ペア = 2 手

    # X が Lv1-c10 へ
    x_move = next(m for m in cross_swaps if m.sku_id == "X")
    assert x_move.from_loc == "00300501"
    assert x_move.to_loc == "00101001"

    # Y が Lv3-c5 へ
    y_move = next(m for m in cross_swaps if m.sku_id == "Y")
    assert y_move.from_loc == "00101001"
    assert y_move.to_loc == "00300501"


# ============================================================================
# テストケース 18: クロスSKUスワップ拒否 - Y の当該行が Y の OLDEST
# ============================================================================
def test_case18_cross_sku_swap_rejected_y_is_oldest():
    """クロスSKUスワップ拒否: Y の Lv1-2 行が唯一かつ OLDEST"""
    inv = _make_inv([
        {"sku": "X", "lot": "20240101", "lot_key": 20240101, "loc": "00300501",
         "qty": 5, "vol_each": 1.0},
        {"sku": "X", "lot": "20250101", "lot_key": 20250101, "loc": "00300502",
         "qty": 5, "vol_each": 1.0},
        # Y の Lv1 行が唯一であり、かつ Y の OLDEST
        {"sku": "Y", "lot": "20250601", "lot_key": 20250601, "loc": "00101001",
         "qty": 3, "vol_each": 1.0},  # Y の OLDEST = Lv1 の唯一行 ← スワップ不可
        {"sku": "Y", "lot": "20260101", "lot_key": 20260101, "loc": "00301501",
         "qty": 3, "vol_each": 1.0},  # Y の newer @ Lv3
    ])
    skus_by_loc = {
        "00300501": {"X"},
        "00300502": {"X"},
        "00101001": {"Y"},
        "00301501": {"Y"},
    }
    lots_by_loc_sku = {
        ("00300501", "X"): {20240101},
        ("00300502", "X"): {20250101},
        ("00101001", "Y"): {20250601},
        ("00301501", "Y"): {20260101},
    }
    shelf_usage = {"00300501": 5.0, "00300502": 5.0, "00101001": 3.0, "00301501": 3.0}

    moves = _call(inv, shelf_usage, skus_by_loc=skus_by_loc, lots_by_loc_sku=lots_by_loc_sku)

    # Y の Lv1-c10 は Y の OLDEST: y_lot_key == y_min_lot なのでスワップ不可
    # かつ Lv1-2 が1行のみなので他 Lv1-2 がなく Policy 1 でも拒否
    cross_swaps = [m for m in moves if (m.chain_group_id or "").startswith("swap_cross_sku_")]
    assert len(cross_swaps) == 0


# ============================================================================
# テストケース 19: クロスSKUスワップ拒否 - Y が当該行以外に Lv1-2 を持っていない
# ============================================================================
def test_case19_cross_sku_swap_rejected_y_unique_lv12():
    """クロスSKUスワップ拒否: Y が当該行以外に Lv1-2 を持っていない"""
    inv = _make_inv([
        {"sku": "X", "lot": "20240101", "lot_key": 20240101, "loc": "00300501",
         "qty": 5, "vol_each": 1.0},
        {"sku": "X", "lot": "20250101", "lot_key": 20250101, "loc": "00300502",
         "qty": 5, "vol_each": 1.0},
        # Y の Lv1 行が唯一の Lv1-2 (OLDEST ではないが unique)
        {"sku": "Y", "lot": "20260101", "lot_key": 20260101, "loc": "00101001",
         "qty": 3, "vol_each": 1.0},  # Y の非 OLDEST but only Lv1
        {"sku": "Y", "lot": "20250101", "lot_key": 20250101, "loc": "00301501",
         "qty": 3, "vol_each": 1.0},  # Y の OLDEST @ Lv3 (FIFO 違反状態)
    ])
    skus_by_loc = {
        "00300501": {"X"},
        "00300502": {"X"},
        "00101001": {"Y"},
        "00301501": {"Y"},
    }
    lots_by_loc_sku = {
        ("00300501", "X"): {20240101},
        ("00300502", "X"): {20250101},
        ("00101001", "Y"): {20260101},
        ("00301501", "Y"): {20250101},
    }
    shelf_usage = {"00300501": 5.0, "00300502": 5.0, "00101001": 3.0, "00301501": 3.0}

    moves = _call(inv, shelf_usage, skus_by_loc=skus_by_loc, lots_by_loc_sku=lots_by_loc_sku)

    # Y の Lv1-2 は c10 の 1 つだけ → スワップしたら Y が Lv1-2 完全喪失 → Policy 1 拒否
    cross_swaps = [m for m in moves if (m.chain_group_id or "").startswith("swap_cross_sku_")]
    assert len(cross_swaps) == 0
