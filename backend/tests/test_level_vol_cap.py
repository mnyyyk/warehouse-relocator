"""Tests for enforce_level_vol_cap: per-level volume cap applied to all destinations."""
import pytest
import pandas as pd
from typing import Dict, Set, Optional


def _make_cfg(**kwargs):
    from app.services.optimizer import OptimizerConfig
    cfg = OptimizerConfig()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def _make_sku_master(rows):
    records = []
    for r in rows:
        records.append({
            "商品ID": str(r["sku_id"]),
            "入数": float(r.get("pack_qty", 30)),
            "carton_volume_m3": float(r.get("vol_m3", 0.05)),
        })
    return pd.DataFrame(records)


def _make_inv(rows):
    records = []
    for r in rows:
        loc = str(r["loc"]).zfill(8)
        lv, col, dep = int(loc[0:3]), int(loc[3:6]), int(loc[6:8])
        records.append({
            "商品ID": str(r["sku"]),
            "ロット": str(r.get("lot", "L1")),
            "lot_key": int(r.get("lot_key", 20250101)),
            "ロケーション": loc,
            "lv": lv,
            "col": col,
            "dep": dep,
            "qty_cases_move": int(r.get("qty", 1)),
            "volume_each_case": float(r.get("vol_each", 0.05)),
            "is_movable": bool(r.get("is_movable", True)),
        })
    return pd.DataFrame(records)


def _make_move(sku, lot, qty, from_loc, to_loc, chain_group_id=None):
    from app.services.optimizer import Move
    return Move(
        sku_id=str(sku),
        lot=str(lot),
        qty=int(qty),
        from_loc=str(from_loc).zfill(8),
        to_loc=str(to_loc).zfill(8),
        lot_date=None,
        reason="test",
        chain_group_id=chain_group_id,
        execution_order=None,
    )


def _call_enforce(sku_master, inv, moves, cfg=None, original_empty_locs=None,
                  original_skus_by_loc=None, original_qty_by_loc_sku=None):
    from app.services.optimizer import enforce_constraints
    return enforce_constraints(
        sku_master=sku_master,
        inventory=inv,
        moves=moves,
        cfg=cfg or _make_cfg(),
        original_skus_by_loc=original_skus_by_loc,
        original_qty_by_loc_sku=original_qty_by_loc_sku,
        original_empty_locs=original_empty_locs,
    )


# ロケーション規則: 001xxxxx = Lv1, 002xxxxx = Lv2, 003xxxxx = Lv3, 004xxxxx = Lv4
FROM_LOC = "00500101"   # Lv5 = 移動元（上限なし）
LV1_LOC = "00100101"
LV2_LOC = "00200101"
LV3_LOC = "00300101"
LV4_LOC = "00400101"


def _base_sku_master(sku_id="SKU_A", vol_m3=0.1):
    return _make_sku_master([{"sku_id": sku_id, "vol_m3": vol_m3}])


def _base_inv(sku_id="SKU_A", qty=10, vol_each=0.1, loc=FROM_LOC):
    return _make_inv([{"sku": sku_id, "loc": loc, "qty": qty, "vol_each": vol_each}])


class TestLevelVolCapSingleSku:
    """単独SKU配置での段別容積上限テスト。"""

    # Case 1: Lv1 単独SKU で 0.9㎥ 超 → reject
    def test_lv1_single_sku_over_cap_rejected(self):
        # 1.0㎥ を Lv1 に投入 → 超過で却下
        sku_master = _base_sku_master(vol_m3=1.0)
        inv = _base_inv(qty=1, vol_each=1.0)
        moves = [_make_move("SKU_A", "L1", 1, FROM_LOC, LV1_LOC)]
        cfg = _make_cfg(enforce_level_vol_cap=True, multi_sku_level1_vol_cap=0.9)
        result = _call_enforce(sku_master, inv, moves, cfg=cfg)
        assert len(result) == 0, f"Expected reject, got {len(result)} moves"

    # Case 2: Lv2 単独SKU で 0.3㎥ 超 → reject
    def test_lv2_single_sku_over_cap_rejected(self):
        sku_master = _base_sku_master(vol_m3=0.5)
        inv = _base_inv(qty=1, vol_each=0.5)
        moves = [_make_move("SKU_A", "L1", 1, FROM_LOC, LV2_LOC)]
        cfg = _make_cfg(enforce_level_vol_cap=True, multi_sku_level2_vol_cap=0.3)
        result = _call_enforce(sku_master, inv, moves, cfg=cfg)
        assert len(result) == 0, f"Expected reject, got {len(result)} moves"

    # Case 3: Lv1 単独SKU で 0.9㎥ ちょうど → accept
    def test_lv1_single_sku_at_cap_accepted(self):
        sku_master = _base_sku_master(vol_m3=0.9)
        inv = _base_inv(qty=1, vol_each=0.9)
        moves = [_make_move("SKU_A", "L1", 1, FROM_LOC, LV1_LOC)]
        cfg = _make_cfg(enforce_level_vol_cap=True, multi_sku_level1_vol_cap=0.9)
        result = _call_enforce(sku_master, inv, moves, cfg=cfg)
        assert len(result) == 1, f"Expected accept, got {len(result)} moves"

    # Case 4: Lv2 単独SKU で 0.3㎥ ちょうど → accept
    def test_lv2_single_sku_at_cap_accepted(self):
        sku_master = _base_sku_master(vol_m3=0.3)
        inv = _base_inv(qty=1, vol_each=0.3)
        moves = [_make_move("SKU_A", "L1", 1, FROM_LOC, LV2_LOC)]
        cfg = _make_cfg(enforce_level_vol_cap=True, multi_sku_level2_vol_cap=0.3)
        result = _call_enforce(sku_master, inv, moves, cfg=cfg)
        assert len(result) == 1, f"Expected accept, got {len(result)} moves"

    # Case 5: Lv3 に単独SKU で 1.0㎥ 投入 → accept（level cap 対象外）
    def test_lv3_no_cap(self):
        sku_master = _base_sku_master(vol_m3=1.0)
        inv = _base_inv(qty=1, vol_each=1.0)
        moves = [_make_move("SKU_A", "L1", 1, FROM_LOC, LV3_LOC)]
        cfg = _make_cfg(enforce_level_vol_cap=True, multi_sku_level1_vol_cap=0.9, multi_sku_level2_vol_cap=0.3)
        result = _call_enforce(sku_master, inv, moves, cfg=cfg)
        assert len(result) == 1, f"Expected accept on Lv3, got {len(result)} moves"

    # Case 7: enforce_level_vol_cap=False で無効化
    def test_disabled_allows_over_cap(self):
        sku_master = _base_sku_master(vol_m3=0.5)
        inv = _base_inv(qty=1, vol_each=0.5)
        moves = [_make_move("SKU_A", "L1", 1, FROM_LOC, LV2_LOC)]
        cfg = _make_cfg(enforce_level_vol_cap=False, multi_sku_level2_vol_cap=0.3)
        result = _call_enforce(sku_master, inv, moves, cfg=cfg)
        assert len(result) == 1, f"Expected accept when disabled, got {len(result)} moves"


class TestLevelVolCapSameSkuAddition:
    """同SKU追加投入での段別容積上限テスト。"""

    # Case 6: 既に 0.2㎥ の Lv2 ロケに同SKUで 0.2㎥ 追加 → 合計0.4㎥ で reject
    def test_lv2_same_sku_additional_over_cap_rejected(self):
        # Lv2 のロケに既に 0.2㎥ 分の在庫がある状態で 0.2㎥ 追加 → 0.4㎥ > 0.3㎥ で却下
        sku_master = _make_sku_master([{"sku_id": "SKU_A", "vol_m3": 0.2}])
        # FROM_LOC から来る在庫
        inv_rows = [
            {"sku": "SKU_A", "loc": FROM_LOC, "qty": 1, "vol_each": 0.2},
            {"sku": "SKU_A", "loc": LV2_LOC, "qty": 1, "vol_each": 0.2},  # 既存在庫
        ]
        inv = _make_inv(inv_rows)
        moves = [_make_move("SKU_A", "L1", 1, FROM_LOC, LV2_LOC)]
        cfg = _make_cfg(enforce_level_vol_cap=True, multi_sku_level2_vol_cap=0.3)
        # original_qty_by_loc_sku: LV2_LOC に 0.2㎥ 相当の既存在庫
        original_qty = {(LV2_LOC, "SKU_A"): 1.0}
        result = _call_enforce(
            sku_master, inv, moves, cfg=cfg,
            original_qty_by_loc_sku=original_qty,
        )
        assert len(result) == 0, f"Expected reject on same-SKU addition over cap, got {len(result)} moves"

    # Case 6b: 既に 0.1㎥ の Lv2 ロケに同SKUで 0.1㎥ 追加 → 合計0.2㎥ ≤ 0.3㎥ → accept
    def test_lv2_same_sku_additional_within_cap_accepted(self):
        # SKU_A (0.1㎥) が LV2_LOC にいる。同SKUをもう1ケース追加 → 0.2㎥ ≤ 0.3㎥ で受入
        sku_master = _make_sku_master([{"sku_id": "SKU_A", "vol_m3": 0.1}])
        inv_rows = [
            {"sku": "SKU_A", "loc": FROM_LOC, "qty": 1, "vol_each": 0.1},
            {"sku": "SKU_A", "loc": LV2_LOC, "qty": 1, "vol_each": 0.1},
        ]
        inv = _make_inv(inv_rows)
        moves = [_make_move("SKU_A", "L1", 1, FROM_LOC, LV2_LOC)]
        cfg = _make_cfg(enforce_level_vol_cap=True, multi_sku_level2_vol_cap=0.3)
        original_qty = {(LV2_LOC, "SKU_A"): 1.0}
        result = _call_enforce(
            sku_master, inv, moves, cfg=cfg,
            original_qty_by_loc_sku=original_qty,
        )
        assert len(result) == 1, f"Expected accept within cap, got {len(result)} moves"


class TestLevelVolCapMoveOut:
    """元々超過しているロケからの move-out / move-in テスト。"""

    # Case 8: 既に 0.4㎥ の Lv2 ロケから move-out → accept（from 側制限なし）
    def test_lv2_overloaded_move_out_accepted(self):
        # 0.4㎥ の在庫を Lv2 から別ロケへ移動 → from 側は制限なし
        sku_master = _make_sku_master([{"sku_id": "SKU_A", "vol_m3": 0.4}])
        inv_rows = [
            {"sku": "SKU_A", "loc": LV2_LOC, "qty": 1, "vol_each": 0.4},
        ]
        inv = _make_inv(inv_rows)
        moves = [_make_move("SKU_A", "L1", 1, LV2_LOC, LV3_LOC)]
        cfg = _make_cfg(enforce_level_vol_cap=True, multi_sku_level2_vol_cap=0.3)
        result = _call_enforce(sku_master, inv, moves, cfg=cfg)
        assert len(result) == 1, f"Expected accept for move-out from overloaded loc, got {len(result)} moves"

    # Case 9: 既に 0.4㎥ の Lv2 ロケへの追加 move-in → reject
    def test_lv2_overloaded_move_in_rejected(self):
        # 0.4㎥ 超過中の Lv2 ロケにさらに追加 → reject
        sku_master = _make_sku_master([{"sku_id": "SKU_A", "vol_m3": 0.1}])
        inv_rows = [
            {"sku": "SKU_A", "loc": FROM_LOC, "qty": 1, "vol_each": 0.1},
            {"sku": "SKU_A", "loc": LV2_LOC, "qty": 4, "vol_each": 0.1},  # 既存0.4㎥
        ]
        inv = _make_inv(inv_rows)
        moves = [_make_move("SKU_A", "L1", 1, FROM_LOC, LV2_LOC)]
        cfg = _make_cfg(enforce_level_vol_cap=True, multi_sku_level2_vol_cap=0.3)
        original_qty = {(LV2_LOC, "SKU_A"): 4.0}
        result = _call_enforce(
            sku_master, inv, moves, cfg=cfg,
            original_qty_by_loc_sku=original_qty,
        )
        assert len(result) == 0, f"Expected reject for move-in to overloaded loc, got {len(result)} moves"


class TestLevelVolCapLv4:
    """Lv4 の場合は段別容積上限なし（Lv3 と同じ）。"""

    def test_lv4_no_cap(self):
        sku_master = _base_sku_master(vol_m3=1.5)
        inv = _base_inv(qty=1, vol_each=1.5)
        moves = [_make_move("SKU_A", "L1", 1, FROM_LOC, LV4_LOC)]
        cfg = _make_cfg(enforce_level_vol_cap=True, multi_sku_level1_vol_cap=0.9, multi_sku_level2_vol_cap=0.3)
        result = _call_enforce(sku_master, inv, moves, cfg=cfg)
        assert len(result) == 1, f"Expected accept on Lv4, got {len(result)} moves"
