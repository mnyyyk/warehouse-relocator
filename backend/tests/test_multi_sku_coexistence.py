"""Unit tests for multi-SKU coexistence in originally empty locations."""
import pytest
import pandas as pd
from typing import Dict, Set, Tuple, Optional


def _make_cfg(**kwargs):
    from app.services.optimizer import OptimizerConfig
    cfg = OptimizerConfig()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def _make_sku_master(rows):
    """SKUマスタ DataFrame を構築するヘルパー。

    各 row は dict で: sku_id, pack_qty, vol_m3
    """
    records = []
    for r in rows:
        records.append({
            "商品ID": str(r["sku_id"]),
            "入数": float(r.get("pack_qty", 30)),
            "carton_volume_m3": float(r.get("vol_m3", 0.05)),
        })
    return pd.DataFrame(records)


def _make_inv(rows):
    """在庫 DataFrame を構築するヘルパー。

    各 row は dict で: sku, lot, lot_key, loc, lv, col, dep, qty, vol_each
    """
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


def _make_move(sku, lot, qty, from_loc, to_loc, chain_group_id=None, execution_order=None):
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
        execution_order=execution_order,
    )


def _call_enforce(sku_master, inv, moves, cfg=None, original_empty_locs=None,
                  original_skus_by_loc=None, original_qty_by_loc_sku=None,
                  blocked_source_locs=None, blocked_dest_locs=None):
    from app.services.optimizer import enforce_constraints
    return enforce_constraints(
        sku_master=sku_master,
        inventory=inv,
        moves=moves,
        cfg=cfg or _make_cfg(),
        original_skus_by_loc=original_skus_by_loc,
        original_qty_by_loc_sku=original_qty_by_loc_sku,
        original_empty_locs=original_empty_locs,
        blocked_source_locs=blocked_source_locs,
        blocked_dest_locs=blocked_dest_locs,
    )


def _call_check(to_loc, sku_id, qty, pack_vol, original_receivable_locs, sim_skus_by_loc,
                sim_vol_by_loc, sku_pack_band, level, cfg=None,
                # 後方互換: 旧引数名でも動作するようにエイリアス
                original_empty_locs=None):
    from app.services.optimizer import _check_multi_sku_coexistence
    # original_empty_locs が渡された場合は receivable_locs として使用（旧テスト互換）
    recv = original_receivable_locs if original_receivable_locs is not None else original_empty_locs
    return _check_multi_sku_coexistence(
        to_loc=to_loc,
        sku_id=sku_id,
        qty=qty,
        pack_vol=pack_vol,
        original_receivable_locs=recv,
        sim_skus_by_loc=sim_skus_by_loc,
        sim_vol_by_loc=sim_vol_by_loc,
        sku_pack_band=sku_pack_band,
        level=level,
        cfg=cfg or _make_cfg(),
    )


# ============================================================================
# ゲート関数 _check_empty_loc_coexistence のユニットテスト
# ============================================================================

class TestCheckMultiSkuCoexistence:
    """_check_empty_loc_coexistence ゲート関数の単体テスト。"""

    def _empty_sets(self, loc="00100100"):
        return (
            {loc},               # original_empty_locs
            {},                  # sim_skus_by_loc
            {},                  # sim_vol_by_loc
        )

    # ------------------------------------------------------------------
    # テスト1: 元々空きロケに3SKU（同pack帯・容積上限内）を段階的に配置 → 全て受理
    # ------------------------------------------------------------------
    def test_case1_three_skus_same_band_within_cap(self):
        """元々空きロケに同pack帯3SKUを順番に追加 → 全て許可"""
        cfg = _make_cfg(
            multi_sku_max_per_loc=3,
            multi_sku_level1_vol_cap=0.9,
            multi_sku_pack_band_match=True,
            multi_sku_target_levels=(1, 2),
        )
        loc = "00100100"
        empty = {loc}
        sim_skus: Dict[str, Set[str]] = {}
        sim_vol: Dict[str, float] = {}
        band = {"SKU_A": "medium", "SKU_B": "medium", "SKU_C": "medium"}

        # 1SKU目（他SKUなし → 実質1SKU目なので許可）
        allowed, reason = _call_check(loc, "SKU_A", 1, 0.1, empty, sim_skus, sim_vol, band, 1, cfg)
        assert allowed is True, f"1SKU目は許可されるべき: {reason}"
        sim_skus.setdefault(loc, set()).add("SKU_A")
        sim_vol[loc] = sim_vol.get(loc, 0.0) + 0.1

        # 2SKU目
        allowed, reason = _call_check(loc, "SKU_B", 1, 0.1, empty, sim_skus, sim_vol, band, 1, cfg)
        assert allowed is True, f"2SKU目は許可されるべき: {reason}"
        sim_skus[loc].add("SKU_B")
        sim_vol[loc] += 0.1

        # 3SKU目
        allowed, reason = _call_check(loc, "SKU_C", 1, 0.1, empty, sim_skus, sim_vol, band, 1, cfg)
        assert allowed is True, f"3SKU目は許可されるべき: {reason}"

    # ------------------------------------------------------------------
    # テスト2: 4SKU目は却下
    # ------------------------------------------------------------------
    def test_case2_fourth_sku_rejected(self):
        """3SKU後に4SKU目を追加しようとすると却下"""
        cfg = _make_cfg(multi_sku_max_per_loc=3, multi_sku_target_levels=(1, 2))
        loc = "00100100"
        empty = {loc}
        sim_skus = {loc: {"SKU_A", "SKU_B", "SKU_C"}}
        sim_vol = {loc: 0.3}
        band = {"SKU_A": "medium", "SKU_B": "medium", "SKU_C": "medium", "SKU_D": "medium"}

        allowed, reason = _call_check(loc, "SKU_D", 1, 0.05, empty, sim_skus, sim_vol, band, 1, cfg)
        assert allowed is False
        assert reason == "multi_sku_count_exceeded"

    # ------------------------------------------------------------------
    # テスト3: 容積上限超過で却下
    # ------------------------------------------------------------------
    def test_case3_vol_cap_exceeded_lv1(self):
        """段別容積上限は capacity ゲート側で処理（全着地に常時適用）。
        _check_empty_loc_coexistence は SKU数/pack帯チェックのみ行い、容積チェックは行わない。"""
        cfg = _make_cfg(multi_sku_level1_vol_cap=0.9, multi_sku_target_levels=(1, 2))
        loc = "00100100"
        empty = {loc}
        sim_skus = {loc: {"SKU_A", "SKU_B"}}
        sim_vol = {loc: 0.85}
        band = {"SKU_A": "medium", "SKU_B": "medium", "SKU_C": "medium"}

        # この関数では容積チェックをしない（capacity ゲートに委譲）→ 許可
        allowed, reason = _call_check(loc, "SKU_C", 1, 0.1, empty, sim_skus, sim_vol, band, 1, cfg)
        assert allowed is True, f"容積チェックは capacity ゲートに委譲されたため許可されるべき: {reason}"

    def test_case3_vol_cap_boundary_lv1(self):
        """Lv1の容積上限境界値（ちょうど0.9㎥）→ 許可"""
        cfg = _make_cfg(multi_sku_level1_vol_cap=0.9, multi_sku_target_levels=(1, 2))
        loc = "00100100"
        empty = {loc}
        sim_skus = {loc: {"SKU_A", "SKU_B"}}
        sim_vol = {loc: 0.8}  # 残り0.1
        band = {"SKU_A": "medium", "SKU_B": "medium", "SKU_C": "medium"}

        # qty=1, pack_vol=0.1 → 追加後0.9 = 0.9 → 許可（ぴったり上限）
        allowed, reason = _call_check(loc, "SKU_C", 1, 0.1, empty, sim_skus, sim_vol, band, 1, cfg)
        assert allowed is True, f"境界値はちょうど通るべき: {reason}"

    def test_case3_vol_cap_exceeded_lv2(self):
        """段別容積上限は capacity ゲート側で処理（全着地に常時適用）。
        _check_empty_loc_coexistence は SKU数/pack帯チェックのみ行い、容積チェックは行わない。"""
        cfg = _make_cfg(multi_sku_level2_vol_cap=0.3, multi_sku_target_levels=(1, 2))
        loc = "00200100"  # Lv2
        empty = {loc}
        sim_skus = {loc: {"SKU_A", "SKU_B"}}
        sim_vol = {loc: 0.25}
        band = {"SKU_A": "medium", "SKU_B": "medium", "SKU_C": "medium"}

        # この関数では容積チェックをしない（capacity ゲートに委譲）→ 許可
        allowed, reason = _call_check(loc, "SKU_C", 1, 0.1, empty, sim_skus, sim_vol, band, 2, cfg)
        assert allowed is True, f"容積チェックは capacity ゲートに委譲されたため許可されるべき: {reason}"

    # ------------------------------------------------------------------
    # テスト4: 異pack帯SKUの混在で却下
    # ------------------------------------------------------------------
    def test_case4_different_pack_band_rejected(self):
        """異なるpack帯のSKUは混在不可"""
        cfg = _make_cfg(multi_sku_pack_band_match=True, multi_sku_target_levels=(1, 2))
        loc = "00100100"
        empty = {loc}
        sim_skus = {loc: {"SKU_SMALL"}}
        sim_vol = {loc: 0.1}
        band = {"SKU_SMALL": "small", "SKU_LARGE": "large"}

        allowed, reason = _call_check(loc, "SKU_LARGE", 1, 0.05, empty, sim_skus, sim_vol, band, 1, cfg)
        assert allowed is False
        assert reason == "multi_sku_pack_band_mismatch"

    # ------------------------------------------------------------------
    # テスト5: Lv3/Lv4 では複数SKU不可
    # ------------------------------------------------------------------
    def test_case5_lv3_disallowed(self):
        """Lv3では複数SKU同居不可"""
        cfg = _make_cfg(multi_sku_target_levels=(1, 2))
        loc = "00300100"  # Lv3
        empty = {loc}
        sim_skus = {loc: {"SKU_A"}}
        sim_vol = {loc: 0.1}
        band = {"SKU_A": "medium", "SKU_B": "medium"}

        allowed, reason = _call_check(loc, "SKU_B", 1, 0.05, empty, sim_skus, sim_vol, band, 3, cfg)
        assert allowed is False
        assert reason == "multi_sku_level_disallowed"

    def test_case5_lv4_disallowed(self):
        """Lv4では複数SKU同居不可"""
        cfg = _make_cfg(multi_sku_target_levels=(1, 2))
        loc = "00400100"  # Lv4
        empty = {loc}
        sim_skus = {loc: {"SKU_A"}}
        sim_vol = {loc: 0.1}
        band = {"SKU_A": "medium", "SKU_B": "medium"}

        allowed, reason = _call_check(loc, "SKU_B", 1, 0.05, empty, sim_skus, sim_vol, band, 4, cfg)
        assert allowed is False
        assert reason == "multi_sku_level_disallowed"

    # ------------------------------------------------------------------
    # テスト: receivable_locs に含まれないロケはフォールスルー
    # ------------------------------------------------------------------
    def test_non_receivable_loc_fallthrough(self):
        """receivable_locs に含まれないロケ（3SKU以上 = strict blocked）→ (False, None) でフォールスルー"""
        cfg = _make_cfg()
        loc = "00100100"
        receivable = {"00100200"}  # 別のロケだけが受入可能
        sim_skus = {loc: {"SKU_A"}}
        sim_vol = {loc: 0.1}
        band = {"SKU_A": "medium", "SKU_B": "medium"}

        allowed, reason = _call_check(loc, "SKU_B", 1, 0.05, receivable, sim_skus, sim_vol, band, 1, cfg)
        assert allowed is False
        assert reason is None  # フォールスルー


# ============================================================================
# enforce_constraints 統合テスト
# ============================================================================

class TestEnforceConstraintsMultiSku:
    """enforce_constraints を通じた複数SKU同居の統合テスト。"""

    def _base_sku_master(self):
        # medium band: pack=30 (12 < 30 < 50)
        return _make_sku_master([
            {"sku_id": "SKU_A", "pack_qty": 30, "vol_m3": 0.05},
            {"sku_id": "SKU_B", "pack_qty": 30, "vol_m3": 0.05},
            {"sku_id": "SKU_C", "pack_qty": 30, "vol_m3": 0.05},
            {"sku_id": "SKU_D", "pack_qty": 30, "vol_m3": 0.05},
            {"sku_id": "SKU_SMALL", "pack_qty": 10, "vol_m3": 0.05},  # small band
        ])

    def _base_inv(self):
        # SKU_A Lv3→Lv1移動元, Lv1に空き追加, SKU_B も Lv3 にある
        return _make_inv([
            {"sku": "SKU_A", "lot": "LA1", "lot_key": 20250101, "loc": "00301001", "qty": 2, "vol_each": 0.05},
            {"sku": "SKU_B", "lot": "LB1", "lot_key": 20250101, "loc": "00301002", "qty": 2, "vol_each": 0.05},
        ])

    # ------------------------------------------------------------------
    # テスト6: 元々3SKU以上のロケは strict blocked（移動元・移動先どちらも不可）
    # ------------------------------------------------------------------
    def test_case6_original_strict_multi_sku_loc_blocked(self):
        """元々3SKU以上の混在ロケへの着地は blocked_dest_locs で拒否される"""
        sku_master = self._base_sku_master()
        inv = self._base_inv()
        strict_loc = "00100200"

        cfg = _make_cfg(allow_empty_loc_multi_sku=True)
        moves = [
            _make_move("SKU_A", "LA1", 2, "00301001", strict_loc, chain_group_id="p2consol_001"),
        ]

        # original_empty_locs には strict_loc を含めない（元々3SKU以上）
        original_empty_locs: Set[str] = set()
        original_skus_by_loc = {
            "00301001": {"SKU_A"},
            "00301002": {"SKU_B"},
            strict_loc: {"SKU_X", "SKU_Y", "SKU_Z"},  # 元々3SKU混在 → strict blocked
        }

        result = _call_enforce(
            sku_master, inv, moves, cfg=cfg,
            original_empty_locs=original_empty_locs,
            original_skus_by_loc=original_skus_by_loc,
            original_qty_by_loc_sku={
                ("00301001", "SKU_A"): 2.0,
                ("00301002", "SKU_B"): 2.0,
                (strict_loc, "SKU_X"): 1.0,
                (strict_loc, "SKU_Y"): 1.0,
                (strict_loc, "SKU_Z"): 1.0,
            },
        )
        # strict blocked なので拒否される
        accepted_to_strict = [m for m in result if str(m.to_loc).zfill(8) == strict_loc]
        assert len(accepted_to_strict) == 0, "元々3SKU以上の混在ロケへの着地は拒否されるべき"

    # ------------------------------------------------------------------
    # テスト7: Pass-C と退避チェーン経由のmoveは複数SKUロケに着地しない
    # ------------------------------------------------------------------
    def test_case7_pass_c_not_allowed_multi_sku(self):
        """Pass-C (p_consol_) は複数SKU同居を許可しない"""
        sku_master = self._base_sku_master()
        inv = self._base_inv()
        target_loc = "00100100"

        cfg = _make_cfg(
            allow_empty_loc_multi_sku=True,
            multi_sku_allowed_chain_prefixes=("p1fifo", "swap_fifo_", "fifo_direct_", "p0rebal_", "p2consol_"),
        )

        # SKU_A が先にtarget_locに入っている状態をシミュ
        original_skus_by_loc = {
            "00301001": {"SKU_A"},
            "00301002": {"SKU_B"},
            target_loc: set(),  # 元々空き
        }
        original_empty_locs = {target_loc}
        original_qty = {
            ("00301001", "SKU_A"): 2.0,
            ("00301002", "SKU_B"): 2.0,
        }

        # Pass-Cのchain_group_id (p_consol_ プレフィックス) で SKU_B → target_loc
        # この時点でtarget_locにSKU_Aが既にいると仮定するため、original_skusに追加
        original_skus_by_loc[target_loc] = {"SKU_A"}

        moves = [
            _make_move("SKU_B", "LB1", 2, "00301002", target_loc, chain_group_id="p_consol_001"),
        ]

        result = _call_enforce(
            sku_master, inv, moves, cfg=cfg,
            original_empty_locs=original_empty_locs,
            original_skus_by_loc=original_skus_by_loc,
            original_qty_by_loc_sku=original_qty,
        )
        # p_consol_ は許可プレフィックスに含まれないので保留（_deferred_foreign_sku に入り最終却下）
        assert len(result) == 0, "Pass-C は複数SKU同居を許可しない"

    def test_case7_evict_chain_not_allowed_multi_sku(self):
        """退避チェーン (evict_) は複数SKU同居を許可しない"""
        sku_master = self._base_sku_master()
        inv = self._base_inv()
        target_loc = "00100100"

        cfg = _make_cfg(
            allow_empty_loc_multi_sku=True,
            multi_sku_allowed_chain_prefixes=("p1fifo", "swap_fifo_", "fifo_direct_", "p0rebal_", "p2consol_"),
        )
        original_skus_by_loc = {
            "00301001": {"SKU_A"},
            "00301002": {"SKU_B"},
            target_loc: {"SKU_A"},  # 既にSKU_Aがいる
        }
        original_empty_locs = {target_loc}
        original_qty = {
            ("00301001", "SKU_A"): 2.0,
            ("00301002", "SKU_B"): 2.0,
            (target_loc, "SKU_A"): 1.0,
        }
        moves = [
            _make_move("SKU_B", "LB1", 2, "00301002", target_loc, chain_group_id="evict_001"),
        ]

        result = _call_enforce(
            sku_master, inv, moves, cfg=cfg,
            original_empty_locs=original_empty_locs,
            original_skus_by_loc=original_skus_by_loc,
            original_qty_by_loc_sku=original_qty,
        )
        assert len(result) == 0, "退避チェーンは複数SKU同居を許可しない"

    # ------------------------------------------------------------------
    # テスト8: シミュ中の逐次状態更新（1SKU→2SKU→3SKUの追加受入）
    # ------------------------------------------------------------------
    def test_case8_sequential_state_update(self):
        """許可されたパスプレフィックスで3つのSKUが順番に空きロケへ着地"""
        from app.services.optimizer import Move

        sku_master = _make_sku_master([
            {"sku_id": "SKU_A", "pack_qty": 30, "vol_m3": 0.1},
            {"sku_id": "SKU_B", "pack_qty": 30, "vol_m3": 0.1},
            {"sku_id": "SKU_C", "pack_qty": 30, "vol_m3": 0.1},
        ])
        inv = _make_inv([
            {"sku": "SKU_A", "lot": "LA1", "lot_key": 20250101, "loc": "00301001", "qty": 1, "vol_each": 0.1},
            {"sku": "SKU_B", "lot": "LB1", "lot_key": 20250101, "loc": "00301002", "qty": 1, "vol_each": 0.1},
            {"sku": "SKU_C", "lot": "LC1", "lot_key": 20250101, "loc": "00301003", "qty": 1, "vol_each": 0.1},
        ])
        target_loc = "00100100"

        cfg = _make_cfg(
            allow_empty_loc_multi_sku=True,
            multi_sku_max_per_loc=3,
            multi_sku_level1_vol_cap=0.9,
            multi_sku_level2_vol_cap=0.3,
            multi_sku_pack_band_match=True,
            multi_sku_target_levels=(1, 2),
            multi_sku_allowed_chain_prefixes=("p2consol_",),
        )

        original_empty_locs = {target_loc}
        original_skus_by_loc = {
            "00301001": {"SKU_A"},
            "00301002": {"SKU_B"},
            "00301003": {"SKU_C"},
            target_loc: set(),
        }
        original_qty = {
            ("00301001", "SKU_A"): 1.0,
            ("00301002", "SKU_B"): 1.0,
            ("00301003", "SKU_C"): 1.0,
        }

        # 3つの移動、全て p2consol_ プレフィックス → 許可される
        moves = [
            _make_move("SKU_A", "LA1", 1, "00301001", target_loc, chain_group_id="p2consol_001", execution_order=1),
            _make_move("SKU_B", "LB1", 1, "00301002", target_loc, chain_group_id="p2consol_002", execution_order=1),
            _make_move("SKU_C", "LC1", 1, "00301003", target_loc, chain_group_id="p2consol_003", execution_order=1),
        ]

        result = _call_enforce(
            sku_master, inv, moves, cfg=cfg,
            original_empty_locs=original_empty_locs,
            original_skus_by_loc=original_skus_by_loc,
            original_qty_by_loc_sku=original_qty,
        )

        # 全3移動が受理されるべき（各SKUは異なるchain_group_idなので別々に処理）
        assert len(result) == 3, f"3移動全て受理されるべき: {len(result)}件のみ受理"
        dest_locs = {str(m.to_loc).zfill(8) for m in result}
        assert target_loc in dest_locs


# ============================================================================
# _get_sku_pack_band のユニットテスト
# ============================================================================

class TestGetSkuPackBand:
    """_get_sku_pack_band ヘルパー関数のテスト。"""

    def test_small_band(self):
        from app.services.optimizer import _get_sku_pack_band
        pack_map = pd.Series({"SKU_S": 10.0})
        cfg = _make_cfg(pack_low_max=12, pack_high_min=50)
        assert _get_sku_pack_band("SKU_S", pack_map, cfg) == "small"

    def test_large_band(self):
        from app.services.optimizer import _get_sku_pack_band
        pack_map = pd.Series({"SKU_L": 60.0})
        cfg = _make_cfg(pack_low_max=12, pack_high_min=50)
        assert _get_sku_pack_band("SKU_L", pack_map, cfg) == "large"

    def test_medium_band(self):
        from app.services.optimizer import _get_sku_pack_band
        pack_map = pd.Series({"SKU_M": 30.0})
        cfg = _make_cfg(pack_low_max=12, pack_high_min=50)
        assert _get_sku_pack_band("SKU_M", pack_map, cfg) == "medium"

    def test_boundary_low_max(self):
        from app.services.optimizer import _get_sku_pack_band
        pack_map = pd.Series({"SKU_BL": 12.0})
        cfg = _make_cfg(pack_low_max=12, pack_high_min=50)
        assert _get_sku_pack_band("SKU_BL", pack_map, cfg) == "small"

    def test_boundary_high_min(self):
        from app.services.optimizer import _get_sku_pack_band
        pack_map = pd.Series({"SKU_BH": 50.0})
        cfg = _make_cfg(pack_low_max=12, pack_high_min=50)
        assert _get_sku_pack_band("SKU_BH", pack_map, cfg) == "large"

    def test_unknown_sku(self):
        from app.services.optimizer import _get_sku_pack_band
        pack_map = pd.Series({"SKU_A": 30.0})
        cfg = _make_cfg()
        assert _get_sku_pack_band("UNKNOWN", pack_map, cfg) == "medium"

    def test_no_pack_map(self):
        from app.services.optimizer import _get_sku_pack_band
        cfg = _make_cfg()
        assert _get_sku_pack_band("SKU_A", None, cfg) == "medium"


# ============================================================================
# 追加テスト（reviewer指摘 W-1/W-2/W-4 回帰防止）
# ============================================================================

class TestMultiSkuReviewerFixes:
    """W-1, W-2, W-4 の修正に対する回帰テスト。"""

    # ------------------------------------------------------------------
    # テスト R-1: allow_empty_loc_multi_sku=False で完全無効化
    # ------------------------------------------------------------------
    def test_r1_feature_disabled_blocks_coexistence(self):
        """allow_empty_loc_multi_sku=False のとき、許可プレフィックスでも同居は却下される"""
        sku_master = _make_sku_master([
            {"sku_id": "SKU_A", "pack_qty": 30, "vol_m3": 0.05},
            {"sku_id": "SKU_B", "pack_qty": 30, "vol_m3": 0.05},
        ])
        inv = _make_inv([
            {"sku": "SKU_A", "lot": "LA1", "lot_key": 20250101, "loc": "00301001", "qty": 1, "vol_each": 0.05},
            {"sku": "SKU_B", "lot": "LB1", "lot_key": 20250101, "loc": "00301002", "qty": 1, "vol_each": 0.05},
        ])
        target_loc = "00100100"

        cfg = _make_cfg(
            allow_empty_loc_multi_sku=False,
            multi_sku_allowed_chain_prefixes=("p1fifo", "swap_fifo_", "fifo_direct_", "p0rebal_", "p2consol_"),
        )
        original_empty_locs = {target_loc}
        original_skus_by_loc = {
            "00301001": {"SKU_A"},
            "00301002": {"SKU_B"},
            target_loc: {"SKU_A"},  # すでにSKU_Aがいる
        }
        original_qty = {
            ("00301001", "SKU_A"): 1.0,
            ("00301002", "SKU_B"): 1.0,
            (target_loc, "SKU_A"): 1.0,
        }

        # p2consol_ プレフィックスでも無効化されていれば却下される
        moves = [
            _make_move("SKU_B", "LB1", 1, "00301002", target_loc, chain_group_id="p2consol_001"),
        ]

        result = _call_enforce(
            sku_master, inv, moves, cfg=cfg,
            original_empty_locs=original_empty_locs,
            original_skus_by_loc=original_skus_by_loc,
            original_qty_by_loc_sku=original_qty,
        )
        assert len(result) == 0, "allow_empty_loc_multi_sku=False なら許可プレフィックスでも却下されるべき"

    # ------------------------------------------------------------------
    # テスト R-2: メインループで複数 move を連続受理する際に _sim_vol_by_loc が累積更新されることを検証
    # （W-2 回帰防止: 複数 move 間で容積追跡が正しく機能することを確認）
    # ------------------------------------------------------------------
    def test_r2_sequential_vol_tracking_across_moves(self):
        """複数 move 間で shelf_usage が累積更新され、容積超過を正しく却下する

        段別容積上限は capacity ゲートで管理（全着地に常時適用）。
        1件目 (0.5㎥) は 0.5 ≤ 0.9 で受理、2件目 (0.5㎥) は累積1.0 > 0.9 で却下。
        target_loc は元々空き（shelf_usage=0）。
        """
        sku_master = _make_sku_master([
            {"sku_id": "SKU_B", "pack_qty": 30, "vol_m3": 0.5},
            {"sku_id": "SKU_C", "pack_qty": 30, "vol_m3": 0.5},
        ])
        target_loc = "00100100"
        inv = _make_inv([
            {"sku": "SKU_B", "lot": "LB1", "lot_key": 20250101, "loc": "00301002", "qty": 1, "vol_each": 0.5},
            {"sku": "SKU_C", "lot": "LC1", "lot_key": 20250101, "loc": "00301003", "qty": 1, "vol_each": 0.5},
        ])

        cfg = _make_cfg(
            allow_empty_loc_multi_sku=True,
            multi_sku_max_per_loc=3,
            multi_sku_level1_vol_cap=0.9,
            multi_sku_level2_vol_cap=0.3,
            multi_sku_pack_band_match=False,
            multi_sku_target_levels=(1, 2),
            multi_sku_allowed_chain_prefixes=("p2consol_",),
            enforce_level_vol_cap=True,
        )
        original_empty_locs = {target_loc}
        original_skus_by_loc = {
            target_loc: set(),  # 元々空き（SKUなし）
            "00301002": {"SKU_B"},
            "00301003": {"SKU_C"},
        }
        original_qty = {
            ("00301002", "SKU_B"): 1.0,
            ("00301003", "SKU_C"): 1.0,
        }

        # SKU_B と SKU_C を target_loc（Lv1、cap 0.9㎥）に送る。
        # SKU_B: used=0 + 0.5 = 0.5 ≤ 0.9 → 受理
        # SKU_C: used=0.5 + 0.5 = 1.0 > 0.9 → 却下
        moves = [
            _make_move("SKU_B", "LB1", 1, "00301002", target_loc, chain_group_id="p2consol_001", execution_order=1),
            _make_move("SKU_C", "LC1", 1, "00301003", target_loc, chain_group_id="p2consol_002", execution_order=2),
        ]

        result = _call_enforce(
            sku_master, inv, moves, cfg=cfg,
            original_empty_locs=original_empty_locs,
            original_skus_by_loc=original_skus_by_loc,
            original_qty_by_loc_sku=original_qty,
        )

        accepted_to_target = [m for m in result if str(m.to_loc).zfill(8) == target_loc]
        assert len(accepted_to_target) == 1, (
            f"容積 0.5+0.5=1.0 > cap 0.9 のため2件目は却下されるべき。受理={len(accepted_to_target)}件"
        )

    # ------------------------------------------------------------------
    # テスト R-3: スワップ受理後に to_loc への追加SKUが容積超過で却下
    # （W-1 回帰防止: スワップ受理時に _sim_vol_by_loc が更新されていることを確認）
    # ------------------------------------------------------------------
    def test_r3_swap_accepted_vol_tracked(self):
        """スワップ受理後に _sim_vol_by_loc が更新され、同ロケへの後続追加が容積超過で却下される"""
        # 手順:
        # 1. SKU_A（loc_a）↔ SKU_B（target_loc）のスワップ → スワップ受理
        #    - target_loc に SKU_A: 容積 0.7 が入る
        # 2. SKU_C → target_loc（p2consol_, 許可）: 容積 0.3 追加 → 合計 1.0 > cap 0.9 → 却下
        # スワップ時に _sim_vol_by_loc が更新されていれば SKU_C は容積超過で却下される

        sku_master = _make_sku_master([
            {"sku_id": "SKU_A", "pack_qty": 30, "vol_m3": 0.7},
            {"sku_id": "SKU_B", "pack_qty": 30, "vol_m3": 0.3},
            {"sku_id": "SKU_C", "pack_qty": 30, "vol_m3": 0.3},
        ])
        loc_a = "00301001"
        target_loc = "00100100"
        loc_c = "00301003"

        inv = _make_inv([
            {"sku": "SKU_A", "lot": "LA1", "lot_key": 20250101, "loc": loc_a, "qty": 1, "vol_each": 0.7},
            {"sku": "SKU_B", "lot": "LB1", "lot_key": 20250101, "loc": target_loc, "qty": 1, "vol_each": 0.3},
            {"sku": "SKU_C", "lot": "LC1", "lot_key": 20250101, "loc": loc_c, "qty": 1, "vol_each": 0.3},
        ])

        cfg = _make_cfg(
            allow_empty_loc_multi_sku=True,
            multi_sku_max_per_loc=3,
            multi_sku_level1_vol_cap=0.9,
            multi_sku_level2_vol_cap=0.3,
            multi_sku_pack_band_match=False,
            multi_sku_target_levels=(1, 2),
            multi_sku_allowed_chain_prefixes=("swap_fifo_", "t_consol_"),
        )
        # target_loc は元々空き（スワップ後 SKU_A が入る）
        original_empty_locs = {target_loc}
        original_skus_by_loc = {
            loc_a: {"SKU_A"},
            target_loc: set(),  # 元々空き（スワップ相手として使う）
            loc_c: {"SKU_C"},
        }
        original_qty = {
            (loc_a, "SKU_A"): 1.0,
            (target_loc, "SKU_B"): 1.0,
            (loc_c, "SKU_C"): 1.0,
        }

        # スワップ: SKU_A(loc_a→target_loc) ↔ SKU_B(target_loc→loc_a)
        # その後: SKU_C → target_loc (p2consol_)
        # enforce_constraints は (chain_group_id, execution_order) でソートするため、
        # スワップが先に処理されるよう、後続の同居moveは辞書順で後ろに来る名前にする
        # swap_ < t_consol_ なので swap が先に処理される
        swap_cg = "swap_fifo_test001"
        moves = [
            _make_move("SKU_A", "LA1", 1, loc_a, target_loc, chain_group_id=swap_cg, execution_order=1),
            _make_move("SKU_B", "LB1", 1, target_loc, loc_a, chain_group_id=swap_cg, execution_order=1),
            _make_move("SKU_C", "LC1", 1, loc_c, target_loc, chain_group_id="t_consol_001", execution_order=2),
        ]

        result = _call_enforce(
            sku_master, inv, moves, cfg=cfg,
            original_empty_locs=original_empty_locs,
            original_skus_by_loc=original_skus_by_loc,
            original_qty_by_loc_sku=original_qty,
        )

        # スワップ2件は受理。SKU_Cへの移動は容積超過で却下されるべき
        accepted_swap = [m for m in result if getattr(m, 'chain_group_id', '') == swap_cg]
        accepted_skuc_to_target = [m for m in result
                                   if str(m.to_loc).zfill(8) == target_loc
                                   and str(m.sku_id) == "SKU_C"]

        assert len(accepted_swap) == 2, f"スワップ2件は受理されるべき: {len(accepted_swap)}件"
        assert len(accepted_skuc_to_target) == 0, (
            f"スワップ後の容積(0.7) + SKU_C(0.3) = 1.0 > cap 0.9 で却下されるべき: {len(accepted_skuc_to_target)}件受理"
        )


# ============================================================================
# 新規テスト: 元々1-2SKUロケへの追加受入（拡張仕様）
# ============================================================================

class TestMultiSkuReceivableLocs:
    """元々1-2SKUのロケへの追加受入テスト（Case R1-R8）。"""

    def _base_cfg(self, **kwargs):
        defaults = dict(
            allow_empty_loc_multi_sku=True,
            multi_sku_max_per_loc=3,
            multi_sku_level1_vol_cap=0.9,
            multi_sku_level2_vol_cap=0.3,
            multi_sku_pack_band_match=True,
            multi_sku_target_levels=(1, 2),
            multi_sku_allowed_chain_prefixes=("p2consol_",),
            enforce_level_vol_cap=True,
        )
        defaults.update(kwargs)
        return _make_cfg(**defaults)

    # ------------------------------------------------------------------
    # Case R1: 元々2SKUロケ（同pack帯・容積内）に1SKU追加 → 受理（3SKU）
    # ------------------------------------------------------------------
    def test_r1_two_sku_loc_accepts_one_more(self):
        """元々2SKUロケ（同pack帯・0.1㎥）に1SKU追加 → 受理して3SKU"""
        cfg = self._base_cfg()
        loc = "00100100"  # Lv1
        receivable = {loc}  # 1-2SKUロケを receivable として登録
        sim_skus = {loc: {"SKU_X", "SKU_Y"}}  # 既に2SKUいる
        sim_vol = {loc: 0.1}
        band = {"SKU_X": "medium", "SKU_Y": "medium", "SKU_A": "medium"}

        allowed, reason = _call_check(loc, "SKU_A", 1, 0.05, receivable, sim_skus, sim_vol, band, 1, cfg)
        assert allowed is True, f"2SKUロケへの追加は受理されるべき: {reason}"

    # ------------------------------------------------------------------
    # Case R2: 元々2SKUロケに2SKU追加 → 1件目受理、2件目却下（4SKU超）
    # ------------------------------------------------------------------
    def test_r2_two_sku_loc_second_addition_rejected(self):
        """元々2SKUロケに2SKU連続追加 → 1件目(3SKU)は受理、2件目(4SKU)は却下"""
        cfg = self._base_cfg()
        loc = "00100100"
        receivable = {loc}
        band = {"SKU_X": "medium", "SKU_Y": "medium", "SKU_A": "medium", "SKU_B": "medium"}

        # 1件目: 2SKU → 3SKU → 受理
        sim_skus = {loc: {"SKU_X", "SKU_Y"}}
        sim_vol = {loc: 0.1}
        allowed1, reason1 = _call_check(loc, "SKU_A", 1, 0.05, receivable, sim_skus, sim_vol, band, 1, cfg)
        assert allowed1 is True, f"1件目は受理されるべき: {reason1}"

        # 状態更新（SKU_A を追加）
        sim_skus[loc].add("SKU_A")
        sim_vol[loc] += 0.05

        # 2件目: 3SKU → 4SKU → 却下（上限超過）
        allowed2, reason2 = _call_check(loc, "SKU_B", 1, 0.05, receivable, sim_skus, sim_vol, band, 1, cfg)
        assert allowed2 is False, "2件目は上限超過で却下されるべき"
        assert reason2 == "multi_sku_count_exceeded"

    # ------------------------------------------------------------------
    # Case R3: 元々1SKUロケに別SKU2つ追加 → 両方受理（3SKU）
    # ------------------------------------------------------------------
    def test_r3_one_sku_loc_accepts_two_more(self):
        """元々1SKUロケに同pack帯SKUを2つ追加 → 両方受理して3SKU"""
        cfg = self._base_cfg()
        loc = "00100100"
        receivable = {loc}
        band = {"SKU_X": "medium", "SKU_A": "medium", "SKU_B": "medium"}

        # 1件目: 1SKU → 2SKU → 受理
        sim_skus = {loc: {"SKU_X"}}
        sim_vol = {loc: 0.1}
        allowed1, reason1 = _call_check(loc, "SKU_A", 1, 0.1, receivable, sim_skus, sim_vol, band, 1, cfg)
        assert allowed1 is True, f"1件目は受理されるべき: {reason1}"
        sim_skus[loc].add("SKU_A")
        sim_vol[loc] += 0.1

        # 2件目: 2SKU → 3SKU → 受理
        allowed2, reason2 = _call_check(loc, "SKU_B", 1, 0.1, receivable, sim_skus, sim_vol, band, 1, cfg)
        assert allowed2 is True, f"2件目は受理されるべき: {reason2}"

    # ------------------------------------------------------------------
    # Case R4: 元々1SKUロケに別SKU3つ追加 → 3つ目は却下
    # ------------------------------------------------------------------
    def test_r4_one_sku_loc_third_addition_rejected(self):
        """元々1SKUロケにSKUを3つ追加しようとすると3つ目で却下"""
        cfg = self._base_cfg()
        loc = "00100100"
        receivable = {loc}
        band = {"SKU_X": "medium", "SKU_A": "medium", "SKU_B": "medium", "SKU_C": "medium"}

        # 2つ追加（1SKU→3SKU）まで受理
        sim_skus = {loc: {"SKU_X", "SKU_A", "SKU_B"}}  # すでに3SKU
        sim_vol = {loc: 0.15}

        # 4つ目は却下
        allowed, reason = _call_check(loc, "SKU_C", 1, 0.05, receivable, sim_skus, sim_vol, band, 1, cfg)
        assert allowed is False, "4SKU目は却下されるべき"
        assert reason == "multi_sku_count_exceeded"

    # ------------------------------------------------------------------
    # Case R5: 元々3SKUロケへの投入 → 拒否（strict blocked）
    # ------------------------------------------------------------------
    def test_r5_strict_three_sku_loc_rejected(self):
        """元々3SKUロケは receivable_locs に含まれない → フォールスルー拒否"""
        cfg = self._base_cfg()
        loc = "00100100"
        # receivable_locs に含まれない（strict blocked）
        receivable: Set[str] = set()
        sim_skus = {loc: {"SKU_X", "SKU_Y", "SKU_Z"}}
        sim_vol = {loc: 0.15}
        band = {"SKU_X": "medium", "SKU_Y": "medium", "SKU_Z": "medium", "SKU_A": "medium"}

        allowed, reason = _call_check(loc, "SKU_A", 1, 0.05, receivable, sim_skus, sim_vol, band, 1, cfg)
        assert allowed is False
        assert reason is None  # フォールスルー（従来ロジックに委ねる）

    # ------------------------------------------------------------------
    # Case R6: 元々2SKUロケで pack帯不一致SKU投入 → 却下
    # ------------------------------------------------------------------
    def test_r6_two_sku_loc_pack_band_mismatch_rejected(self):
        """元々2SKUロケ（medium）にlarge SKUを追加 → pack帯不一致で却下"""
        cfg = self._base_cfg(multi_sku_pack_band_match=True)
        loc = "00100100"
        receivable = {loc}
        sim_skus = {loc: {"SKU_X", "SKU_Y"}}
        sim_vol = {loc: 0.1}
        band = {"SKU_X": "medium", "SKU_Y": "medium", "SKU_LARGE": "large"}

        allowed, reason = _call_check(loc, "SKU_LARGE", 1, 0.1, receivable, sim_skus, sim_vol, band, 1, cfg)
        assert allowed is False
        assert reason == "multi_sku_pack_band_mismatch"

    # ------------------------------------------------------------------
    # Case R7: 元々2SKUロケ（Lv2、合計0.25㎥）に0.1㎥SKU → level_vol_cap で却下
    # ------------------------------------------------------------------
    def test_r7_two_sku_loc_lv2_vol_cap_exceeded(self):
        """元々2SKUロケ（Lv2, 0.25㎥）に0.1㎥SKU追加 → 合計0.35 > cap 0.3 で却下"""
        sku_master = _make_sku_master([
            {"sku_id": "SKU_X", "pack_qty": 30, "vol_m3": 0.125},
            {"sku_id": "SKU_Y", "pack_qty": 30, "vol_m3": 0.125},
            {"sku_id": "SKU_A", "pack_qty": 30, "vol_m3": 0.1},
        ])
        target_loc = "00200100"  # Lv2

        inv = _make_inv([
            # target_loc に既存2SKU（合計0.25㎥）を追加 → shelf_usage[target_loc]=0.25 が計算される
            {"sku": "SKU_X", "lot": "LX1", "lot_key": 20250101, "loc": target_loc, "qty": 1, "vol_each": 0.125},
            {"sku": "SKU_Y", "lot": "LY1", "lot_key": 20250101, "loc": target_loc, "qty": 1, "vol_each": 0.125},
            {"sku": "SKU_A", "lot": "LA1", "lot_key": 20250101, "loc": "00301003", "qty": 1, "vol_each": 0.1},
        ])

        cfg = _make_cfg(
            allow_empty_loc_multi_sku=True,
            multi_sku_max_per_loc=3,
            multi_sku_level1_vol_cap=0.9,
            multi_sku_level2_vol_cap=0.3,
            multi_sku_pack_band_match=False,
            multi_sku_target_levels=(1, 2),
            multi_sku_allowed_chain_prefixes=("p2consol_",),
            enforce_level_vol_cap=True,
        )
        # target_loc は元々2SKU（receivable）で既に0.25㎥使用中
        original_empty_locs: Set[str] = set()  # 空きロケはなし
        original_skus_by_loc = {
            "00301001": {"SKU_X"},
            "00301002": {"SKU_Y"},
            "00301003": {"SKU_A"},
            target_loc: {"SKU_X", "SKU_Y"},  # 元々2SKU
        }
        original_qty = {
            ("00301001", "SKU_X"): 1.0,
            ("00301002", "SKU_Y"): 1.0,
            ("00301003", "SKU_A"): 1.0,
            (target_loc, "SKU_X"): 1.0,
            (target_loc, "SKU_Y"): 1.0,
        }
        moves = [
            _make_move("SKU_A", "LA1", 1, "00301003", target_loc, chain_group_id="p2consol_001"),
        ]

        result = _call_enforce(
            sku_master, inv, moves, cfg=cfg,
            original_empty_locs=original_empty_locs,
            original_skus_by_loc=original_skus_by_loc,
            original_qty_by_loc_sku=original_qty,
        )
        # level_vol_cap (Lv2: 0.3㎥) を超えるため却下
        accepted_to_target = [m for m in result if str(m.to_loc).zfill(8) == target_loc]
        assert len(accepted_to_target) == 0, (
            f"Lv2容積 0.25+0.1=0.35 > cap 0.3 のため却下されるべき: {len(accepted_to_target)}件受理"
        )

    # ------------------------------------------------------------------
    # Case R8: 元々2SKUロケからの move-out → 依然として禁止（source blocked）
    # ------------------------------------------------------------------
    def test_r8_two_sku_loc_source_blocked(self):
        """元々2SKUロケからの移動は source_blocked で拒否される"""
        sku_master = _make_sku_master([
            {"sku_id": "SKU_X", "pack_qty": 30, "vol_m3": 0.05},
            {"sku_id": "SKU_Y", "pack_qty": 30, "vol_m3": 0.05},
        ])
        source_loc = "00100100"  # Lv1: 元々2SKU
        dest_loc = "00100200"    # Lv1: 元々空き

        inv = _make_inv([
            {"sku": "SKU_X", "lot": "LX1", "lot_key": 20250101, "loc": source_loc, "qty": 1, "vol_each": 0.05},
        ])

        cfg = _make_cfg(
            allow_empty_loc_multi_sku=True,
            multi_sku_allowed_chain_prefixes=("p2consol_",),
        )
        original_empty_locs: Set[str] = {dest_loc}
        original_skus_by_loc = {
            source_loc: {"SKU_X", "SKU_Y"},  # 元々2SKU → source blocked
            dest_loc: set(),
        }
        original_qty = {
            (source_loc, "SKU_X"): 1.0,
            (source_loc, "SKU_Y"): 1.0,
        }
        moves = [
            _make_move("SKU_X", "LX1", 1, source_loc, dest_loc, chain_group_id="p2consol_001"),
        ]

        # source_blocked を明示: 元々2SKU以上のロケは移動元に使えない
        result = _call_enforce(
            sku_master, inv, moves, cfg=cfg,
            original_empty_locs=original_empty_locs,
            original_skus_by_loc=original_skus_by_loc,
            original_qty_by_loc_sku=original_qty,
            blocked_source_locs={source_loc},
        )
        # source_blocked のため拒否
        assert len(result) == 0, f"元々2SKUロケからの移動は拒否されるべき: {len(result)}件受理"
