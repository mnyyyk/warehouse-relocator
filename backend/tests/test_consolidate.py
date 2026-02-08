"""Tests for _consolidate_eviction_chains post-consolidation cleanup."""
import pytest


def _consolidate():
    """Import helper – import at call time to avoid import-order issues."""
    from app.routers.upload import _consolidate_eviction_chains
    return _consolidate_eviction_chains


# ------------------------------------------------------------------
# 1. Self-ref removal: consolidated moves where from_loc == to_loc
# ------------------------------------------------------------------

def test_consolidated_self_ref_removed():
    """A two-step chain that nets out to from==to should be dropped."""
    moves = [
        # Step 1: A → B
        {"sku_id": "X1", "lot": "L1", "from_loc": "00100101", "to_loc": "00200202",
         "chain_group_id": "evict_abc", "execution_order": 1, "reason": "pass-1"},
        # Step 2: B → A  (same SKU+lot comes back)
        {"sku_id": "X1", "lot": "L1", "from_loc": "00200202", "to_loc": "00100101",
         "chain_group_id": "evict_abc", "execution_order": 2, "reason": "pass-0"},
    ]
    result = _consolidate()(moves)
    # After consolidation: from=00100101 to=00100101 → removed
    assert len(result) == 0


def test_single_self_ref_removed():
    """A single move with from_loc == to_loc should be dropped."""
    moves = [
        {"sku_id": "X1", "lot": "L1", "from_loc": "00100101", "to_loc": "00100101",
         "chain_group_id": "p1fifo_abc", "execution_order": 1, "reason": "noop"},
    ]
    result = _consolidate()(moves)
    assert len(result) == 0


def test_non_self_ref_preserved():
    """Normal moves (from != to) are preserved."""
    moves = [
        {"sku_id": "X1", "lot": "L1", "from_loc": "00100101", "to_loc": "00200202",
         "chain_group_id": "p1fifo_abc", "execution_order": 1, "reason": "move"},
    ]
    result = _consolidate()(moves)
    assert len(result) == 1
    assert result[0]["from_loc"] == "00100101"
    assert result[0]["to_loc"] == "00200202"


# ------------------------------------------------------------------
# 2. execution_order renumbering after self-ref removal
# ------------------------------------------------------------------

def test_execution_order_renumbered_after_removal():
    """When a self-ref member is removed from a group, orders renumber from 1."""
    moves = [
        # Real move (different SKU, so not consolidated with next)
        {"sku_id": "A1", "lot": "L1", "from_loc": "00102505", "to_loc": "00100511",
         "chain_group_id": "dep_aaa", "execution_order": 1, "reason": "evac"},
        # Self-ref (different SKU)
        {"sku_id": "B1", "lot": "L2", "from_loc": "00102505", "to_loc": "00102505",
         "chain_group_id": "dep_aaa", "execution_order": 3, "reason": "noop"},
        # Another real move
        {"sku_id": "C1", "lot": "L3", "from_loc": "00100511", "to_loc": "00200303",
         "chain_group_id": "dep_aaa", "execution_order": 2, "reason": "place"},
    ]
    result = _consolidate()(moves)
    # B1 self-ref removed → 2 remaining, renumbered to [1, 2]
    assert len(result) == 2
    orders = [m["execution_order"] for m in result]
    assert sorted(orders) == [1, 2]


def test_execution_order_single_member_gets_1():
    """After removal, if only 1 member left in group, order = 1."""
    moves = [
        {"sku_id": "A1", "lot": "L1", "from_loc": "00100101", "to_loc": "00200202",
         "chain_group_id": "dep_xxx", "execution_order": 3, "reason": "move"},
        # This one is self-ref with same chain_group but different SKU
        {"sku_id": "B1", "lot": "L2", "from_loc": "00300303", "to_loc": "00300303",
         "chain_group_id": "dep_xxx", "execution_order": 5, "reason": "noop"},
    ]
    result = _consolidate()(moves)
    assert len(result) == 1
    assert result[0]["execution_order"] == 1
    assert result[0]["sku_id"] == "A1"


# ------------------------------------------------------------------
# 3. CSV row ordering: within each chain group, rows are in
#    execution_order sequence
# ------------------------------------------------------------------

def test_csv_row_order_matches_execution_order():
    """Within a dep_ group, CSV rows must be in execution_order sequence."""
    moves = [
        # Placer (wants to go to X)
        {"sku_id": "P1", "lot": "LP", "from_loc": "00400101", "to_loc": "00200303",
         "chain_group_id": "dep_bbb", "execution_order": 2, "reason": "place"},
        # Some standalone move in between
        {"sku_id": "S1", "lot": "LS", "from_loc": "00100101", "to_loc": "00300404",
         "chain_group_id": "p1fifo_zzz", "execution_order": 1, "reason": "standalone"},
        # Evacuator (must go first — order 1)
        {"sku_id": "E1", "lot": "LE", "from_loc": "00200303", "to_loc": "00500505",
         "chain_group_id": "dep_bbb", "execution_order": 1, "reason": "evac"},
    ]
    result = _consolidate()(moves)
    assert len(result) == 3

    # Find dep_bbb members in result order
    dep_members = [m for m in result if m["chain_group_id"] == "dep_bbb"]
    assert len(dep_members) == 2
    assert dep_members[0]["execution_order"] == 1  # evacuator first
    assert dep_members[1]["execution_order"] == 2  # placer second


# ------------------------------------------------------------------
# 4. Multi-step consolidation that doesn't net to self-ref
# ------------------------------------------------------------------

def test_multi_step_consolidation_preserves_non_self_ref():
    """A→B→C consolidates to A→C (not self-ref), should be kept."""
    moves = [
        {"sku_id": "X1", "lot": "L1", "from_loc": "00100101", "to_loc": "00200202",
         "chain_group_id": "evict_abc", "execution_order": 1, "reason": "step1"},
        {"sku_id": "X1", "lot": "L1", "from_loc": "00200202", "to_loc": "00300303",
         "chain_group_id": "evict_abc", "execution_order": 2, "reason": "step2"},
    ]
    result = _consolidate()(moves)
    assert len(result) == 1
    assert result[0]["from_loc"] == "00100101"
    assert result[0]["to_loc"] == "00300303"


# ------------------------------------------------------------------
# 5. Mixed scenario: some consolidated to self-ref, some real
# ------------------------------------------------------------------

def test_mixed_scenario():
    """Mix of self-ref removals and real moves with correct ordering."""
    moves = [
        # Chain 1: A→B→A (nets to self-ref → removed)
        {"sku_id": "X1", "lot": "L1", "from_loc": "00100101", "to_loc": "00200202",
         "chain_group_id": "evict_1", "execution_order": 1, "reason": "s1"},
        {"sku_id": "X1", "lot": "L1", "from_loc": "00200202", "to_loc": "00100101",
         "chain_group_id": "evict_1", "execution_order": 2, "reason": "s2"},
        # Chain 2: single real move (kept)
        {"sku_id": "Y1", "lot": "L2", "from_loc": "00300303", "to_loc": "00400404",
         "chain_group_id": "p1fifo_2", "execution_order": 1, "reason": "real"},
        # Chain 3: pre-existing self-ref (removed)
        {"sku_id": "Z1", "lot": "L3", "from_loc": "00500505", "to_loc": "00500505",
         "chain_group_id": "dep_3", "execution_order": 2, "reason": "noop"},
    ]
    result = _consolidate()(moves)
    assert len(result) == 1
    assert result[0]["sku_id"] == "Y1"
    assert result[0]["execution_order"] == 1
