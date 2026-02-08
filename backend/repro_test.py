"""Reproduce execution_order issue with real test data."""
import sys, os
os.environ['DATABASE_URL'] = 'sqlite:///test_repro.db'
os.environ.setdefault('OPENAI_API_KEY', '')

import pandas as pd
from collections import defaultdict

data_dir = '../warehouse-optimizer_testdata'

def read_csv_auto(path):
    for enc in ['utf-8', 'utf-8-sig', 'cp932', 'utf-16']:
        try:
            return pd.read_csv(path, dtype=str, encoding=enc)
        except:
            continue

# Load data
sku_df = read_csv_auto(f'{data_dir}/sku.csv')
inv_df = read_csv_auto(f'{data_dir}/在庫.csv')
ship_df = read_csv_auto(f'{data_dir}/出荷実績_20260119.csv')
recv_df = read_csv_auto(f'{data_dir}/入荷実績_20260119.csv')
valid_df = read_csv_auto(f'{data_dir}/20250806 ハイネスロケ一覧.xlsx - Bネス有効ロケ.csv')
invalid_df = read_csv_auto(f'{data_dir}/20250806 ハイネスロケ一覧.xlsx - 無効化したロケ一覧.csv')
high_df = read_csv_auto(f'{data_dir}/20250806 ハイネスロケ一覧.xlsx - ハイネスロケ.csv')

# De-duplicate SKU master by 商品ID (same as DB upsert would)
sku_id_col = '商品ID'
if sku_id_col in sku_df.columns:
    sku_df = sku_df.drop_duplicates(subset=[sku_id_col], keep='last')
    print(f"SKU master de-duped: {len(sku_df)} unique SKUs")

from app.routers.upload import _build_loc_union
loc_rows = _build_loc_union(valid_df, invalid_df, high_df, block='B', quality='良品')
loc_master_df = pd.DataFrame(loc_rows)
print(f'Location master: {len(loc_master_df)} rows')

from app.services.optimizer import OptimizerConfig, plan_relocation
cfg = OptimizerConfig()
cfg.max_moves = None
cfg.fill_rate = 0.90
cfg.use_ai = True
cfg.require_volume = True
cfg.rotation_window_days = 90
cfg.chain_depth = 1
cfg.eviction_budget = 50
cfg.touch_budget = 100
cfg.block_codes = ['B']
cfg.quality_names = ['良品']
cfg.max_source_locs_per_sku = 2

print('Starting plan_relocation...')
moves = plan_relocation(
    sku_master=sku_df,
    inventory=inv_df,
    cfg=cfg,
    ai_col_hints={},
    loc_master=loc_master_df,
)
print(f'Total moves: {len(moves)}')

# ---- Analysis ----
chain_groups = defaultdict(list)
for i, m in enumerate(moves):
    cgid = getattr(m, 'chain_group_id', None) or 'NONE'
    chain_groups[cgid].append((i, m))

print()
print('=== execution_order issues (groups with wrong numbering) ===')
issue_count = 0
for cgid, members in sorted(chain_groups.items()):
    orders = [getattr(m, 'execution_order', None) for _, m in members]
    if len(members) >= 2:
        expected = list(range(1, len(members)+1))
        actual = sorted([o for o in orders if o is not None])
        if actual != expected:
            issue_count += 1
            print(f'  {cgid}: orders={orders} (expected {expected})')
            for idx, m in members:
                print(f'    Row {idx+1}: {m.sku_id} from={m.from_loc} to={m.to_loc} order={getattr(m,"execution_order",None)}')

if issue_count == 0:
    print('  None found!')

# Check to_loc/from_loc conflicts
print()
print('=== to_loc/from_loc conflicts (CSV order) ===')
conflict_count = 0
for i, a in enumerate(moves):
    for j, b in enumerate(moves):
        if i != j and a.to_loc == b.from_loc:
            ok = j < i
            same = (getattr(a,'chain_group_id',None) == getattr(b,'chain_group_id',None))
            status = 'OK' if ok and same else 'NG'
            if status == 'NG':
                conflict_count += 1
                print(f'  {status} Loc={a.to_loc}: Evac[{j+1}] {b.sku_id} (chain={getattr(b,"chain_group_id",None)}, order={getattr(b,"execution_order",None)}) -> Placer[{i+1}] {a.sku_id} (chain={getattr(a,"chain_group_id",None)}, order={getattr(a,"execution_order",None)})')

if conflict_count == 0:
    print('  None found! All conflicts resolved correctly.')

# Specific check: dep_ groups with order 3,2
print()
print('=== dep_ groups with order starting > 1 ===')
for cgid, members in sorted(chain_groups.items()):
    if not cgid.startswith('dep_'):
        continue
    orders = [getattr(m, 'execution_order', None) for _, m in members]
    min_order = min(o for o in orders if o is not None)
    if min_order > 1:
        print(f'  {cgid}: orders={orders}')
        for idx, m in members:
            print(f'    Row {idx+1}: {m.sku_id} from={m.from_loc} to={m.to_loc} order={getattr(m,"execution_order",None)}')

print()
print(f'Summary: {len(moves)} moves, {issue_count} order issues, {conflict_count} CSV conflicts')
