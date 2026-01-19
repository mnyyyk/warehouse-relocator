# リロケーション移動案のソート機能

## 概要

リロケーション最適化で生成される移動案は、**効果が高い順（ease_gain降順）**に自動的にソートされます。

## ソート基準

### Ease Gain（取りやすさ改善度）

各移動の効果を以下の式で計算：

```
ease_gain = (from_ease - to_ease) × qty

where:
  ease = level × 10000 + (42 - column) × 100 + depth
  
  - level: レベル（段）低いほど良い
  - column: カラム（列）高いほど良い（入口に近い = 42に近い）
  - depth: 奥行き 低いほど良い
  - qty: ケース数（移動量が多いほど重要）
```

### ソート順序

1. **最も効果が高い移動が最初**
   - ease_gainが大きい = より取りやすい場所への移動
   - ケース数が多い移動を優先

2. **効果の低い移動は後ろ**
   - ease_gainが小さい、またはマイナス

## 実装詳細

### コード位置

`/backend/app/services/optimizer.py` の `plan_relocation()` 関数内：

```python
# Lines 2717-2767
# Sort moves by effectiveness (ease_gain descending)
moves_with_gain = [(m, _calc_ease_gain(m)) for m in moves]
moves_with_gain.sort(key=lambda x: x[1], reverse=True)
moves = [m for m, _ in moves_with_gain]
```

### 計算例

#### 移動例1: 高効果
```
From: 00304201 (level=3, col=4, dep=1)
  → ease = 3×10000 + (42-4)×100 + 1 = 33,801

To: 00101501 (level=1, col=15, dep=1)
  → ease = 1×10000 + (42-15)×100 + 1 = 12,701

Quantity: 24ケース

Ease Gain = (33,801 - 12,701) × 24 = 506,400 ⭐ 高効果
```

#### 移動例2: 低効果
```
From: 00102101 (level=1, col=21, dep=1)
  → ease = 1×10000 + (42-21)×100 + 1 = 12,101

To: 00102201 (level=1, col=22, dep=1)
  → ease = 1×10000 + (42-22)×100 + 1 = 12,001

Quantity: 5ケース

Ease Gain = (12,101 - 12,001) × 5 = 500 ⭐ 低効果
```

## エクスポート時の順序

### API レスポンス

`POST /v1/upload/relocation/start` のレスポンス：

```json
{
  "moves": [
    {
      "sku_id": "A8553N22",
      "lot": "0001JP20250107015989",
      "qty": 24,
      "from_loc": "00304201",
      "to_loc": "00101501",
      "lot_date": "20250107"
    },
    // ... 効果が高い順に並ぶ
  ],
  "count": 56,
  "summary": {
    "ease_gain_sum": 1234567,
    "ease_gain_avg": 22046.2
  }
}
```

### CSV/Excel エクスポート

フロントエンドでCSV/Excelに変換する際、**APIから返される順序をそのまま使用**すれば、効果が高い順に並びます。

```javascript
// Example: Export to CSV in order
const csvRows = response.moves.map((move, index) => ({
  順位: index + 1,  // 1位が最も効果が高い
  SKU_ID: move.sku_id,
  ロット: move.lot,
  ケース数: move.qty,
  移動元: move.from_loc,
  移動先: move.to_loc,
  効果指標: calculateEaseGain(move)  // Optional
}));
```

## 効果指標の表示（オプション）

フロントエンドで各移動の効果を可視化する場合：

```javascript
function calculateEaseGain(move) {
  const parseLocation = (loc) => {
    const str = String(loc).padStart(8, '0');
    return {
      level: parseInt(str.substring(0, 3)),
      column: parseInt(str.substring(3, 6)),
      depth: parseInt(str.substring(6, 8))
    };
  };
  
  const from = parseLocation(move.from_loc);
  const to = parseLocation(move.to_loc);
  
  const fromEase = from.level * 10000 + (42 - from.column) * 100 + from.depth;
  const toEase = to.level * 10000 + (42 - to.column) * 100 + to.depth;
  
  return (fromEase - toEase) * (move.qty || 1);
}
```

## 優先度の解釈

### Ease Gain値の意味

| Ease Gain | 優先度 | 意味 |
|-----------|--------|------|
| > 100,000 | ⭐⭐⭐ 最優先 | 大きなレベル改善（3段→1段）× 大量ケース |
| 10,000 - 100,000 | ⭐⭐ 高優先 | レベル改善または大きなカラム改善 |
| 1,000 - 10,000 | ⭐ 中優先 | カラム改善または小規模レベル改善 |
| 0 - 1,000 | 低優先 | わずかな改善（同レベル内の微調整） |
| < 0 | 要確認 | 理論上は発生しないはず（悪化） |

## メリット

1. **作業効率の最適化**
   - 効果が高い移動から実施できる
   - 時間や人員が限られている場合、上位のみ実施も可能

2. **進捗の可視化**
   - 「上位50%完了」など、効果ベースの進捗管理

3. **意思決定の支援**
   - 効果が低い移動は後回しにする判断が可能
   - コスト対効果の評価

## 注意事項

- **依存関係**: 移動の順序を変更する場合、ロケーションの容量制約に注意
- **ソートエラー**: ソート処理に失敗しても最適化自体は継続（元の順序で返却）
- **パフォーマンス**: 大規模データ（1000+移動）でも高速にソート可能

## ログ出力

ソート実行時のログ：

```
[optimizer] Sorted 56 moves by effectiveness (ease_gain descending)
[optimizer] planned moves=56 (limit=None)
```

SSE進捗メッセージ：
```
移動案作成完了: 56件（効果順ソート済み）
```
