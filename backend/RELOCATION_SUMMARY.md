# リロケーション結果サマリー機能

## 概要

リロケーション最適化実行後、自動的に総合評価レポートが生成されます。

## レポート内容

### 📊 実施結果
- **計画移動数**: 最適化アルゴリズムが提案した総移動件数
- **承認移動数**: 制約チェックを通過した移動件数
- **却下移動数**: ハードルール違反等で却下された移動件数
- **移動率**: 全在庫に対する移動の割合（%）
- **影響SKU数**: 移動対象となったSKUの種類数
- **総ケース数**: 移動されるケースの合計数

### 🔄 SKU集約状況
主要SKU（上位5種類）について：
- **移動前のロケーション分散数**
- **移動後のロケーション集約数**
- 集約度の改善状況を表示
  - ✅ 改善: ロケーション数が減少
  - ⚠️ 分散: ロケーション数が増加
  - → 変化なし

例：
```
A8553N12: ✅ 5→2ロケ
A8553N22: ✅ 6→4ロケ
T87B5N11: ✅ 4→3ロケ
```

### ✅ ハードルール検証
1. **ロット混在チェック**
   - 同一ロケーション内の同一SKUに異なるロットが混在していないか検証
   - 違反がある場合は詳細を表示（ロケーション、SKU、ロット数）
   - ✅ ロット混在なし = 正常
   - ⚠️ ロット混在あり = ハードルール違反

2. **良品在庫の状況**
   - 品質区分別の在庫行数
   - 良品（品質区分=1）の在庫数と全体に対する割合

### 💡 推奨事項
システムが自動的に以下を評価して推奨事項を提示：
- 移動率が低い（<2%）→ より積極的な最適化が可能
- ロット混在検出 → ハードルール違反の修正が必要
- 少数SKUへの集中移動 → 在庫集約が効果的に機能

## データの取得方法

### 1. SSEストリーム経由（リアルタイム）

リロケーション実行中にリアルタイムでレポートを受信：

```javascript
const eventSource = new EventSource(`/v1/upload/relocation/stream?trace_id=${traceId}`);

eventSource.addEventListener('message', (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === 'summary_report') {
    console.log(data.report);  // フォーマット済みテキストレポート
    console.log(data.data);    // 構造化データ
  }
});
```

### 2. コンソールログ

サーバー側のログにも同じレポートが出力されます：

```bash
tail -f /Users/kounoyousuke/warehouse-optimizer/backend/uvicorn.log
```

## レポート例

```
======================================================================
📊 リロケーション結果の総合評価
======================================================================

【実施結果】
  計画移動数: 56 件
  承認移動数: 56 件
  却下移動数: 0 件
  移動率: 1.4%
  影響SKU数: 34 種類
  総ケース数: 294 ケース

【SKU集約状況】
  A8553N12: ✅ 5→2ロケ
  A8553N22: ✅ 6→4ロケ
  T87B5N11: ✅ 4→3ロケ
  A8340NA1: → 4ロケ (変化なし)

【ハードルール検証】
  ✅ ロット混在なし
  良品在庫: 3,789 行 / 全体 5,621 行

【推奨事項】
  • 移動率が低い（<2%）- より積極的な最適化が可能かもしれません
  • 少数のSKUに集中した移動 - 在庫集約が効果的に機能しています

======================================================================
```

## データ構造

### summary_report.data の構造

```typescript
interface RelocationSummary {
  total_planned: number;           // 計画移動数
  total_rejected: number;          // 却下移動数
  total_accepted: number;          // 承認移動数
  affected_skus: number;           // 影響SKU数
  from_locations: number;          // 移動元ロケーション数
  to_locations: number;            // 移動先ロケーション数
  total_cases: number;             // 総ケース数
  move_rate_percent: number;       // 移動率（%）
  
  sku_consolidation: {
    before: { [sku: string]: number };  // SKU別の移動前ロケ数
    after: { [sku: string]: number };   // SKU別の移動後ロケ数
  };
  
  lot_mixing_issues: number;       // ロット混在件数
  lot_mixing_details: Array<{
    location: string;
    sku: string;
    lot_count: number;
    lots: string[];
  }>;
  
  quality_breakdown?: {
    good_items: number;            // 良品行数
    total_items: number;           // 総行数
  };
  
  recommendations: string[];       // 推奨事項の配列
}
```

## 実装詳細

### コード位置
- **レポート生成関数**: `/backend/app/services/optimizer.py` の `_generate_relocation_summary()`
- **SSE送信**: `plan_relocation()` 関数内、`enforce_constraints()` 実行後

### カスタマイズ方法

レポートの内容や形式をカスタマイズしたい場合：

1. `_generate_relocation_summary()` 関数を編集して分析項目を追加
2. レポート生成部分（lines 2750-2820）でフォーマットを調整
3. フロントエンドで `data.data` の構造化データを利用して独自のUIを構築

## 注意事項

- レポート生成に失敗してもリロケーション処理自体は影響を受けません（エラーハンドリング済み）
- 大規模データセット（10,000+ SKU）の場合、上位5件のみを分析して性能を最適化
- SSEストリームは自動的にバッファリングされるため、接続タイミングが遅れても過去のメッセージを受信可能
