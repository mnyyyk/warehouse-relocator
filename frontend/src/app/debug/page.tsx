'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';

// ====== API helpers (fallback: direct → proxy) ======
const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://127.0.0.1:8000';

async function getWithFallback(path: string): Promise<{ json: any; via: 'proxy' | 'direct' }> {
  try {
    const res = await fetch(`${API_BASE}${path}`);
    const json = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(typeof json === 'object' ? JSON.stringify(json) : String(res.statusText));
    return { json, via: 'direct' };
  } catch (_e) {
    const res2 = await fetch(path);
    const json2 = await res2.json().catch(() => ({}));
    if (!res2.ok) throw new Error(typeof json2 === 'object' ? JSON.stringify(json2) : String(res2.statusText));
    return { json: json2, via: 'proxy' };
  }
}

// ====== Utilities ======
// 8桁ロケ DDDCCCDD → (段/列/奥)
const parseLoc8 = (loc: string) => {
  const s = String(loc || '');
  const m = s.match(/^(\d{3})(\d{3})(\d{2})$/);
  if (!m) return null;
  return { level: parseInt(m[1], 10), column: parseInt(m[2], 10), depth: parseInt(m[3], 10) };
};
const distanceOf = (a: string, b: string): number | undefined => {
  const pa = parseLoc8(a);
  const pb = parseLoc8(b);
  if (!pa || !pb) return undefined;
  return Math.abs(pa.level - pb.level) + Math.abs(pa.column - pb.column) + Math.abs(pa.depth - pb.depth);
};

// lot → YYYY-MM-DD（無ければ YYYY-MM-01）推定（バックエンド未付与時のフォールバック）
function deriveLotDate(lot: any): string | undefined {
  const s = String(lot ?? '').trim();
  if (!s) return undefined;
  const tokens = s.match(/\d{6,8}/g);
  if (!tokens) return undefined;
  for (const t of tokens) {
    if (t.length === 8) {
      const y = parseInt(t.slice(0, 4), 10);
      const m = parseInt(t.slice(4, 6), 10);
      const d = parseInt(t.slice(6, 8), 10);
      if (y >= 1990 && y <= 2099 && m >= 1 && m <= 12 && d >= 1 && d <= 31) {
        return `${t.slice(0, 4)}-${t.slice(4, 6)}-${t.slice(6, 8)}`;
      }
    }
  }
  for (const t of tokens) {
    if (t.length === 6) {
      const y = parseInt(t.slice(0, 4), 10);
      const m = parseInt(t.slice(4, 6), 10);
      if (y >= 1990 && y <= 2099 && m >= 1 && m <= 12) {
        return `${t.slice(0, 4)}-${t.slice(4, 6)}-01`;
      }
    }
  }
  return undefined;
}

// ====== Types ======
type AnyRow = Record<string, any>;
type Tab = 'sku' | 'inventory' | 'recv_tx' | 'ship_tx' | 'locations' | 'metrics';

type MetricsFilter = {
  sku_id: string;
  window_days: number; // 90/180/365/0
  sort: string;        // backend whitelist key
  order: 'asc' | 'desc';
};

export default function DebugPage() {
  const [tab, setTab] = useState<Tab>('sku');

  // Shared paging
  const [rows, setRows] = useState<AnyRow[]>([]);
  const [totalRows, setTotalRows] = useState(0);
  const [page, setPage] = useState(0);
  const [limit, setLimit] = useState(20);
  const [loading, setLoading] = useState(false);

  // Filters (tab-specific)
  const [skuQ, setSkuQ] = useState('');
  const [invFilter, setInvFilter] = useState({ block: 'B', sku_id: '', location_like: '', lot_like: '' });
  const [recvFilter, setRecvFilter] = useState({ sku_id: '', start: '', end: '' });
  const [shipFilter, setShipFilter] = useState({ sku_id: '', start: '', end: '' });
  const [locFilter, setLocFilter] = useState({ block: 'B', quality: '' });
  const [metricsFilter, setMetricsFilter] = useState<MetricsFilter>({
    sku_id: '',
    window_days: 90,
    sort: 'cases_per_day',
    order: 'desc',
  });

  const fetchDebug = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      params.set('limit', String(limit));
      params.set('offset', String(page * limit));

      let url = '';
      if (tab === 'sku') {
        url = '/v1/debug/sku';
        if (skuQ) params.set('q', skuQ);
      } else if (tab === 'inventory') {
        url = '/v1/debug/inventory';
        if (invFilter.block) invFilter.block.split(',').map(s => s.trim()).filter(Boolean).forEach(b => params.append('block', b));
        if (invFilter.sku_id) params.set('sku_id', invFilter.sku_id);
        if (invFilter.location_like) params.set('location_like', invFilter.location_like);
        if (invFilter.lot_like) params.set('lot_like', invFilter.lot_like);
      } else if (tab === 'recv_tx') {
        url = '/v1/debug/recv_tx';
        if (recvFilter.sku_id) params.set('sku_id', recvFilter.sku_id);
        if (recvFilter.start) params.set('start', recvFilter.start);
        if (recvFilter.end) params.set('end', recvFilter.end);
      } else if (tab === 'ship_tx') {
        url = '/v1/debug/ship_tx';
        if (shipFilter.sku_id) params.set('sku_id', shipFilter.sku_id);
        if (shipFilter.start) params.set('start', shipFilter.start);
        if (shipFilter.end) params.set('end', shipFilter.end);
      } else if (tab === 'metrics') {
        url = '/v1/debug/sku_metrics';
        // Backend expects 90/180/365/99999. UIの「無期限=0」は 99999 にマッピングする。
        const wd = Number(metricsFilter.window_days) === 0 ? 99999 : Number(metricsFilter.window_days);
        params.set('window_days', String(wd));
        if (metricsFilter.sku_id) params.set('sku_id', metricsFilter.sku_id);
        if (metricsFilter.sort) params.set('sort', metricsFilter.sort);
        if (metricsFilter.order) params.set('order', metricsFilter.order);
      } else if (tab === 'locations') {
        url = '/v1/debug/locations';
        if (locFilter.block) locFilter.block.split(',').map(s => s.trim()).filter(Boolean).forEach(b => params.append('block', b));
        if (locFilter.quality) locFilter.quality.split(',').map(s => s.trim()).filter(Boolean).forEach(q => params.append('quality', q));
      }

      const fullPath = `${url}?${params.toString()}`;
      const { json } = await getWithFallback(fullPath);

      let rws: AnyRow[] = Array.isArray(json?.rows) ? json.rows : [];
      if (tab === 'inventory') {
        rws = rws.map((r: AnyRow) => {
          const lotVal = r.lot ?? r['ロット'] ?? r['lot_no'] ?? r['lot_number'] ?? r['製造ロット'] ?? '';
          const ld = r.lot_date ?? deriveLotDate(lotVal);
          // 利用可能数（在庫 − 引当）をフロント側で補助計算
          const qty = Number(r?.qty ?? 0);
          const allocated = Number(r?.allocated_qty ?? 0);
          const available_qty = Number.isFinite(qty - allocated) ? qty - allocated : undefined;
          return { ...r, lot_date: ld ?? '', available_qty };
        });
      }
      setRows(rws);
      setTotalRows(typeof json?.total === 'number' ? json.total : 0);
    } catch (_e) {
      setRows([]);
      setTotalRows(0);
    } finally {
      setLoading(false);
    }
  }, [tab, limit, page, skuQ, invFilter, recvFilter, shipFilter, locFilter, metricsFilter]);

  useEffect(() => { fetchDebug(); }, [tab, page, limit, fetchDebug]);

  // 表示カラム（在庫タブは lot_date を優先、column/level/depth は末尾）
  const displayCols = useMemo(() => {
    if (!rows.length) return [] as string[];
    if (tab === 'inventory') {
      const pref = ['sku_id','lot','lot_date','location_id','pack_qty','qty','available_qty','allocated_qty','block_code','quality_name','cases','column','level','depth'];
      const present = pref.filter((k) => rows.some((r) => r[k] !== undefined));
      const others = Object.keys(rows[0]).filter((k) => !present.includes(k));
      // allocated_qty/available_qty を確実に表示できるよう、上限を拡張
      return [...present, ...others].slice(0, 14);
    }
    if (tab === 'locations') {
      const pref = ['block_code','quality_name','total_slots','can_receive','cannot_receive','highness','used_slots','usage_rate'];
      const present = pref.filter((k) => rows.some((r) => r[k] !== undefined));
      const others = Object.keys(rows[0]).filter((k) => !present.includes(k));
      return [...present, ...others].slice(0, 12);
    }
    if (tab === 'metrics') {
      const pref = [
        'sku_id','window_days',
        'cases_per_day','hits_per_day','cube_per_day',
        'recv_cases_per_day','recv_hits_per_day','recv_cube_per_day',
        'shipped_cases_all','current_cases','turnover_rate',
        'updated_at'
      ];
      const present = pref.filter((k) => rows.some((r) => r[k] !== undefined));
      const others = Object.keys(rows[0]).filter((k) => !present.includes(k));
      return [...present, ...others].slice(0, 12);
    }
    return Object.keys(rows[0]).slice(0, 12);
  }, [rows, tab]);

  const handleSearch = useCallback(() => {
    setPage(0);
    fetchDebug();
  }, [fetchDebug]);

  const exportDebugCsv = useCallback(() => {
    if (!rows.length) return;
    const headers = (displayCols.length ? displayCols : Array.from(new Set(rows.flatMap(r => Object.keys(r))))).slice(0, 20);
    const lines = [
      headers.join(','),
      ...rows.map(r => headers.map(h => '"' + String(r?.[h] ?? '').replace(/"/g, '""') + '"').join(',')),
    ];
    const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `debug_${tab}_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [rows, tab, displayCols]);

  return (
    <main role="main" className="space-y-8">
      <header className="flex items-end justify-between">
        <h1 className="text-2xl md:text-3xl font-semibold tracking-tight">DBビューア（閲覧）</h1>
        <span className="text-xs text-gray-500">/debug</span>
      </header>

      <section className="bg-white rounded-2xl shadow-sm ring-1 ring-black/5 p-6">
        {/* Tabs */}
        <div className="flex flex-wrap gap-2 mb-4">
          {(['sku', 'inventory', 'recv_tx', 'ship_tx', 'locations', 'metrics'] as const).map((k) => (
            <button
              key={k}
              onClick={() => { setTab(k); setPage(0); }}
              className={`px-3 py-2 rounded-lg text-sm border transition ${
                tab === k ? 'bg-black text-white border-black' : 'bg-white text-gray-900 border-black/10 hover:bg-gray-50'
              }`}
            >
              {k === 'sku' ? 'SKU'
                : k === 'inventory' ? '在庫'
                : k === 'recv_tx' ? '入荷'
                : k === 'ship_tx' ? '出荷'
                : k === 'locations' ? 'ロケ状態'
                : '指標（速度）'}
            </button>
          ))}
        </div>

        {/* Filters */}
        <div className="flex flex-wrap items-end gap-3 mb-3">
          {tab === 'sku' && (
            <label className="text-sm">
              SKU 検索：
              <input
                className="ml-2 rounded-lg border border-black/10 bg-white px-3 py-2 text-sm"
                value={skuQ}
                onChange={(e) => setSkuQ(e.target.value)}
                placeholder="部分一致"
              />
            </label>
          )}

          {tab === 'inventory' && (
            <>
              <label className="text-sm">
                ブロック（カンマ区切り）：
                <input
                  className="ml-2 rounded-lg border border-black/10 bg-white px-3 py-2 text-sm w-44"
                  value={invFilter.block}
                  onChange={(e) => setInvFilter((p) => ({ ...p, block: e.target.value }))}
                  placeholder="B"
                />
              </label>
              <label className="text-sm">
                SKU：
                <input
                  className="ml-2 rounded-lg border border-black/10 bg-white px-3 py-2 text-sm w-44"
                  value={invFilter.sku_id}
                  onChange={(e) => setInvFilter((p) => ({ ...p, sku_id: e.target.value }))}
                />
              </label>
              <label className="text-sm">
                ロケ（部分）：
                <input
                  className="ml-2 rounded-lg border border-black/10 bg-white px-3 py-2 text-sm w-36"
                  value={invFilter.location_like}
                  onChange={(e) => setInvFilter((p) => ({ ...p, location_like: e.target.value }))}
                />
              </label>
              <label className="text-sm">
                ロット（部分）：
                <input
                  className="ml-2 rounded-lg border border-black/10 bg-white px-3 py-2 text-sm w-36"
                  value={invFilter.lot_like}
                  onChange={(e) => setInvFilter((p) => ({ ...p, lot_like: e.target.value }))}
                />
              </label>
            </>
          )}

          {tab === 'recv_tx' && (
            <>
              <label className="text-sm">
                SKU：
                <input
                  className="ml-2 rounded-lg border border-black/10 bg-white px-3 py-2 text-sm w-44"
                  value={recvFilter.sku_id}
                  onChange={(e) => setRecvFilter((p) => ({ ...p, sku_id: e.target.value }))}
                />
              </label>
              <label className="text-sm">
                開始日：
                <input
                  type="date"
                  className="ml-2 rounded-lg border border-black/10 bg-white px-3 py-2 text-sm"
                  value={recvFilter.start}
                  onChange={(e) => setRecvFilter((p) => ({ ...p, start: e.target.value }))}
                />
              </label>
              <label className="text-sm">
                終了日：
                <input
                  type="date"
                  className="ml-2 rounded-lg border border-black/10 bg-white px-3 py-2 text-sm"
                  value={recvFilter.end}
                  onChange={(e) => setRecvFilter((p) => ({ ...p, end: e.target.value }))}
                />
              </label>
            </>
          )}

          {tab === 'ship_tx' && (
            <>
              <label className="text-sm">
                SKU：
                <input
                  className="ml-2 rounded-lg border border-black/10 bg-white px-3 py-2 text-sm w-44"
                  value={shipFilter.sku_id}
                  onChange={(e) => setShipFilter((p) => ({ ...p, sku_id: e.target.value }))}
                />
              </label>
              <label className="text-sm">
                開始日：
                <input
                  type="date"
                  className="ml-2 rounded-lg border border-black/10 bg-white px-3 py-2 text-sm"
                  value={shipFilter.start}
                  onChange={(e) => setShipFilter((p) => ({ ...p, start: e.target.value }))}
                />
              </label>
              <label className="text-sm">
                終了日：
                <input
                  type="date"
                  className="ml-2 rounded-lg border border-black/10 bg-white px-3 py-2 text-sm"
                  value={shipFilter.end}
                  onChange={(e) => setShipFilter((p) => ({ ...p, end: e.target.value }))}
                />
              </label>
            </>
          )}

          {tab === 'metrics' && (
            <>
              <label className="text-sm">
                回転期間：
                <select
                  className="ml-2 rounded-lg border border-black/10 bg-white px-3 py-2 text-sm w-36"
                  value={String(metricsFilter.window_days)}
                  onChange={(e) => {
                    const v = parseInt(e.target.value, 10);
                    setMetricsFilter((p) => ({ ...p, window_days: Number.isFinite(v) ? v : 90 }));
                  }}
                  title="0 を選ぶと全期間（無期限）を表示"
                >
                  <option value="90">90日</option>
                  <option value="180">180日</option>
                  <option value="365">365日</option>
                  <option value="0">無期限（全期間）</option>
                </select>
              </label>
              <label className="text-sm">
                SKU：
                <input
                  className="ml-2 rounded-lg border border-black/10 bg-white px-3 py-2 text-sm w-44"
                  value={metricsFilter.sku_id}
                  onChange={(e) => setMetricsFilter((p) => ({ ...p, sku_id: e.target.value }))}
                  placeholder="部分一致（例: A1653）"
                />
              </label>
              <label className="text-sm">
                ソート：
                <select
                  className="ml-2 rounded-lg border border-black/10 bg-white px-3 py-2 text-sm w-44"
                  value={metricsFilter.sort}
                  onChange={(e) => setMetricsFilter((p) => ({ ...p, sort: e.target.value }))}
                  title="並べ替え対象の指標"
                >
                  <option value="cases_per_day">出荷: cases_per_day</option>
                  <option value="hits_per_day">出荷: hits_per_day</option>
                  <option value="cube_per_day">出荷: cube_per_day</option>
                  <option value="recv_cases_per_day">入荷: recv_cases_per_day</option>
                  <option value="recv_hits_per_day">入荷: recv_hits_per_day</option>
                  <option value="recv_cube_per_day">入荷: recv_cube_per_day</option>
                  <option value="shipped_cases_all">累計出荷ケース</option>
                  <option value="current_cases">現在庫（ケース）</option>
                  <option value="turnover_rate">回転率（参考）</option>
                  <option value="updated_at">更新日時</option>
                  <option value="sku_id">SKU</option>
                </select>
              </label>
              <label className="text-sm">
                順序：
                <select
                  className="ml-2 rounded-lg border border-black/10 bg-white px-3 py-2 text-sm w-24"
                  value={metricsFilter.order}
                  onChange={(e) => setMetricsFilter((p) => ({ ...p, order: (e.target.value as 'asc' | 'desc') }))}
                >
                  <option value="desc">降順</option>
                  <option value="asc">昇順</option>
                </select>
              </label>
            </>
          )}

          {tab === 'locations' && (
            <>
              <label className="text-sm">
                ブロック（カンマ区切り）：
                <input
                  className="ml-2 rounded-lg border border-black/10 bg-white px-3 py-2 text-sm w-44"
                  value={locFilter.block}
                  onChange={(e) => setLocFilter((p) => ({ ...p, block: e.target.value }))}
                  placeholder="B"
                />
              </label>
              <label className="text-sm">
                品質区分（カンマ区切り）：
                <input
                  className="ml-2 rounded-lg border border-black/10 bg-white px-3 py-2 text-sm w-44"
                  value={locFilter.quality}
                  onChange={(e) => setLocFilter((p) => ({ ...p, quality: e.target.value }))}
                  placeholder="良品,保留,不良"
                />
              </label>
            </>
          )}

          <label className="text-sm">
            表示件数：
            <select
              className="ml-2 rounded-lg border border-black/10 bg-white px-3 py-2 text-sm"
              value={limit}
              onChange={(e) => {
                setLimit(parseInt(e.target.value, 10));
                setPage(0);
              }}
            >
              <option value={20}>20</option>
              <option value={50}>50</option>
              <option value={100}>100</option>
            </select>
          </label>

          <button
            onClick={handleSearch}
            disabled={loading}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition active:scale-[.99] ${
              loading ? 'bg-gray-300 text-gray-600 cursor-not-allowed' : 'bg-black text-white hover:bg-black/90 focus:outline-none focus:ring-2 focus:ring-black/20'
            }`}
          >
            {loading ? '検索中…' : '検索'}
          </button>

          <button
            onClick={exportDebugCsv}
            disabled={!rows.length}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition border ${
              !rows.length
                ? 'bg-gray-100 text-gray-400 border-black/10 cursor-not-allowed'
                : 'bg-white text-gray-900 border-black/10 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-black/20'
            }`}
          >
            CSVエクスポート（このページ）
          </button>
        </div>

        {/* Result + table */}
        <div className="mt-3 bg-white rounded-xl ring-1 ring-black/5">
          <div className="px-4 py-3 text-sm text-gray-700">
            {loading ? '読込中…' : totalRows ? `${page * limit + 1}–${page * limit + rows.length} / ${totalRows} 件` : '0 件'}
          </div>

          {rows.length > 0 && (
            <div className="overflow-x-auto">
              <table className="min-w-full text-xs divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr className="text-left">
                    {displayCols.map((k) => (
                      <th key={k} className="py-2 px-3">{k}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {rows.map((r, i) => (
                    <tr key={i} className="hover:bg-gray-50">
                      {displayCols.map((k) => (
                        <td key={k} className="py-2 px-3">{String(r?.[k] ?? '')}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <div className="flex items-center gap-2 px-4 py-3">
            <button
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={loading || page === 0}
              className={`px-3 py-2 rounded-lg text-sm border ${
                page === 0
                  ? 'bg-gray-100 text-gray-400 border-black/10 cursor-not-allowed'
                  : 'bg-white text-gray-900 border-black/10 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-black/20'
              }`}
            >
              前へ
            </button>
            <button
              onClick={() => {
                const maxPage = Math.max(0, Math.ceil(totalRows / limit) - 1);
                setPage((p) => Math.min(maxPage, p + 1));
              }}
              disabled={loading || (page + 1) * limit >= totalRows}
              className={`px-3 py-2 rounded-lg text-sm border ${
                (page + 1) * limit >= totalRows
                  ? 'bg-gray-100 text-gray-400 border-black/10 cursor-not-allowed'
                  : 'bg-white text-gray-900 border-black/10 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-black/20'
              }`}
            >
              次へ
            </button>
          </div>
        </div>
      </section>
    </main>
  );
}
