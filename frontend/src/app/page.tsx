'use client';

import React, { useCallback, useMemo, useState, useEffect } from 'react';
import FileUploader from '@/components/FileUploader';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://127.0.0.1:8000';

async function postWithFallback(path: string, payload: any): Promise<{ json: any; via: 'proxy' | 'direct' }>{
  const body = JSON.stringify(payload);
  // 1) Try direct to backend first to avoid Next dev-proxy noise (ECONNRESET on reload)
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
    });
    const json = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(typeof json === 'object' ? JSON.stringify(json) : String(res.statusText));
    return { json, via: 'direct' };
  } catch (_e) {
    // 2) Fallback to relative path (through Next proxy/rewrites if any)
    const res2 = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
    });
    const json2 = await res2.json().catch(() => ({}));
    if (!res2.ok) throw new Error(typeof json2 === 'object' ? JSON.stringify(json2) : String(res2.statusText));
    return { json: json2, via: 'proxy' };
  }
}

async function getWithFallback(path: string): Promise<{ json: any; via: 'proxy' | 'direct' }>{
  // 1) Try direct first
  try {
    const res = await fetch(`${API_BASE}${path}`);
    const json = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(typeof json === 'object' ? JSON.stringify(json) : String(res.statusText));
    return { json, via: 'direct' };
  } catch (_e) {
    // 2) Fallback to relative (proxy)
    const res2 = await fetch(path);
    const json2 = await res2.json().catch(() => ({}));
    if (!res2.ok) throw new Error(typeof json2 === 'object' ? JSON.stringify(json2) : String(res2.statusText));
    return { json: json2, via: 'proxy' };
  }
}

// 8桁ロケ DDDCCCDD を (段,列,奥) に分解 & 距離計算（Manhattan）
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

// lot → 日付（YYYY-MM-DD or YYYY-MM-01）を推定（バックエンド未付与時のフォールバック）
function deriveLotDate(lot: any): string | undefined {
  const s = String(lot ?? '').trim();
  if (!s) return undefined;
  const tokens = s.match(/\d{6,8}/g);
  if (!tokens) return undefined;
  // 8桁(YYYYMMDD)を優先
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
  // 次に6桁(YYYYMM)を採用（日付は01で固定）
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

type AnalysisResp = {
  updated: number;
  window_days: number;
  blocks: string[];
  elapsed_sec?: number;
  total_skus?: number;
  run_at?: string;
};

type Move = {
  sku_id: string;
  lot?: string;
  lot_date?: string;
  qty: number;
  from_loc: string;
  to_loc: string;
  distance?: number;
};

export default function Home() {
  // ---- Analysis controls ----
  const [blockText, setBlockText] = useState('B');
  const blocks = useMemo(
    () => blockText.split(',').map((s) => s.trim()).filter(Boolean),
    [blockText]
  );
  const [qualityChoices, setQualityChoices] = useState<string[]>([]);
  const [selectedQualities, setSelectedQualities] = useState<string[]>([]);
  const qualities = selectedQualities;
  const loadQualityChoices = useCallback(async () => {
    const limit = 500;
    let offset = 0;
    const setVals = new Set<string>();
    try {
      while (true) {
        const { json } = await getWithFallback(`/v1/debug/inventory?limit=${limit}&offset=${offset}`);
        const rows = Array.isArray(json?.rows) ? json.rows : [];
        rows.forEach((r: any) => {
          const v = (r.quality_name ?? r['品質区分名'] ?? '').toString().trim();
          if (v) setVals.add(v);
        });
        const total = typeof json?.total === 'number' ? json.total : 0;
        offset += rows.length;
        if (!rows.length || offset >= total || offset >= 10000) break; // safety cap
      }
      setQualityChoices(Array.from(setVals).sort());
    } catch {
      setQualityChoices([]);
    }
  }, []);
  useEffect(() => { loadQualityChoices(); }, [loadQualityChoices]);
  const [windowDays, setWindowDays] = useState(90);
  const [status, setStatus] = useState('');
  const [resp, setResp] = useState<AnalysisResp | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [relocating, setRelocating] = useState(false);

  const startAnalysis = useCallback(async () => {
    setAnalyzing(true);
    setStatus('解析を実行中...');
    setResp(null);
    try {
      const payload: any = { window_days: windowDays, block_codes: blocks };
      if (qualities.length) payload.quality_names = qualities;
      const { json, via } = await postWithFallback('/v1/upload/analysis/start', payload);
      setStatus(`✔ 解析完了 (${via})`);
      setResp(json);
    } catch (e: any) {
      setStatus(`✖ 解析に失敗: ${e?.message || e}`);
    } finally {
      setAnalyzing(false);
    }
  }, [blocks, windowDays, selectedQualities]);

  // ---- Relocation (optimizer) ----
  const [maxMovesInput, setMaxMovesInput] = useState<string>(''); // 空欄OK
  const maxMoves = useMemo(() => {
    const n = parseInt(maxMovesInput, 10);
    return Number.isFinite(n) && n > 0 ? n : undefined; // 0/無効/空欄 → 未指定
  }, [maxMovesInput]);
  const [fillRate, setFillRate] = useState<number>(0.95);
  const [useAiMain, setUseAiMain] = useState<boolean>(true);
  // 連鎖移動の制御パラメータ（空欄 or 0 で無効）
  const [chainDepthInput, setChainDepthInput] = useState<string>('');
  const chainDepth = useMemo(() => {
    const n = parseInt(chainDepthInput, 10);
    return Number.isFinite(n) && n > 0 ? n : undefined;
  }, [chainDepthInput]);

  const [evictionBudgetInput, setEvictionBudgetInput] = useState<string>('');
  const evictionBudget = useMemo(() => {
    const n = parseInt(evictionBudgetInput, 10);
    return Number.isFinite(n) && n > 0 ? n : undefined;
  }, [evictionBudgetInput]);

  const [touchBudgetInput, setTouchBudgetInput] = useState<string>('');
  const touchBudget = useMemo(() => {
    const n = parseInt(touchBudgetInput, 10);
    return Number.isFinite(n) && n > 0 ? n : undefined;
  }, [touchBudgetInput]);
  const [reloStatus, setReloStatus] = useState<string>('');
  const [moves, setMoves] = useState<Move[]>([]);
  const [reloMeta, setReloMeta] = useState<any>(null);

  const startRelocation = useCallback(async () => {
    setRelocating(true);
    setReloStatus('最適化を実行中...');
    setMoves([]);
    try {
      const payload: any = {
        block_codes: blocks,
        fill_rate: fillRate,
        use_ai_main: useAiMain,
        rotation_window_days: windowDays,
      };
      if (typeof maxMoves === 'number') payload.max_moves = maxMoves; // 未指定なら送らない＝無制限
      if (qualities.length) payload.quality_names = qualities;
      if (typeof chainDepth === 'number') payload.chain_depth = chainDepth;
      if (typeof evictionBudget === 'number') payload.eviction_budget = evictionBudget;
      if (typeof touchBudget === 'number') payload.touch_budget = touchBudget;
      const { json, via } = await postWithFallback('/v1/upload/relocation/start', payload);

      const meta = json && typeof json === 'object' ? {
        use_ai: json.use_ai,
        use_ai_main: json.use_ai_main,
        count: json.count,
        blocks: json.blocks,
        quality_names: json.quality_names,
        max_moves: json.max_moves,
        fill_rate: json.fill_rate,
        ai_hints_skus: json.ai_hints_skus,
        violations_summary: json.violations_summary,
        trace_id: json.trace_id, // ← add trace id from backend
      } : null;
      setReloMeta(meta);

      const mv: Move[] = Array.isArray(json)
        ? json
        : Array.isArray(json?.moves)
        ? json.moves
        : Array.isArray(json?.data)
        ? json.data
        : [];
      const mvWithDist: Move[] = mv.map((m) => ({
        ...m,
        distance: m.distance ?? distanceOf(m.from_loc, m.to_loc),
        lot_date: m.lot_date ?? deriveLotDate(m.lot),
      }));
      setMoves(mvWithDist);
      setReloStatus(`✔ 最適化完了（${mvWithDist.length}件, ${via}）`);
    } catch (e: any) {
      setReloStatus(`✖ 最適化に失敗: ${e?.message || e}`);
    } finally {
      setRelocating(false);
    }
  }, [blocks, maxMoves, fillRate, selectedQualities, useAiMain, windowDays, chainDepth, evictionBudget, touchBudget]);

  const exportCsv = useCallback(() => {
    if (!moves.length) return;
    const header = ['sku_id', 'lot', 'lot_date', 'qty', 'from_loc', 'to_loc', 'distance'];
    const lines = [
      header.join(','),
      ...moves.map((m) =>
        [
          m.sku_id ?? '',
          m.lot ?? '',
          m.lot_date ?? '',
          String(m.qty ?? ''),
          m.from_loc ?? '',
          m.to_loc ?? '',
          (m.distance ?? '').toString(),
        ]
          .map((v) => '"' + String(v).replace(/"/g, '""') + '"')
          .join(',')
      ),
    ];
    const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `relocation_moves_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [moves]);

  // ---- Debug Viewer (read-only) ----
  type AnyRow = Record<string, any>;
  const [tab, setTab] = useState<'sku' | 'inventory' | 'recv_tx' | 'ship_tx'>('sku');
  const [rows, setRows] = useState<AnyRow[]>([]);
  const [totalRows, setTotalRows] = useState(0);
  const [page, setPage] = useState(0);
  const [limit, setLimit] = useState(20);
  const [loadingDebug, setLoadingDebug] = useState(false);
  const [skuQ, setSkuQ] = useState('');
  const [invFilter, setInvFilter] = useState({ block: '', sku_id: '', location_like: '', lot_like: '' });
  const [recvFilter, setRecvFilter] = useState({ sku_id: '', start: '', end: '' });
  const [shipFilter, setShipFilter] = useState({ sku_id: '', start: '', end: '' });

  const fetchDebug = useCallback(async () => {
    setLoadingDebug(true);
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
      } else {
        url = '/v1/debug/ship_tx';
        if (shipFilter.sku_id) params.set('sku_id', shipFilter.sku_id);
        if (shipFilter.start) params.set('start', shipFilter.start);
        if (shipFilter.end) params.set('end', shipFilter.end);
      }
      const fullPath = `${url}?${params.toString()}`;
      const { json } = await getWithFallback(fullPath);
      let rws: AnyRow[] = Array.isArray(json?.rows) ? json.rows : [];
      if (tab === 'inventory') {
        rws = rws.map((r: AnyRow) => {
          const lotVal = r.lot ?? r['ロット'] ?? r['lot_no'] ?? r['lot_number'] ?? r['製造ロット'] ?? '';
          const ld = r.lot_date ?? deriveLotDate(lotVal);
          return { ...r, lot_date: ld ?? '' };
        });
      }
      setRows(rws);
      setTotalRows(typeof json?.total === 'number' ? json.total : 0);
    } catch (_e) {
      setRows([]);
      setTotalRows(0);
    } finally {
      setLoadingDebug(false);
    }
  }, [tab, limit, page, skuQ, invFilter, recvFilter, shipFilter]);
  // DBビューア表示順を固定（在庫タブで lot_date を優先表示し、column/level/depth は末尾へ）
  const displayCols = useMemo(() => {
    if (!rows.length) return [] as string[];
    if (tab === 'inventory') {
      const pref = ['sku_id','lot','lot_date','location_id','pack_qty','qty','block_code','quality_name','cases','column','level','depth'];
      const present = pref.filter((k) => rows.some((r) => r[k] !== undefined));
      const others = Object.keys(rows[0]).filter((k) => !present.includes(k));
      return [...present, ...others].slice(0, 12);
    }
    return Object.keys(rows[0]).slice(0, 12);
  }, [rows, tab]);

  useEffect(() => {
    fetchDebug();
  }, [tab, page, limit, fetchDebug]);

  const handleSearch = useCallback(() => {
    setPage(0);
    fetchDebug();
  }, [fetchDebug]);

  const exportDebugCsv = useCallback(() => {
    if (!rows.length) return;
    const headers = (displayCols.length ? displayCols : Array.from(new Set(rows.flatMap(r => Object.keys(r))))).slice(0, 20);
    const lines = [
      headers.join(','),
      ...rows.map(r => headers.map(h => '"' + String(r?.[h] ?? '').replace(/\"/g, '""') + '"').join(',')),
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
    <div className="max-w-3xl mx-auto mt-12 space-y-10 px-4">
      <h1 className="text-2xl font-bold">データ取込 & 分析（MVP）</h1>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold">1) ファイルアップロード</h2>
        <div className="grid grid-cols-1 gap-4">
          <div>
            <div className="text-sm font-medium mb-1">SKUマスター</div>
            <FileUploader endpoint="/v1/upload/sku" />
          </div>
          <div>
            <div className="text-sm font-medium mb-1">入荷（発注のみ集計）</div>
            <FileUploader endpoint="/v1/upload/recv_tx" />
          </div>
          <div>
            <div className="text-sm font-medium mb-1">出荷</div>
            <FileUploader endpoint="/v1/upload/ship_tx" />
          </div>
          <div>
            <div className="text-sm font-medium mb-1">在庫</div>
            <FileUploader endpoint="/v1/upload/inventory" />
          </div>
        </div>
        <p className="text-xs text-gray-500">※ 在庫の「ロケーション」は8桁（DDDCCCDD）。ブロック略称で対象を絞り込みます。</p>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold">2) 分析実行（ブロック指定）</h2>
        <div className="flex flex-wrap items-center gap-3">
          <label className="text-sm">
            ブロック（カンマ区切り）：
            <input
              className="ml-2 border rounded px-2 py-1 text-sm"
              value={blockText}
              onChange={(e) => setBlockText(e.target.value)}
              placeholder="B"
            />
          </label>
          <div className="flex items-center gap-2">
            <label className="text-sm">
              品質区分（複数選択可）：
              <select
                multiple
                className="ml-2 border rounded px-2 py-1 text-sm w-56"
                value={selectedQualities}
                onChange={(e) => {
                  const opts = Array.from((e.target as HTMLSelectElement).selectedOptions);
                  setSelectedQualities(opts.map(o => o.value));
                }}
                size={Math.min(6, Math.max(3, qualityChoices.length || 3))}
                title="未選択なら全て対象"
              >
                {qualityChoices.map((q) => (
                  <option key={q} value={q}>{q}</option>
                ))}
              </select>
            </label>
            <button
              type="button"
              onClick={() => setSelectedQualities(qualityChoices)}
              className="px-2 py-1 rounded text-xs border bg-white hover:bg-gray-50"
              title="全てを選択"
            >
              全選択
            </button>
            <button
              type="button"
              onClick={() => setSelectedQualities([])}
              className="px-2 py-1 rounded text-xs border bg-white hover:bg-gray-50"
              title="すべて解除"
            >
              解除
            </button>
            <button
              type="button"
              onClick={loadQualityChoices}
              className="px-2 py-1 rounded text-xs border bg-white hover:bg-gray-50"
              title="DBから最新の品質区分を再取得"
            >
              再読込
            </button>
          </div>
          <label className="text-sm">
            回転期間：
            <select
              className="ml-2 border rounded px-2 py-1 text-sm w-36"
              value={String(windowDays)}
              onChange={(e) => {
                const v = parseInt(e.target.value, 10);
                setWindowDays(Number.isFinite(v) ? v : 90);
              }}
              title="0 を選ぶと全期間（無期限）で計算します"
            >
              <option value="90">90日</option>
              <option value="180">180日</option>
              <option value="365">365日</option>
              <option value="0">無期限（全期間）</option>
            </select>
          </label>
          <button
            onClick={startAnalysis}
            disabled={analyzing}
            className={`px-3 py-1.5 rounded text-sm ${analyzing ? 'bg-gray-300 text-gray-600 cursor-not-allowed' : 'bg-black text-white hover:opacity-90'}`}
          >
            {analyzing ? '解析中…' : '解析を開始'}
          </button>
        </div>
        <div className="border rounded p-3 bg-gray-50 mt-2">
          <div className="text-sm mb-2">{status || '待機中…'}</div>
          {resp && (
            <pre className="text-xs whitespace-pre-wrap break-words">{JSON.stringify(resp, null, 2)}</pre>
          )}
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold">3) リロケーション（最適化）</h2>
        <div className="flex flex-wrap items-center gap-3">
          <label className="text-sm">
            最大移動行数：
            <input
              type="number"
              className="ml-2 border rounded px-2 py-1 text-sm w-28"
              min={0}
              value={maxMovesInput}
              onChange={(e) => setMaxMovesInput(e.target.value)}
              placeholder="空欄=無制限"
              title="0 または空欄で無制限（バックエンド既定）"
            />
          </label>
          <label className="text-sm">
            充填率上限（0.1–1.0）：
            <input
              type="number"
              step={0.05}
              className="ml-2 border rounded px-2 py-1 text-sm w-28"
              min={0.1}
              max={1.0}
              value={fillRate}
              onChange={(e) => setFillRate(parseFloat(e.target.value || '0.95'))}
            />
          </label>
          <label className="text-sm">
            回転期間：
            <select
              className="ml-2 border rounded px-2 py-1 text-sm w-36"
              value={String(windowDays)}
              onChange={(e) => {
                const v = parseInt(e.target.value, 10);
                setWindowDays(Number.isFinite(v) ? v : 90);
              }}
              title="0 を選ぶと全期間（無期限）で計算します"
            >
              <option value="90">90日</option>
              <option value="180">180日</option>
              <option value="365">365日</option>
              <option value="0">無期限（全期間）</option>
            </select>
          </label>
          <label className="text-sm">
            AIメインを使用：
            <input
              type="checkbox"
              className="ml-2 align-middle"
              checked={useAiMain}
              onChange={(e) => setUseAiMain(e.target.checked)}
            />
          </label>
          <label className="text-sm">
            連鎖の深さ：
            <input
              type="number"
              className="ml-2 border rounded px-2 py-1 text-sm w-24"
              min={0}
              value={chainDepthInput}
              onChange={(e) => setChainDepthInput(e.target.value)}
              placeholder="空欄=無効"
              title="0 または空欄で無効（深さ1=一段退避を許可、2=二段まで）"
            />
          </label>
          <label className="text-sm">
            退避予算：
            <input
              type="number"
              className="ml-2 border rounded px-2 py-1 text-sm w-28"
              min={0}
              value={evictionBudgetInput}
              onChange={(e) => setEvictionBudgetInput(e.target.value)}
              placeholder="空欄=無制限?"
              title="退避（eviction）として許可する最大移動数。0/空欄は未指定。"
            />
          </label>
          <label className="text-sm">
            タッチ上限：
            <input
              type="number"
              className="ml-2 border rounded px-2 py-1 text-sm w-28"
              min={0}
              value={touchBudgetInput}
              onChange={(e) => setTouchBudgetInput(e.target.value)}
              placeholder="空欄=未指定"
              title="連鎖で触って良いユニークなロケ数の上限。0/空欄は未指定。"
            />
          </label>
          <button
            onClick={startRelocation}
            disabled={relocating}
            className={`px-3 py-1.5 rounded text-sm ${relocating ? 'bg-gray-300 text-gray-600 cursor-not-allowed' : 'bg-black text-white hover:opacity-90'}`}
          >
            {relocating ? '最適化中…' : '最適化を実行'}
          </button>
          <button
            onClick={exportCsv}
            disabled={!moves.length || relocating}
            className={`px-3 py-1.5 rounded text-sm border ${(!moves.length || relocating) ? 'bg-gray-100 text-gray-400 cursor-not-allowed' : 'bg-white hover:bg-gray-50'}`}
          >
            CSVエクスポート
          </button>
        </div>
        <div className="border rounded p-3 bg-gray-50 mt-2">
          <div className="text-sm mb-2">{reloStatus || '未実行'}</div>
          {reloMeta?.trace_id && (
            <div className="mb-2 text-xs flex items-center gap-2 bg-yellow-50 border border-yellow-200 rounded px-2 py-1">
              <span className="font-medium">Trace ID:</span>
              <code className="px-1">{reloMeta.trace_id}</code>
              <button
                type="button"
                className="px-2 py-0.5 border rounded bg-white hover:bg-gray-50"
                title="Trace ID をクリップボードにコピー"
                onClick={() => { try { navigator.clipboard.writeText(String(reloMeta.trace_id)); } catch(_) {} }}
              >
                コピー
              </button>
            </div>
          )}
          {reloMeta && (
            <pre className="text-xs whitespace-pre-wrap break-words mb-2 bg-white border rounded p-2">{JSON.stringify(reloMeta, null, 2)}</pre>
          )}
          {moves.length > 0 && (
            <div className="overflow-x-auto">
              <table className="min-w-full text-xs">
                <thead>
                  <tr className="text-left border-b">
                    <th className="py-1 pr-3">#</th>
                    <th className="py-1 pr-3">SKU</th>
                    <th className="py-1 pr-3">Lot</th>
                    <th className="py-1 pr-3">Lot日付</th>
                    <th className="py-1 pr-3">Qty(ケース)</th>
                    <th className="py-1 pr-3">From</th>
                    <th className="py-1 pr-3">To</th>
                    <th className="py-1 pr-3">距離</th>
                  </tr>
                </thead>
                <tbody>
                  {moves.slice(0, 500).map((m, i) => (
                    <tr key={i} className="border-b last:border-0">
                      <td className="py-1 pr-3">{i + 1}</td>
                      <td className="py-1 pr-3">{m.sku_id}</td>
                      <td className="py-1 pr-3">{m.lot || ''}</td>
                      <td className="py-1 pr-3">{m.lot_date || ''}</td>
                      <td className="py-1 pr-3">{m.qty}</td>
                      <td className="py-1 pr-3 font-mono">{m.from_loc}</td>
                      <td className="py-1 pr-3 font-mono">{m.to_loc}</td>
                      <td className="py-1 pr-3">{m.distance ?? ''}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {moves.length > 500 && (
                <div className="text-xs text-gray-500 mt-2">※ 表示は先頭500件まで。CSVで全件を出力できます。</div>
              )}
            </div>
          )}
        </div>
      </section>
      <section className="space-y-4">
        <h2 className="text-xl font-semibold">4) DBビューア（閲覧）</h2>
        <div className="flex flex-wrap gap-2">
          {(['sku','inventory','recv_tx','ship_tx'] as const).map(k => (
            <button
              key={k}
              onClick={() => { setTab(k); setPage(0); }}
              className={`px-3 py-1.5 rounded text-sm border ${tab === k ? 'bg-black text-white' : 'bg-white hover:bg-gray-50'}`}
            >
              {k === 'sku' ? 'SKU' : k === 'inventory' ? '在庫' : k === 'recv_tx' ? '入荷' : '出荷'}
            </button>
          ))}
        </div>

        <div className="flex flex-wrap items-end gap-3">
          {tab === 'sku' && (
            <label className="text-sm">
              SKU 検索：
              <input
                className="ml-2 border rounded px-2 py-1 text-sm"
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
                  className="ml-2 border rounded px-2 py-1 text-sm w-44"
                  value={invFilter.block}
                  onChange={(e) => setInvFilter((p) => ({ ...p, block: e.target.value }))}
                  placeholder={blockText}
                />
              </label>
              <label className="text-sm">
                SKU：
                <input
                  className="ml-2 border rounded px-2 py-1 text-sm w-44"
                  value={invFilter.sku_id}
                  onChange={(e) => setInvFilter((p) => ({ ...p, sku_id: e.target.value }))}
                />
              </label>
              <label className="text-sm">
                ロケ（部分）：
                <input
                  className="ml-2 border rounded px-2 py-1 text-sm w-36"
                  value={invFilter.location_like}
                  onChange={(e) => setInvFilter((p) => ({ ...p, location_like: e.target.value }))}
                />
              </label>
              <label className="text-sm">
                ロット（部分）：
                <input
                  className="ml-2 border rounded px-2 py-1 text-sm w-36"
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
                  className="ml-2 border rounded px-2 py-1 text-sm w-44"
                  value={recvFilter.sku_id}
                  onChange={(e) => setRecvFilter((p) => ({ ...p, sku_id: e.target.value }))}
                />
              </label>
              <label className="text-sm">
                開始日：
                <input
                  type="date"
                  className="ml-2 border rounded px-2 py-1 text-sm"
                  value={recvFilter.start}
                  onChange={(e) => setRecvFilter((p) => ({ ...p, start: e.target.value }))}
                />
              </label>
              <label className="text-sm">
                終了日：
                <input
                  type="date"
                  className="ml-2 border rounded px-2 py-1 text-sm"
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
                  className="ml-2 border rounded px-2 py-1 text-sm w-44"
                  value={shipFilter.sku_id}
                  onChange={(e) => setShipFilter((p) => ({ ...p, sku_id: e.target.value }))}
                />
              </label>
              <label className="text-sm">
                開始日：
                <input
                  type="date"
                  className="ml-2 border rounded px-2 py-1 text-sm"
                  value={shipFilter.start}
                  onChange={(e) => setShipFilter((p) => ({ ...p, start: e.target.value }))}
                />
              </label>
              <label className="text-sm">
                終了日：
                <input
                  type="date"
                  className="ml-2 border rounded px-2 py-1 text-sm"
                  value={shipFilter.end}
                  onChange={(e) => setShipFilter((p) => ({ ...p, end: e.target.value }))}
                />
              </label>
            </>
          )}

          <label className="text-sm">
            表示件数：
            <select
              className="ml-2 border rounded px-2 py-1 text-sm"
              value={limit}
              onChange={(e) => { setLimit(parseInt(e.target.value, 10)); setPage(0); }}
            >
              <option value={20}>20</option>
              <option value={50}>50</option>
              <option value={100}>100</option>
            </select>
          </label>

          <button
            onClick={handleSearch}
            disabled={loadingDebug}
            className={`px-3 py-1.5 rounded text-sm ${loadingDebug ? 'bg-gray-300 text-gray-600 cursor-not-allowed' : 'bg-black text-white hover:opacity-90'}`}
          >
            {loadingDebug ? '検索中…' : '検索'}
          </button>

          <button
            onClick={exportDebugCsv}
            disabled={!rows.length}
            className={`px-3 py-1.5 rounded text-sm border ${!rows.length ? 'bg-gray-100 text-gray-400 cursor-not-allowed' : 'bg-white hover:bg-gray-50'}`}
          >
            CSVエクスポート（このページ）
          </button>
        </div>

        <div className="border rounded p-3 bg-gray-50 mt-2">
          <div className="text-sm mb-2">
            {loadingDebug ? '読込中…' : totalRows ? `${page*limit + 1}–${page*limit + rows.length} / ${totalRows} 件` : '0 件'}
          </div>
          {rows.length > 0 && (
            <div className="overflow-x-auto">
              <table className="min-w-full text-xs">
                <thead>
                  <tr className="text-left border-b">
                    {displayCols.map((k) => (
                      <th key={k} className="py-1 pr-3">{k}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r, i) => (
                    <tr key={i} className="border-b last:border-0">
                      {displayCols.map((k) => (
                        <td key={k} className="py-1 pr-3">{String(r?.[k] ?? '')}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <div className="flex items-center gap-2 mt-2">
            <button
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={loadingDebug || page === 0}
              className={`px-3 py-1.5 rounded text-sm border ${page === 0 ? 'bg-gray-100 text-gray-400 cursor-not-allowed' : 'bg-white hover:bg-gray-50'}`}
            >
              前へ
            </button>
            <button
              onClick={() => { const maxPage = Math.max(0, Math.ceil(totalRows / limit) - 1); setPage((p) => Math.min(maxPage, p + 1)); }}
              disabled={loadingDebug || (page + 1) * limit >= totalRows}
              className={`px-3 py-1.5 rounded text-sm border ${ (page + 1) * limit >= totalRows ? 'bg-gray-100 text-gray-400 cursor-not-allowed' : 'bg-white hover:bg-gray-50'}`}
            >
              次へ
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}