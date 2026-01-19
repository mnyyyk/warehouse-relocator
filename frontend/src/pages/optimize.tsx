import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { NextPage } from 'next';

// ===== API helpers =====
const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://127.0.0.1:8000';

async function postWithFallback(path: string, payload: any): Promise<{ json: any; via: 'proxy' | 'direct' }>{
  const body = JSON.stringify(payload);
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

// ===== utilities =====
const parseLoc8 = (loc: string) => {
  const s = String(loc || '');
  const m = s.match(/^(\d{3})(\d{3})(\d{2})$/);
  if (!m) return null;
  return { level: parseInt(m[1], 10), column: parseInt(m[2], 10), depth: parseInt(m[3], 10) };
};
const distanceOf = (a?: string, b?: string): number | undefined => {
  if (!a || !b) return undefined;
  const pa = parseLoc8(a);
  const pb = parseLoc8(b);
  if (!pa || !pb) return undefined;
  return Math.abs(pa.level - pb.level) + Math.abs(pa.column - pb.column) + Math.abs(pa.depth - pb.depth);
};

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

// Expand comma-/space-separated numbers and ranges like "35-41"
function expandNumberRanges(input: string, min = 1, max = 41): number[] {
  const s = String(input || '').trim();
  if (!s) return [];
  const out: number[] = [];
  const push = (n: number) => {
    if (Number.isFinite(n) && n >= min && n <= max && !out.includes(n)) out.push(n);
  };
  s.split(/[\,\s]+/).forEach((tok) => {
    if (!tok) return;
    const m = tok.match(/^(\d+)\s*-\s*(\d+)$/);
    if (m) {
      let a = parseInt(m[1], 10);
      let b = parseInt(m[2], 10);
      if (!Number.isFinite(a) || !Number.isFinite(b)) return;
      if (a > b) [a, b] = [b, a];
      for (let n = a; n <= b; n++) push(n);
    } else {
      const n = parseInt(tok, 10);
      push(n);
    }
  });
  return out.sort((a, b) => a - b);
}

// Split comma/newline-separated keywords, unique & trimmed
function splitCSV(input: string): string[] {
  const s = String(input || '').trim();
  if (!s) return [];
  const seen = new Set<string>();
  s.split(/[\,\n]+/)
    .map((x) => x.trim())
    .filter(Boolean)
    .forEach((v) => {
      if (!seen.has(v)) seen.add(v);
    });
  return Array.from(seen);
}

// ===== types =====
export type Move = {
  sku_id: string;
  lot?: string;
  lot_date?: string;
  qty: number;
  from_loc: string;
  to_loc: string;
  distance?: number;
};

export type DropSummaryItem = { reason: string; count: number };

export type DropDetail = {
  reason?: string;
  stage?: string;
  rule?: string;
  message?: string;
  sku_id?: string;
  lot?: string;
  from_loc?: string;
  to_loc?: string;
  qty?: number;
  distance?: number;
};

function coerceDropSummary(json: any): DropSummaryItem[] {
  const arr = Array.isArray(json?.summary)
    ? json.summary
    : Array.isArray(json?.rows)
    ? json.rows
    : Array.isArray(json)
    ? json
    : [];
  return arr
    .map((x: any) => ({
      reason: String(x.reason ?? x.key ?? x.rule ?? x.stage ?? '不明'),
      count: Number(x.count ?? x.value ?? x.total ?? x.cnt ?? 0),
    }))
    .filter((i: DropSummaryItem) => Number.isFinite(i.count) && i.count > 0)
    .sort((a: DropSummaryItem, b: DropSummaryItem) => b.count - a.count);
}

function coerceDropDetails(json: any): DropDetail[] {
  const arr = Array.isArray(json?.rows)
    ? json.rows
    : Array.isArray(json?.data)
    ? json.data
    : Array.isArray(json)
    ? json
    : [];
  return arr
    .map((x: any) => ({
      reason: x.reason ?? x.rule ?? x.stage ?? x.message,
      stage: x.stage,
      rule: x.rule,
      message: x.message,
      sku_id: x.sku_id,
      lot: x.lot ?? x.lot_no ?? x.lot_number,
      from_loc: x.from_loc ?? x.from ?? x.src,
      to_loc: x.to_loc ?? x.to ?? x.dst,
      qty: Number(x.qty ?? x.cases ?? x.qty_cases ?? 0),
    }))
    .map((d: DropDetail) => ({
      ...d,
      distance: d.from_loc && d.to_loc ? distanceOf(String(d.from_loc), String(d.to_loc)) : undefined,
    }));
}

const OptimizePage: NextPage & { pageTitle?: string } = () => {
  // ---- Common selectors ----
  const [blockText, setBlockText] = useState('B');
  const blocks = useMemo(() => blockText.split(',').map((s) => s.trim()).filter(Boolean), [blockText]);

  const [qualityChoices, setQualityChoices] = useState<string[]>([]);
  const [selectedQualities, setSelectedQualities] = useState<string[]>([]);
  const qualities = selectedQualities;
  const [didAutoSelectQuality, setDidAutoSelectQuality] = useState<boolean>(false);

  const loadQualityChoices = useCallback(async () => {
    const limit = 5000;
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
      const choices = Array.from(setVals).sort();
      setQualityChoices(choices);
      if (!didAutoSelectQuality && selectedQualities.length === 0 && choices.includes('良品')) {
        setSelectedQualities(['良品']);
        setDidAutoSelectQuality(true);
      }
    } catch {
      setQualityChoices([]);
    }
  }, [didAutoSelectQuality, selectedQualities.length]);

  useEffect(() => { loadQualityChoices(); }, [loadQualityChoices]);

  const [windowDays, setWindowDays] = useState(90);

  // ---- Advanced budgets / depths (optional inputs; send only when provided) ----
  // 連鎖退避をデフォルトで“有効(深さ2)”にし、予算も有効値に設定
  const [chainDepthInput, setChainDepthInput] = useState<string>('2');
  const [evictionBudgetInput, setEvictionBudgetInput] = useState<string>('50');
  const [touchBudgetInput, setTouchBudgetInput] = useState<string>('1000');

  // ---- Advanced (band preferences) ----
  const [advancedOpen, setAdvancedOpen] = useState<boolean>(false);
  const [packLowMaxInput, setPackLowMaxInput] = useState<string>('12');
  const [packHighMinInput, setPackHighMinInput] = useState<string>('50');
  const [bandPrefWeightInput, setBandPrefWeightInput] = useState<string>('20.0');
  const [nearColsInput, setNearColsInput] = useState<string>('35-41');
  const [farColsInput, setFarColsInput] = useState<string>('1-11');
  const [promoKeywordsInput, setPromoKeywordsInput] = useState<string>('販促資材,販促,什器,資材');
  
  // ---- Pick/Storage levels ----
  const [pickLevelsInput, setPickLevelsInput] = useState<string>('1,2');
  const [storageLevelsInput, setStorageLevelsInput] = useState<string>('3,4');

  const chainDepth = useMemo(() => {
    if (chainDepthInput === '') return undefined;
    const n = parseInt(chainDepthInput, 10);
    return Number.isFinite(n) && n >= 0 ? n : undefined;
  }, [chainDepthInput]);

  const evictionBudget = useMemo(() => {
    if (evictionBudgetInput === '') return undefined;
    const n = parseInt(evictionBudgetInput, 10);
    return Number.isFinite(n) && n >= 0 ? n : undefined;
  }, [evictionBudgetInput]);

  const touchBudget = useMemo(() => {
    if (touchBudgetInput === '') return undefined;
    const n = parseInt(touchBudgetInput, 10);
    return Number.isFinite(n) && n >= 0 ? n : undefined;
  }, [touchBudgetInput]);

  const packLowMax = useMemo(() => {
    if (packLowMaxInput === '') return undefined;
    const n = parseInt(packLowMaxInput, 10);
    return Number.isFinite(n) && n >= 0 ? n : undefined;
  }, [packLowMaxInput]);

  const packHighMin = useMemo(() => {
    if (packHighMinInput === '') return undefined;
    const n = parseInt(packHighMinInput, 10);
    return Number.isFinite(n) && n >= 0 ? n : undefined;
  }, [packHighMinInput]);

  const bandPrefWeight = useMemo(() => {
    if (bandPrefWeightInput === '') return undefined;
    const n = parseFloat(bandPrefWeightInput);
    return Number.isFinite(n) ? n : undefined;
  }, [bandPrefWeightInput]);

  const nearCols = useMemo(() => {
    const arr = expandNumberRanges(nearColsInput);
    return arr.length ? arr : undefined;
  }, [nearColsInput]);

  const farCols = useMemo(() => {
    const arr = expandNumberRanges(farColsInput);
    return arr.length ? arr : undefined;
  }, [farColsInput]);

  const promoKeywords = useMemo(() => {
    const arr = splitCSV(promoKeywordsInput);
    return arr.length ? arr : undefined;
  }, [promoKeywordsInput]);

  const pickLevels = useMemo(() => {
    const arr = expandNumberRanges(pickLevelsInput);
    return arr.length ? arr : undefined;
  }, [pickLevelsInput]);

  const storageLevels = useMemo(() => {
    const arr = expandNumberRanges(storageLevelsInput);
    return arr.length ? arr : undefined;
  }, [storageLevelsInput]);

  // ---- Relocation (optimizer) ----
  const [maxMovesInput, setMaxMovesInput] = useState<string>(''); // 空欄OK
  const maxMoves = useMemo(() => {
    const n = parseInt(maxMovesInput, 10);
    return Number.isFinite(n) && n > 0 ? n : undefined; // 0/無効/空欄 → 未指定
  }, [maxMovesInput]);

  // SKU移動元ロケーション数制限（デフォルト: 2）
  const [maxSourceLocsInput, setMaxSourceLocsInput] = useState<string>('2');
  const maxSourceLocsPerSku = useMemo(() => {
    if (maxSourceLocsInput === '') return undefined; // 空欄=無制限
    const n = parseInt(maxSourceLocsInput, 10);
    return Number.isFinite(n) && n > 0 ? n : undefined;
  }, [maxSourceLocsInput]);

  const [fillRate, setFillRate] = useState<number>(0.9); // デフォルト0.9
  const [useAiMain, setUseAiMain] = useState<boolean>(true);
  const [reloStatus, setReloStatus] = useState<string>('');
  const [relocating, setRelocating] = useState<boolean>(false);
  const [usingSSE, setUsingSSE] = useState<boolean>(false);
  const esRef = useRef<EventSource | null>(null);

  // Live relocation debug polling (planned/accepted)
  const [livePlanned, setLivePlanned] = useState<number | null>(null);
  const [liveAccepted, setLiveAccepted] = useState<number | null>(null);
  const [liveRejections, setLiveRejections] = useState<Record<string, number> | null>(null);

  const [moves, setMoves] = useState<Move[]>([]);
  const [reloMeta, setReloMeta] = useState<any>(null);
  const [summary, setSummary] = useState<any | null>(null);
  const [summaryReport, setSummaryReport] = useState<string | null>(null); // 総合評価レポート

  // ---- Drop diagnostics ----
  const [showDrops, setShowDrops] = useState<boolean>(false);
  const [dropSummary, setDropSummary] = useState<DropSummaryItem[] | null>(null);
  const [dropDetails, setDropDetails] = useState<DropDetail[]>([]);
  const [dropTotal, setDropTotal] = useState<number>(0);
  const [dropPage, setDropPage] = useState<number>(0);
  const [dropLimit, setDropLimit] = useState<number>(20);
  const [dropLoading, setDropLoading] = useState<boolean>(false);

  // ---- Location export ----
  const [locationExporting, setLocationExporting] = useState<boolean>(false);
  const [comparisonExporting, setComparisonExporting] = useState<boolean>(false);

  const startRelocation = useCallback(async () => {
    setRelocating(true);
    setReloStatus('最適化を実行中...');
    setMoves([]);
    setSummaryReport(null); // 総合評価レポートをリセット
    try {
      // --- Prepare a client trace id and open SSE before firing the POST ---
      const traceId = (() => {
        try {
          const arr = new Uint8Array(6);
          crypto.getRandomValues(arr);
          return Array.from(arr).map((b) => b.toString(16).padStart(2, '0')).join('');
        } catch {
          return Math.random().toString(16).slice(2, 14);
        }
      })();

      // Open SSE stream (via Next.js rewrite in dev)
      try {
        if (esRef.current) { esRef.current.close(); esRef.current = null; }
        const es = new EventSource(`/v1/upload/relocation/stream?trace_id=${encodeURIComponent(traceId)}`);
        console.log('[SSE] Opening connection with trace_id:', traceId);
        esRef.current = es;
        setUsingSSE(true);
        
        es.onopen = () => {
          console.log('[SSE] Connection opened successfully');
        };
        
        es.onmessage = (ev) => {
          console.log('[SSE] Raw message received:', ev.data); // デバッグログ追加
          try {
            const evt = JSON.parse(ev.data || '{}');
            console.log('[SSE] Parsed event:', evt.type, evt); // デバッグログ追加
            if (evt.type === 'planned' && typeof evt.count === 'number') {
              setLivePlanned(evt.count);
              setReloStatus(`最適化を実行中... (planned=${evt.count}, accepted=${liveAccepted ?? '-'})`);
            } else if (evt.type === 'progress' && typeof evt.moves === 'number') {
              setReloStatus(`最適化を実行中... (scanning ${evt.processed ?? '?'} / ${evt.total ?? '?'}, moves=${evt.moves})`);
            } else if (evt.type === 'enforce_progress') {
              if (typeof evt.accepted === 'number') setLiveAccepted(evt.accepted);
              setReloStatus(`制約適用中... (${evt.processed ?? '?'} / ${evt.total ?? '?'}, accepted=${evt.accepted ?? '-'})`);
            } else if (evt.type === 'enforce_done') {
              if (typeof evt.accepted === 'number') setLiveAccepted(evt.accepted);
              if (typeof evt.planned === 'number') setLivePlanned(evt.planned);
              if (evt.rejections && typeof evt.rejections === 'object') setLiveRejections(evt.rejections as Record<string, number>);
              setReloStatus(`最終集計中... (planned=${evt.planned ?? '-'}, accepted=${evt.accepted ?? '-'})`);
            } else if (evt.type === 'summary_report') {
              // 総合評価レポートを受信
              console.log('[SSE] Summary report received:', evt);
              if (evt.report && typeof evt.report === 'string') {
                console.log('[SSE] Setting summary report, length:', evt.report.length);
                setSummaryReport(evt.report);
              } else {
                console.warn('[SSE] Summary report event but no report field:', evt);
              }
              // summary_report受信後にSSE接続を閉じる
              setTimeout(() => {
                try { es.close(); } catch {}
                esRef.current = null;
                setUsingSSE(false);
              }, 500);
            } else if (evt.type === 'done') {
              if (typeof evt.accepted === 'number') setLiveAccepted(evt.accepted);
              setReloStatus(`最適化が完了しました（accepted=${evt.accepted ?? '-'}）`);
              // done後はsummary_reportを待つため、すぐには閉じない
            }
          } catch { /* ignore parse errors */ }
        };
        es.onerror = () => {
          // Keep polling fallback; just mark SSE off if it errors repeatedly
          try { es.close(); } catch {}
          esRef.current = null;
          setUsingSSE(false);
        };
      } catch {
        setUsingSSE(false);
      }

      const payload: any = {
        block_codes: blocks,
        fill_rate: fillRate,
        use_ai_main: useAiMain,
        rotation_window_days: windowDays,
        ai_mode: 'fill_to_max',
      };
      // Attach client-provided trace id so SSE stream aligns
      payload.trace_id = traceId;
      if (typeof maxMoves === 'number') payload.max_moves = maxMoves; // 未指定なら送らない
      if (typeof maxSourceLocsPerSku === 'number') payload.max_source_locs_per_sku = maxSourceLocsPerSku;
      if (qualities.length) payload.quality_names = qualities;
      if (typeof chainDepth === 'number') payload.chain_depth = chainDepth;
      if (typeof evictionBudget === 'number') payload.eviction_budget = evictionBudget;
      if (typeof touchBudget === 'number') payload.touch_budget = touchBudget;
      payload.include_debug = true;

      // --- Advanced (band preferences) : send only when specified ---
      if (typeof packLowMax === 'number') payload.pack_low_max = packLowMax;
      if (typeof packHighMin === 'number') payload.pack_high_min = packHighMin;
      if (typeof bandPrefWeight === 'number') payload.band_pref_weight = bandPrefWeight;
      if (Array.isArray(nearCols) && nearCols.length) payload.near_cols = nearCols;
      if (Array.isArray(farCols) && farCols.length) payload.far_cols = farCols;
      if (Array.isArray(promoKeywords) && promoKeywords.length) payload.promo_quality_keywords = promoKeywords;

      // --- Pick/Storage levels : send only when specified ---
      if (Array.isArray(pickLevels) && pickLevels.length) payload.pick_levels = pickLevels;
      if (Array.isArray(storageLevels) && storageLevels.length) payload.storage_levels = storageLevels;

      // 実際にAPIを呼び出す（相対→rewrites / 直叩き 両対応）
  const { json, via } = await postWithFallback('/v1/upload/relocation/start', payload);
      console.log('API response:', json); // デバッグログ

      // メタ情報（あれば）
      const meta = json && typeof json === 'object' && !Array.isArray(json)
        ? {
            use_ai: json.use_ai,
            use_ai_main: json.use_ai_main,
            count: json.count ?? (Array.isArray(json?.moves) ? json.moves.length : undefined),
            blocks: json.blocks ?? blocks,
            quality_names: json.quality_names ?? qualities,
            max_moves: json.max_moves ?? maxMoves,
            fill_rate: json.fill_rate ?? fillRate,
            ai_hints_skus: json.ai_hints_skus,
            violations_summary: json.violations_summary,
            trace_id: json.trace_id ?? json?.meta?.trace_id,
          }
        : null;
  setReloMeta(meta ? { ...meta, trace_id: meta.trace_id || traceId } : { trace_id: traceId });
  // keep trace id in meta; live polling uses /relocation/debug snapshot
      setSummary(json?.summary ?? null);

      // 新しい計画開始時はドロップ診断をリセット
      setShowDrops(false);
      setDropSummary(null);
      setDropDetails([]);
      setDropTotal(0);
      setDropPage(0);

      // 移動案抽出（配列のフィールドのどれかを優先）
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
      
      // 最適化完了後、debugエンドポイントからsummary_reportを取得
      try {
        const debugPath = `/v1/upload/relocation/debug?trace_id=${encodeURIComponent(traceId)}`;
        console.log('[Relocation] Fetching summary report from:', debugPath);
        const { json: debugJson } = await getWithFallback(debugPath);
        console.log('[Relocation] Debug response:', debugJson);
        if (debugJson?.summary_report && typeof debugJson.summary_report === 'string') {
          console.log('[Relocation] Setting summary report, length:', debugJson.summary_report.length);
          setSummaryReport(debugJson.summary_report);
        } else {
          console.warn('[Relocation] No summary_report in debug response');
        }
      } catch (err) {
        console.warn('[Relocation] Failed to fetch summary report:', err);
      }
    } catch (e: any) {
      console.error('Relocation error:', e); // デバッグログ
      setReloStatus(`✖ 最適化に失敗: ${e?.message ?? String(e)}`);
    } finally {
      setRelocating(false);
      // Close SSE if still open
      try { if (esRef.current) { esRef.current.close(); esRef.current = null; } } catch {}
      setUsingSSE(false);
    }
  }, [
    blocks, maxMoves, fillRate, qualities, useAiMain, windowDays,
    chainDepth, evictionBudget, touchBudget,
    packLowMax, packHighMin, bandPrefWeight, nearCols, farCols, promoKeywords,
    liveAccepted,
  ]);

  // Poll relocation/debug while running to show real-time counts
  useEffect(() => {
    if (!relocating) return;
    if (usingSSE) return; // SSE中はポーリング抑止
    let stop = false;
    let timer: any = null;
    const tick = async () => {
      try {
        const { json } = await getWithFallback('/v1/upload/relocation/debug');
        if (stop) return;
        const p = typeof json?.planned === 'number' ? json.planned : null;
        const a = typeof json?.accepted === 'number' ? json.accepted : null;
        if (p !== null) setLivePlanned(p);
        if (a !== null) setLiveAccepted(a);
        if (p !== null || a !== null) {
          setReloStatus(`最適化を実行中... (planned=${p ?? '-'}, accepted=${a ?? '-'})`);
        }
      } catch {
        // ignore transient errors
      } finally {
        if (!stop) timer = setTimeout(tick, 2000);
      }
    };
    tick();
    return () => { stop = true; if (timer) clearTimeout(timer); };
  }, [relocating, usingSSE]);

  const exportCsv = useCallback(() => {
    if (!moves.length) return;
    const header = ['sku_id', 'lot', 'lot_date', 'qty', 'from_loc', 'to_loc', 'distance', 'reason'];
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
          (m as any).reason ?? '',
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

  // ローカルでエビクションチェーンを統合する関数
  const consolidateMovesLocally = useCallback((movesData: any[]) => {
    const chains = new Map<string, any[]>();
    
    // SKU+ロット別にチェーンを構築
    for (const move of movesData) {
      const key = `${move.sku_id || ''}_${move.lot || ''}`;
      if (!chains.has(key)) {
        chains.set(key, []);
      }
      chains.get(key)!.push(move);
    }
    
    const finalMoves: any[] = [];
    let consolidatedCount = 0;
    
    for (const [key, chain] of chains.entries()) {
      if (chain.length === 1) {
        // 単一移動の場合はそのまま
        finalMoves.push(chain[0]);
      } else {
        // 複数移動の場合は統合
        const firstMove = chain[0];
        const lastMove = chain[chain.length - 1];
        
        const consolidatedMove = {
          ...lastMove,
          from_loc: firstMove.from_loc,
          chain_info: {
            is_consolidated: true,
            original_steps: chain.length,
            intermediate_locations: chain.slice(0, -1).map(m => m.to_loc)
          }
        };
        
        finalMoves.push(consolidatedMove);
        consolidatedCount++;
      }
    }
    
    return {
      moves: finalMoves,
      count: finalMoves.length,
      original_count: movesData.length,
      consolidated_chains: consolidatedCount,
      efficiency_percent: (finalMoves.length / movesData.length) * 100
    };
  }, []);

  const exportFinalMovesOnly = useCallback(async () => {
    console.log('最終移動のみエクスポート開始');
    if (!moves.length) {
      alert('移動データがありません');
      return;
    }

    setRelocating(true);
    try {
      const payload = {
        max_moves: maxMoves,
        fill_rate: fillRate,
        block_codes: blocks,
        quality_names: qualities,
        use_ai: useAiMain,
        chain_depth: chainDepth,
        eviction_budget: evictionBudget,
        touch_budget: touchBudget,
      };

      console.log('最終移動API呼び出し中...', payload);
      let json: any;
      try {
        const result = await postWithFallback('/v1/upload/relocation/start/final-moves', payload);
        json = result.json;
      } catch (apiError) {
        console.warn('API呼び出しに失敗、既存データから統合処理を実行:', apiError);
        
        // 既存の移動データから統合処理を実行
        json = consolidateMovesLocally(moves);
      }
      
      if (!json.moves || !Array.isArray(json.moves)) {
        throw new Error('無効な応答データです');
      }

      console.log(`最終移動取得完了: ${json.original_count}件 → ${json.count}件（効率: ${json.efficiency_percent?.toFixed(1)}%）`);

      // CSV形式でエクスポート
      const header = ['sku_id', 'lot', 'lot_date', 'qty', 'from_loc', 'to_loc', 'distance', 'is_consolidated', 'original_steps', 'intermediate_locations', 'reason'];
      const lines = [
        header.join(','),
        ...json.moves.map((m: any) => {
          const chainInfo = m.chain_info || {};
          return [
            m.sku_id ?? '',
            m.lot ?? '',
            m.lot_date ?? '',
            String(m.qty ?? ''),
            m.from_loc ?? '',
            m.to_loc ?? '',
            (m.distance ?? '').toString(),
            chainInfo.is_consolidated ? 'はい' : 'いいえ',
            String(chainInfo.original_steps || 1),
            (chainInfo.intermediate_locations || []).join('; '),
            m.reason ?? '',
          ]
            .map((v) => '"' + String(v).replace(/"/g, '""') + '"')
            .join(',');
        }),
      ];

      const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `final_moves_only_${Date.now()}.csv`;
      a.click();
      URL.revokeObjectURL(url);

      alert(`作業員向け最終移動リストをエクスポートしました\n原移動: ${json.original_count}件\n最終移動: ${json.count}件\n統合効率: ${json.efficiency_percent?.toFixed(1)}%`);

    } catch (err: any) {
      console.error('最終移動エクスポートエラー:', err);
      const errorMessage = err?.message || String(err);
      alert(`最終移動エクスポートでエラーが発生しました:\n\n${errorMessage}\n\n※ バックエンドサーバーが起動していることを確認してください`);
    } finally {
      setRelocating(false);
    }
  }, [
    moves, maxMoves, fillRate, blocks, qualities, useAiMain,
    chainDepth, evictionBudget, touchBudget
  ]);

  const exportLocationSnapshot = useCallback(async () => {
    console.log('ロケーション一覧エクスポート開始', { blocks, moves });
    setLocationExporting(true);
    try {
      // 全在庫データを取得（変更前の状態）
      const beforeMoves = new Map<string, any>();
      const afterMoves = new Map<string, any>();
      
      // 現在の在庫状態を取得
      const limit = 5000;
      let offset = 0;
      const inventoryRows: any[] = [];
      
      while (true) {
        const blockParam = blocks.length ? `&block=${blocks.join(',')}` : '';
        const { json } = await getWithFallback(`/v1/debug/inventory?limit=${limit}&offset=${offset}${blockParam}`);
        console.log(`在庫データ取得: offset=${offset}, total=${json?.total}`, json);
        const rows = Array.isArray(json?.rows) ? json.rows : [];
        if (!rows.length) break;
        
        inventoryRows.push(...rows);
        const total = typeof json?.total === 'number' ? json.total : 0;
        offset += rows.length;
        if (offset >= total || offset >= 50000) break; // 安全上限
      }

      // 変更前の状態を記録
      inventoryRows.forEach((row: any) => {
        const key = `${row.sku_id}_${row.lot || 'NOLOT'}_${row.location_id || row.location || ''}`;
        beforeMoves.set(key, {
          sku_id: row.sku_id,
          sku_name: row.sku_name || row['商品名'] || row.sku_id || '',
          lot: row.lot || row['ロット'] || '',
          lot_date: row.lot_date || row['ロット日付'] || '',
          location: row.location_id || row.location || '',
          qty_cases: row.cases || row.qty_cases || row['ケース数'] || 0,
          quality_name: row.quality_name || row['品質区分名'] || '',
          status: '変更前'
        });
        afterMoves.set(key, { ...beforeMoves.get(key), status: '変更後' });
      });

      // 移動計画を変更後の状態に反映
      moves.forEach((move) => {
        const fromKey = `${move.sku_id}_${move.lot || 'NOLOT'}_${move.from_loc}`;
        const toKey = `${move.sku_id}_${move.lot || 'NOLOT'}_${move.to_loc}`;
        
        // 移動元から数量を減らす
        const fromItem = afterMoves.get(fromKey);
        if (fromItem) {
          fromItem.qty_cases = Math.max(0, (fromItem.qty_cases || 0) - move.qty);
          if (fromItem.qty_cases === 0) {
            afterMoves.delete(fromKey);
          }
        }
        
        // 移動先に数量を追加
        const toItem = afterMoves.get(toKey);
        if (toItem) {
          toItem.qty_cases = (toItem.qty_cases || 0) + move.qty;
        } else {
          const baseItem = beforeMoves.get(fromKey);
          if (baseItem) {
            afterMoves.set(toKey, {
              ...baseItem,
              location: move.to_loc,
              qty_cases: move.qty,
              status: '変更後'
            });
          }
        }
      });

      // CSVデータを生成
      const allItems = [
        ...Array.from(beforeMoves.values()),
        ...Array.from(afterMoves.values())
      ];

      const header = [
        'status', 'sku_id', 'sku_name', 'lot', 'lot_date', 'location', 
        'qty_cases', 'quality_name'
      ];
      
      const lines = [
        header.join(','),
        ...allItems.map((item) => [
          item.status,
          item.sku_id,
          item.sku_name,
          item.lot,
          item.lot_date,
          item.location,
          String(item.qty_cases),
          item.quality_name,
        ].map((v) => '"' + String(v).replace(/"/g, '""') + '"').join(','))
      ];

      const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `location_snapshot_${Date.now()}.csv`;
      a.click();
      URL.revokeObjectURL(url);
      
    } catch (error) {
      console.error('ロケーション一覧のエクスポートに失敗しました:', error);
      alert(`ロケーション一覧のエクスポートに失敗しました: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setLocationExporting(false);
    }
  }, [blocks, moves]);

  const exportComparisonData = useCallback(async () => {
    setComparisonExporting(true);
    try {
      // 現在の在庫状態を取得
      const limit = 5000;
      let offset = 0;
      const inventoryRows: any[] = [];
      
      while (true) {
        const blockParam = blocks.length ? `&block=${blocks.join(',')}` : '';
        const { json } = await getWithFallback(`/v1/debug/inventory?limit=${limit}&offset=${offset}${blockParam}`);
        const rows = Array.isArray(json?.rows) ? json.rows : [];
        if (!rows.length) break;
        
        inventoryRows.push(...rows);
        const total = typeof json?.total === 'number' ? json.total : 0;
        offset += rows.length;
        if (offset >= total || offset >= 50000) break;
      }

      // ロケーション別に集約（変更前の状態）
      const beforeLocationData = new Map<string, any>();
      const afterLocationData = new Map<string, any>();

      inventoryRows.forEach((row: any) => {
        const location = row.location_id || row.location || '';
        if (!location) return;

        const key = location;
        if (!beforeLocationData.has(key)) {
          beforeLocationData.set(key, {
            location,
            total_qty: 0,
            sku_count: 0,
            skus: new Set<string>()
          });
        }
        
        const locData = beforeLocationData.get(key)!;
        locData.total_qty += row.cases || row.qty_cases || row['ケース数'] || 0;
        if (row.sku_id) {
          locData.skus.add(row.sku_id);
        }
        locData.sku_count = locData.skus.size;

        // 変更後の初期状態をコピー
        afterLocationData.set(key, {
          location,
          total_qty: locData.total_qty,
          sku_count: locData.sku_count,
          skus: new Set(locData.skus)
        });
      });

      // 移動計画を変更後の状態に反映
      moves.forEach((move) => {
        const fromLoc = move.from_loc;
        const toLoc = move.to_loc;
        const qty = move.qty || 0;

        // 移動元の数量を減らす
        if (afterLocationData.has(fromLoc)) {
          const fromData = afterLocationData.get(fromLoc)!;
          fromData.total_qty = Math.max(0, fromData.total_qty - qty);
        }

        // 移動先の数量を増やす
        if (!afterLocationData.has(toLoc)) {
          afterLocationData.set(toLoc, {
            location: toLoc,
            total_qty: 0,
            sku_count: 0,
            skus: new Set<string>()
          });
        }
        const toData = afterLocationData.get(toLoc)!;
        toData.total_qty += qty;
        if (move.sku_id) {
          toData.skus.add(move.sku_id);
          toData.sku_count = toData.skus.size;
        }
      });

      // 比較データを生成
      const allLocations = new Set([
        ...beforeLocationData.keys(),
        ...afterLocationData.keys()
      ]);

      const comparisonData: any[] = [];
      allLocations.forEach(location => {
        const before = beforeLocationData.get(location);
        const after = afterLocationData.get(location);
        
        const beforeQty = before?.total_qty || 0;
        const afterQty = after?.total_qty || 0;
        const beforeSkuCount = before?.sku_count || 0;
        const afterSkuCount = after?.sku_count || 0;
        
        const qtyChange = afterQty - beforeQty;
        const skuChange = afterSkuCount - beforeSkuCount;
        
        // 変更があった場合、または元々データがある場合のみ出力
        if (beforeQty > 0 || afterQty > 0) {
          comparisonData.push({
            location,
            before_qty: beforeQty,
            after_qty: afterQty,
            qty_change: qtyChange,
            before_sku_count: beforeSkuCount,
            after_sku_count: afterSkuCount,
            sku_change: skuChange,
            change_type: qtyChange > 0 ? '増加' : qtyChange < 0 ? '減少' : '変更なし'
          });
        }
      });

      // 変更量でソート（変更量の大きい順）
      comparisonData.sort((a, b) => Math.abs(b.qty_change) - Math.abs(a.qty_change));

      // CSVデータを生成
      const header = [
        'ロケーション', '最適化前_ケース数', '最適化後_ケース数', 'ケース数変更',
        '最適化前_SKU数', '最適化後_SKU数', 'SKU数変更', '変更種別'
      ];
      
      const lines = [
        header.join(','),
        ...comparisonData.map((item) => [
          item.location,
          String(item.before_qty),
          String(item.after_qty),
          String(item.qty_change),
          String(item.before_sku_count),
          String(item.after_sku_count),
          String(item.sku_change),
          item.change_type
        ].map((v) => '"' + String(v).replace(/"/g, '""') + '"').join(','))
      ];

      const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `location_comparison_${Date.now()}.csv`;
      a.click();
      URL.revokeObjectURL(url);
      
    } catch (error) {
      console.error('比較データのエクスポートに失敗しました:', error);
      alert(`比較データのエクスポートに失敗しました: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setComparisonExporting(false);
    }
  }, [blocks, moves]);

  // 移動前のロケーション詳細データを出力
  const exportBeforeLocationData = useCallback(async () => {
    setComparisonExporting(true);
    try {
      // 現在の在庫状態を取得
      const limit = 5000;
      let offset = 0;
      const inventoryRows: any[] = [];
      
      while (true) {
        const blockParam = blocks.length ? `&block=${blocks.join(',')}` : '';
        const { json } = await getWithFallback(`/v1/debug/inventory?limit=${limit}&offset=${offset}${blockParam}`);
        const rows = Array.isArray(json?.rows) ? json.rows : [];
        if (!rows.length) break;
        
        inventoryRows.push(...rows);
        const total = typeof json?.total === 'number' ? json.total : 0;
        offset += rows.length;
        if (offset >= total || offset >= 50000) break;
      }

      // 移動前の詳細データを準備
      const beforeData = inventoryRows.map((row: any) => ({
        location: row.location_id || row.location || '',
        sku_id: row.sku_id || '',
        sku_name: row.sku_name || row['商品名'] || '',
        cases: row.cases || row.qty_cases || row['ケース数'] || 0,
        quality_name: row.quality_name || row['品質'] || '',
        lot: row.lot || '',
        lot_date: row.lot_date || '',
        volume_m3: row.volume_m3 || 0
      })).filter((item: any) => item.location && item.sku_id && item.cases > 0);

      // ロケーション順にソート
      beforeData.sort((a: any, b: any) => a.location.localeCompare(b.location));

      // CSVデータを生成
      const header = [
        'ロケーション', 'SKU_ID', 'SKU名', 'ケース数', '品質', 'ロット', 'ロット日付', '体積_m3'
      ];
      
      const lines = [
        header.join(','),
        ...beforeData.map((item: any) => [
          item.location,
          item.sku_id,
          item.sku_name,
          String(item.cases),
          item.quality_name,
          item.lot,
          item.lot_date,
          String(item.volume_m3)
        ].map((v) => '"' + String(v).replace(/"/g, '""') + '"').join(','))
      ];

      const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `location_before_${Date.now()}.csv`;
      a.click();
      URL.revokeObjectURL(url);
      
    } catch (error) {
      console.error('移動前データのエクスポートに失敗しました:', error);
      alert(`移動前データのエクスポートに失敗しました: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setComparisonExporting(false);
    }
  }, [blocks]);

  // 移動後のロケーション詳細データを出力
  const exportAfterLocationData = useCallback(async () => {
    setComparisonExporting(true);
    try {
      // 現在の在庫状態を取得
      const limit = 5000;
      let offset = 0;
      const inventoryRows: any[] = [];
      
      while (true) {
        const blockParam = blocks.length ? `&block=${blocks.join(',')}` : '';
        const { json } = await getWithFallback(`/v1/debug/inventory?limit=${limit}&offset=${offset}${blockParam}`);
        const rows = Array.isArray(json?.rows) ? json.rows : [];
        if (!rows.length) break;
        
        inventoryRows.push(...rows);
        const total = typeof json?.total === 'number' ? json.total : 0;
        offset += rows.length;
        if (offset >= total || offset >= 50000) break;
      }

      // 移動前の状態をマップに変換（ロケーション + SKU + ロットの組み合わせをキーとする）
      const inventoryMap = new Map<string, any>();
      inventoryRows.forEach((row: any) => {
        const location = row.location_id || row.location || '';
        const sku_id = row.sku_id || '';
        const lot = row.lot || '';
        const key = `${location}|${sku_id}|${lot}`;
        
        inventoryMap.set(key, {
          location,
          sku_id,
          sku_name: row.sku_name || row['商品名'] || '',
          cases: row.cases || row.qty_cases || row['ケース数'] || 0,
          quality_name: row.quality_name || row['品質'] || '',
          lot,
          lot_date: row.lot_date || '',
          volume_m3: row.volume_m3 || 0,
          relocated: false,  // 移動フラグを追加
          relocation_info: ''  // 移動情報を追加
        });
      });

      // 移動されたアイテムの追跡用マップ
      const movedItems = new Map<string, { from_loc: string; to_loc: string; qty: number }[]>();

      // 移動計画を反映
      moves.forEach((move) => {
        const fromLoc = move.from_loc;
        const toLoc = move.to_loc;
        const sku_id = move.sku_id || '';
        const lot = move.lot || '';
        const qty = move.qty || 0;
        
        // 移動情報を追跡
        const moveKey = `${sku_id}|${lot}`;
        if (!movedItems.has(moveKey)) {
          movedItems.set(moveKey, []);
        }
        movedItems.get(moveKey)!.push({ from_loc: fromLoc, to_loc: toLoc, qty });
        
        // 移動元から減らす
        const fromKey = `${fromLoc}|${sku_id}|${lot}`;
        if (inventoryMap.has(fromKey)) {
          const fromItem = inventoryMap.get(fromKey)!;
          fromItem.cases = Math.max(0, fromItem.cases - qty);
          
          // 0になった場合は削除
          if (fromItem.cases === 0) {
            inventoryMap.delete(fromKey);
          }
        }
        
        // 移動先に追加
        const toKey = `${toLoc}|${sku_id}|${lot}`;
        if (inventoryMap.has(toKey)) {
          const toItem = inventoryMap.get(toKey)!;
          toItem.cases += qty;
          // 既存のアイテムが移動先になった場合
          toItem.relocated = true;
          toItem.relocation_info = `移動先(+${qty})`;
        } else {
          // 新しいエントリを作成
          const sourceItem = inventoryRows.find(row => 
            (row.location_id || row.location) === fromLoc && 
            (row.sku_id || '') === sku_id && 
            (row.lot || '') === lot
          );
          
          if (sourceItem) {
            inventoryMap.set(toKey, {
              location: toLoc,
              sku_id,
              sku_name: sourceItem.sku_name || sourceItem['商品名'] || '',
              cases: qty,
              quality_name: sourceItem.quality_name || sourceItem['品質'] || '',
              lot,
              lot_date: sourceItem.lot_date || '',
              volume_m3: sourceItem.volume_m3 || 0,
              relocated: true,  // 新しい場所に移動されたアイテム
              relocation_info: `移動元: ${fromLoc}`
            });
          }
        }
      });

      // 移動後のデータを配列に変換
      const afterData = Array.from(inventoryMap.values())
        .filter((item: any) => item.cases > 0)
        .sort((a: any, b: any) => a.location.localeCompare(b.location));

      // CSVデータを生成
      const header = [
        'ロケーション', 'SKU_ID', 'SKU名', 'ケース数', '品質', 'ロット', 'ロット日付', '体積_m3', '移動フラグ', '移動情報'
      ];
      
      const lines = [
        header.join(','),
        ...afterData.map((item: any) => [
          item.location,
          item.sku_id,
          item.sku_name,
          String(item.cases),
          item.quality_name,
          item.lot,
          item.lot_date,
          String(item.volume_m3),
          item.relocated ? '移動済み' : '元の位置',
          item.relocation_info || ''
        ].map((v) => '"' + String(v).replace(/"/g, '""') + '"').join(','))
      ];

      const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `location_after_${Date.now()}.csv`;
      a.click();
      URL.revokeObjectURL(url);
      
    } catch (error) {
      console.error('移動後データのエクスポートに失敗しました:', error);
      alert(`移動後データのエクスポートに失敗しました: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setComparisonExporting(false);
    }
  }, [blocks, moves]);

  const fetchDropSummary = useCallback(async () => {
    if (!reloMeta?.trace_id) return;
    setDropLoading(true);
    try {
      const { json } = await getWithFallback(`/v1/debug/relocation/trace/${reloMeta.trace_id}/drops?agg=reason`);
      const items = coerceDropSummary(json);
      setDropSummary(items);
    } catch (_e) {
      setDropSummary([]);
    } finally {
      setDropLoading(false);
    }
  }, [reloMeta?.trace_id]);

  const fetchDropDetails = useCallback(async () => {
    if (!reloMeta?.trace_id) return;
    setDropLoading(true);
    try {
      const params = new URLSearchParams();
      params.set('limit', String(dropLimit));
      params.set('offset', String(dropPage * dropLimit));
      const { json } = await getWithFallback(`/v1/debug/relocation/trace/${reloMeta.trace_id}/drops?` + params.toString());
      const details = coerceDropDetails(json);
      setDropDetails(details);
      const total = Number(json?.total ?? json?.count ?? 0);
      setDropTotal(Number.isFinite(total) ? total : details.length + dropPage * dropLimit);
    } catch (_e) {
      setDropDetails([]);
      setDropTotal(0);
    } finally {
      setDropLoading(false);
    }
  }, [reloMeta?.trace_id, dropLimit, dropPage]);

  const exportDropCsv = useCallback(() => {
    if (!dropDetails.length) return;
    const header = ['sku_id', 'lot', 'qty', 'from_loc', 'to_loc', 'distance', 'reason', 'stage', 'rule', 'message'];
    const lines = [
      header.join(','),
      ...dropDetails.map((d) =>
        [
          d.sku_id ?? '',
          d.lot ?? '',
          d.qty ?? '',
          d.from_loc ?? '',
          d.to_loc ?? '',
          d.distance ?? '',
          d.reason ?? '',
          d.stage ?? '',
          d.rule ?? '',
          d.message ?? '',
        ]
          .map((v) => '"' + String(v).replace(/"/g, '""') + '"')
          .join(',')
      ),
    ];
    const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `rejected_candidates_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [dropDetails]);

  // NOTE: 外枠の幅・余白は Layout 側が管理。ここでは container を持たない
  return (
    <main role="main" className="space-y-8">
      {/* ページ見出しは Topbar の pageTitle を使う（重複回避） */}

      {/* 総合評価レポート */}
      {summaryReport && (
        <section className="bg-white rounded-2xl shadow-sm ring-1 ring-black/5 p-6">
          <h2 className="text-lg font-semibold mb-4">📊 リロケーション結果の総合評価</h2>
          <div style={{ 
            background: '#f5f5f5', 
            border: '1px solid #ddd', 
            borderRadius: 8, 
            padding: 16,
            fontFamily: 'monospace',
            fontSize: '13px',
            whiteSpace: 'pre-wrap',
            overflowX: 'auto',
            lineHeight: 1.5
          }}>
            {summaryReport}
          </div>
        </section>
      )}

      <section className="bg-white rounded-2xl shadow-sm ring-1 ring-black/5 p-6">
        <h2 className="text-lg font-semibold mb-4">対象とパラメータ</h2>
        <div className="flex flex-wrap items-center gap-3">
          {/* ブロック/品質 */}
          <label className="text-sm">
            ブロック（カンマ区切り）：
            <input
              className="ml-2 rounded-lg border border-black/10 bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-black/20"
              value={blockText}
              onChange={(e) => setBlockText(e.target.value)}
              placeholder="B"
            />
          </label>
          <div className="flex items-center gap-2">
            <div className="text-sm">
              <div className="mb-1">品質区分（複数選択可）：</div>
              <div className="border border-black/10 rounded-lg bg-white p-2 w-64 h-28 overflow-auto">
                {qualityChoices.length ? (
                  qualityChoices.map((q) => (
                    <label key={q} className="flex items-center gap-2 text-sm py-0.5 cursor-pointer select-none">
                      <input
                        type="checkbox"
                        className="h-3.5 w-3.5 rounded border-black/20 text-black focus:ring-black/20"
                        checked={selectedQualities.includes(q)}
                        onChange={(e) => {
                          setSelectedQualities((prev) =>
                            e.target.checked ? [...prev.filter((x) => x !== q), q] : prev.filter((x) => x !== q)
                          );
                        }}
                      />
                      <span>{q}</span>
                    </label>
                  ))
                ) : (
                  <div className="text-xs text-gray-500">候補がありません</div>
                )}
              </div>
              <div className="text-[11px] text-gray-500 mt-1">クリックで選択/解除できます（未選択なら全てが対象）</div>
            </div>
            <button
              type="button"
              onClick={() => setSelectedQualities(qualityChoices)}
              className="px-3 py-2 rounded-lg text-xs border border-black/10 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-black/20"
              title="全てを選択"
            >
              全選択
            </button>
            <button
              type="button"
              onClick={() => setSelectedQualities([])}
              className="px-3 py-2 rounded-lg text-xs border border-black/10 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-black/20"
              title="すべて解除"
            >
              解除
            </button>
            <button
              type="button"
              onClick={loadQualityChoices}
              className="px-3 py-2 rounded-lg text-xs border border-black/10 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-black/20"
              title="DBから最新の品質区分を再取得"
            >
              再読込
            </button>
          </div>

          {/* パラメータ */}
          <label className="text-sm">
            最大移動行数：
            <input
              type="number"
              className="ml-2 rounded-lg border border-black/10 bg-white px-3 py-2 text-sm w-28 focus:outline-none focus:ring-2 focus:ring-black/20"
              min={0}
              value={maxMovesInput}
              onChange={(e) => setMaxMovesInput(e.target.value)}
              placeholder="空欄=無制限"
              title="0 または空欄で無制限（バックエンド既定）"
            />
          </label>
          <label className="text-sm">
            1SKU最大移動元ロケ数：
            <input
              type="number"
              className="ml-2 rounded-lg border border-black/10 bg-white px-3 py-2 text-sm w-20 focus:outline-none focus:ring-2 focus:ring-black/20"
              min={1}
              value={maxSourceLocsInput}
              onChange={(e) => setMaxSourceLocsInput(e.target.value)}
              placeholder="空欄=無制限"
              title="1SKUあたり最大何ロケーションから移動するか制限（空欄で無制限）。デフォルト2 = 最も古いロットの2ロケーションを優先"
            />
          </label>
          <label className="text-sm">
            充填率上限（0.1–1.0）：
            <input
              type="number"
              step={0.05}
              className="ml-2 rounded-lg border border-black/10 bg-white px-3 py-2 text-sm w-28 focus:outline-none focus:ring-2 focus:ring-black/20"
              min={0.1}
              max={1.0}
              value={fillRate}
              onChange={(e) => setFillRate(parseFloat(e.target.value || '0.95'))}
              title="📦 棚の容量をどの程度まで使用するかを設定します。
1.0 = 100%（満杯まで使用）
0.95 = 95%（推奨：余裕を持った設定）
0.8 = 80%（保守的な設定）

例：0.95の場合、100個入る棚には最大95個まで配置します。"
            />
          </label>
          <label className="text-sm">
            出荷データ分析期間：
            <select
              className="ml-2 rounded-lg border border-black/10 bg-white px-3 py-2 text-sm w-36 focus:outline-none focus:ring-2 focus:ring-black/20"
              value={String(windowDays)}
              onChange={(e) => {
                const v = parseInt(e.target.value, 10);
                setWindowDays(Number.isFinite(v) ? v : 90);
              }}
              title="📊 出荷頻度の高い商品を判断する期間を設定します。
過去3ヶ月：最新のトレンドに基づいた分析
過去6ヶ月：季節性も含めたバランス分析（推奨）
過去1年：長期的なトレンド分析
全期間：蓄積された全データで分析"
            >
              <option value="90">過去3ヶ月</option>
              <option value="180">過去6ヶ月</option>
              <option value="365">過去1年</option>
              <option value="0">全期間</option>
            </select>
          </label>
          <label className="text-sm">
            段階移動の深さ：
            <input
              type="number"
              min={0}
              className="ml-2 rounded-lg border border-black/10 bg-white px-3 py-2 text-sm w-28 focus:outline-none focus:ring-2 focus:ring-black/20"
              value={chainDepthInput}
              onChange={(e) => setChainDepthInput(e.target.value)}
              placeholder="推奨: 0"
              title="🔄 商品を段階的に移動する深さを設定します。
0：シンプルな1対1移動のみ
1：A → B → 空き棚（2段階）
2：A → B → C → 空き棚（3段階・推奨）
3以上：より複雑な移動（時間がかかる）

数値が大きいほど最適化は向上しますが、作業時間も増加します。"
            />
          </label>
          <label className="text-sm">
            一時移動の上限回数：
            <input
              type="number"
              min={0}
              className="ml-2 rounded-lg border border-black/10 bg-white px-3 py-2 text-sm w-28 focus:outline-none focus:ring-2 focus:ring-black/20"
              value={evictionBudgetInput}
              onChange={(e) => setEvictionBudgetInput(e.target.value)}
              placeholder="推奨: 0"
              title="📦 商品を一時的に別の場所に退避させる最大回数。
0：一時移動なし（最小限の作業）
100-300：小規模な一時移動（推奨）
500-1000：大規模な一時移動
制限なし：最適化優先（作業量大）

理想の配置にするため、商品を一時的に別の棚に移してスペースを作る作業の上限です。"
            />
          </label>
          <label className="text-sm">
            移動対象ロケーション数：
            <input
              type="number"
              min={0}
              className="ml-2 rounded-lg border border-black/10 bg-white px-3 py-2 text-sm w-28 focus:outline-none focus:ring-2 focus:ring-black/20"
              value={touchBudgetInput}
              onChange={(e) => setTouchBudgetInput(e.target.value)}
              placeholder="推奨: 0"
              title="🏢 一度の最適化で触る棚の場所の最大数。
100-300：小規模な範囲での最適化
500-1000：中規模な範囲での最適化（推奨）
1500以上：大規模な倉庫全体の最適化

作業員の負担を調整するため、移動作業の範囲を制限します。小さい数値ほど作業が楽になります。"
            />
          </label>
          <label className="text-sm">
            AI自動最適化を使用：
            <input
              type="checkbox"
              className="ml-2 align-middle h-4 w-4 rounded border-black/10 text-black focus:ring-black/20"
              checked={useAiMain}
              onChange={(e) => setUseAiMain(e.target.checked)}
              title="🤖 AI（人工知能）による高度な最適化機能を使用します。
ON：AIが倉庫全体を分析し、最適な配置を自動提案（推奨）
OFF：従来のルールベース最適化を使用

AIを使用することで、より効率的で実用的な配置提案が得られます。"
            />
          </label>

          <button
            onClick={startRelocation}
            disabled={relocating}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition active:scale-[.99] ${
              relocating
                ? 'bg-gray-300 text-gray-600 cursor-not-allowed'
                : 'bg-black text-white hover:bg-black/90 focus:outline-none focus:ring-2 focus:ring-black/20'
            }`}
            title="🚀 設定した条件に基づいて倉庫の最適化を開始します。
・よく出荷される商品を1-2段目（ピック専用段）に配置
・古いロットを優先的に出荷しやすい位置に移動  
・商品の重さに応じたエリア分けを実施
・作業効率と保管効率を両立した配置を提案

処理には数分かかる場合があります。"
          >
            {relocating ? '最適化中…' : '最適化を実行'}
          </button>
          <button
            onClick={exportCsv}
            disabled={!moves.length || relocating}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition border ${
              !moves.length || relocating
                ? 'bg-gray-100 text-gray-400 border-black/10 cursor-not-allowed'
                : 'bg-white text-gray-900 border-black/10 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-black/20'
            }`}
            title="📊 最適化によって提案された移動計画をCSVファイルで出力します。
商品名・ロット・移動元・移動先・数量・距離などの詳細情報が含まれます。
Excelで開いて詳細分析や作業計画の作成に活用できます。"
          >
            CSVエクスポート
          </button>
          <button
            onClick={exportFinalMovesOnly}
            disabled={!moves.length || relocating}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition border ${
              !moves.length || relocating
                ? 'bg-gray-100 text-gray-400 border-black/10 cursor-not-allowed'
                : 'bg-green-50 text-green-800 border-green-200 hover:bg-green-100 focus:outline-none focus:ring-2 focus:ring-green-200'
            }`}
            title="🎯 エビクションチェーンの中間移動を統合し、作業員向けの最終移動のみをエクスポートします。
同一SKU+ロットの複数回移動を統合して、実際に必要な移動（開始位置→最終位置）のみを出力します。
作業効率を向上させ、二度手間を防ぐことができます。"
          >
            最終移動のみエクスポート
          </button>
          <button
            onClick={exportLocationSnapshot}
            disabled={locationExporting}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition border ${
              locationExporting
                ? 'bg-gray-100 text-gray-400 border-black/10 cursor-not-allowed'
                : 'bg-blue-50 text-blue-700 border-blue-200 hover:bg-blue-100 focus:outline-none focus:ring-2 focus:ring-blue/20'
            }`}
            title="📋 変更前・変更後の全ロケーション状態をCSVファイルで出力します。
どの商品がどこからどこへ移動したかが一目でわかる一覧表を作成できます。
作業指示書や記録として活用してください。"
          >
            {locationExporting ? 'エクスポート中…' : 'ロケーション一覧'}
          </button>
          <button
            onClick={exportComparisonData}
            disabled={!moves.length || comparisonExporting}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition border ${
              !moves.length || comparisonExporting
                ? 'bg-gray-100 text-gray-400 border-black/10 cursor-not-allowed'
                : 'bg-green-50 text-green-700 border-green-200 hover:bg-green-100 focus:outline-none focus:ring-2 focus:ring-green/20'
            }`}
            title="📊 ロケーション別の最適化前後比較表をCSVファイルで出力します。
各ロケーションのケース数とSKU数の変化が一覧でわかります。
・変更量の大きい順にソート
・増加/減少/変更なしの分類
・倉庫全体の配置変化の把握に便利"
          >
            {comparisonExporting ? 'エクスポート中…' : '比較データ'}
          </button>
          <button
            type="button"
            onClick={exportBeforeLocationData}
            disabled={comparisonExporting}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition border ${
              comparisonExporting
                ? 'bg-gray-100 text-gray-400 border-black/10 cursor-not-allowed'
                : 'bg-blue-50 text-blue-700 border-blue-200 hover:bg-blue-100 focus:outline-none focus:ring-2 focus:ring-blue/20'
            }`}
            title="📋 移動前の詳細ロケーション一覧をCSVファイルで出力します。
各ロケーションにどのSKUがいくつあるかの現在状況が確認できます。
・ロケーション別詳細
・SKU名、ケース数、品質、ロット情報を含む
・現在の在庫配置の把握に便利"
          >
            {comparisonExporting ? 'エクスポート中…' : '移動前詳細'}
          </button>
          <button
            type="button"
            onClick={exportAfterLocationData}
            disabled={!moves.length || comparisonExporting}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition border ${
              !moves.length || comparisonExporting
                ? 'bg-gray-100 text-gray-400 border-black/10 cursor-not-allowed'
                : 'bg-purple-50 text-purple-700 border-purple-200 hover:bg-purple-100 focus:outline-none focus:ring-2 focus:ring-purple/20'
            }`}
            title="📋 移動後の詳細ロケーション一覧をCSVファイルで出力します。
移動計画を実行した場合の各ロケーションの詳細状況が確認できます。
・移動計画適用後の配置
・SKU名、ケース数、品質、ロット情報を含む
・最適化後の在庫配置の把握に便利"
          >
            {comparisonExporting ? 'エクスポート中…' : '移動後詳細'}
          </button>
          <button
            type="button"
            onClick={() => setAdvancedOpen((v) => !v)}
            className="px-3 py-2 rounded-lg text-xs border border-black/10 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-black/20"
            title="高度な設定（帯の嗜好・しきい値）を開く/閉じる"
          >
            {advancedOpen ? '高度な設定を隠す' : '高度な設定'}
          </button>
        </div>

        {advancedOpen && (
          <div className="mt-3 p-3 border border-black/10 rounded-xl bg-[#fafafa]">
            <div className="text-xs font-medium mb-2">高度な設定（未入力はバックエンド既定を使用）</div>
            <div className="flex flex-wrap items-center gap-3">
              <label className="text-sm">
                重い商品の基準（入り数上限）：
                <input
                  type="number"
                  min={0}
                  className="ml-2 rounded-lg border border-black/10 bg-white px-3 py-2 text-sm w-28 focus:outline-none focus:ring-2 focus:ring-black/20"
                  value={packLowMaxInput}
                  onChange={(e) => setPackLowMaxInput(e.target.value)}
                  placeholder="例: 12"
                  title="📦 1ケースあたりの入り数がこの値以下の商品を「重い商品」として扱います。
例：入り数12個以下 → 1個が重い商品（米、調味料など）
重い商品は奥のエリア（1-11列）に配置され、作業負荷を軽減します。"
                />
              </label>
              <label className="text-sm">
                軽い商品の基準（入り数下限）：
                <input
                  type="number"
                  min={0}
                  className="ml-2 rounded-lg border border-black/10 bg-white px-3 py-2 text-sm w-28 focus:outline-none focus:ring-2 focus:ring-black/20"
                  value={packHighMinInput}
                  onChange={(e) => setPackHighMinInput(e.target.value)}
                  placeholder="例: 50"
                  title="📦 1ケースあたりの入り数がこの値以上の商品を「軽い商品」として扱います。
例：入り数50個以上 → 1個が軽い商品（お菓子、小物など）
軽い商品は出入口に近いエリア（35-41列）に配置され、頻繁にピッキングしやすくします。"
                />
              </label>
              <label className="text-sm">
                エリア分けの厳格さ：
                <input
                  type="number"
                  step={0.1}
                  className="ml-2 rounded-lg border border-black/10 bg-white px-3 py-2 text-sm w-28 focus:outline-none focus:ring-2 focus:ring-black/20"
                  value={bandPrefWeightInput}
                  onChange={(e) => setBandPrefWeightInput(e.target.value)}
                  placeholder="例: 20.0"
                  title="⚖️ 商品を適切なエリアに配置する際の厳格さを設定します。

【動作詳細】
• 重い商品（入り数≤12個）→ 出入口近い列（35-41列）
• 軽い商品（入り数≥50個）→ 奥の列（1-11列）  
• 販促品 → 奥の列（1-11列）

【数値の効果】
• 0: エリア分けなし（商品種別を無視）
• 10.0: 緩やかなエリア分け
• 20.0: 標準的なエリア分け（推奨）
• 50.0以上: 非常に厳格なエリア分け（混在を強く回避）

数値が大きいほど、異なる種別の商品が同じエリアに配置されることを強く避けます。"
                />
              </label>
              <label className="text-sm">
                出入口に近いエリア：
                <input
                  type="text"
                  className="ml-2 rounded-lg border border-black/10 bg-white px-3 py-2 text-sm w-40 focus:outline-none focus:ring-2 focus:ring-black/20"
                  value={nearColsInput}
                  onChange={(e) => setNearColsInput(e.target.value)}
                  placeholder="35-41 または 35,36,37"
                  title="🚪 出入口に近く、アクセスしやすいエリアの列番号を指定します。
軽い商品や頻繁に出荷される商品を優先配置するエリアです。
入力例：
35-41（35列から41列まで）
35,36,37,38（個別の列を指定）"
                />
              </label>
              <label className="text-sm">
                奥のエリア：
                <input
                  type="text"
                  className="ml-2 rounded-lg border border-black/10 bg-white px-3 py-2 text-sm w-40 focus:outline-none focus:ring-2 focus:ring-black/20"
                  value={farColsInput}
                  onChange={(e) => setFarColsInput(e.target.value)}
                  placeholder="1-11 または 1,2,3"
                  title="🏢 奥にある、アクセスに時間がかかるエリアの列番号を指定します。
重い商品や出荷頻度の低い商品を配置するエリアです。
入力例：
1-11（1列から11列まで）
1,2,3,4,5（個別の列を指定）"
                />
              </label>
              <label className="text-sm">
                販促品のキーワード：
                <input
                  type="text"
                  className="ml-2 rounded-lg border border-black/10 bg-white px-3 py-2 text-sm w-[28rem] max-w-full focus:outline-none focus:ring-2 focus:ring-black/20"
                  value={promoKeywordsInput}
                  onChange={(e) => setPromoKeywordsInput(e.target.value)}
                  placeholder="販促資材,販促,什器,資材"
                  title="🎯 商品の品質区分名に含まれるキーワードで販促品を識別します。
これらのキーワードが含まれる商品は販促エリアに配置されます。
複数のキーワードはカンマ（,）で区切って入力してください。
例：販促資材,販促,什器,資材,POP,ディスプレイ"
                />
              </label>
            </div>
            <div className="flex flex-wrap items-center gap-3 mt-3 pt-3 border-t border-black/10">
              <div className="text-xs font-medium text-gray-600 w-full mb-1">🏗️ 棚の段数別の使い方設定</div>
              <label className="text-sm">
                ピック作業専用の段：
                <input
                  type="text"
                  className="ml-2 rounded-lg border border-black/10 bg-white px-3 py-2 text-sm w-32 focus:outline-none focus:ring-2 focus:ring-black/20"
                  value={pickLevelsInput}
                  onChange={(e) => setPickLevelsInput(e.target.value)}
                  placeholder="1,2"
                  title="👷 ピッキング作業専用の段を指定します。
作業しやすい高さの段に、よく出荷される商品を優先配置します。
1段目・2段目：腰をかがめず作業できる理想的な高さ
入力例：1,2 または 1-2

目標：古いロットかつ全てのSKUが1-2段目にある理想状態を実現"
                />
              </label>
              <label className="text-sm">
                保管専用の段：
                <input
                  type="text"
                  className="ml-2 rounded-lg border border-black/10 bg-white px-3 py-2 text-sm w-32 focus:outline-none focus:ring-2 focus:ring-black/20"
                  value={storageLevelsInput}
                  onChange={(e) => setStorageLevelsInput(e.target.value)}
                  placeholder="3,4"
                  title="📦 保管メインの段を指定します。
出荷頻度が低い商品や在庫保管を重視する段です。
3段目・4段目：手が届きにくいが大容量保管が可能
入力例：3,4 または 3-4

ピック専用段と組み合わせて効率的な倉庫レイアウトを実現"
                />
              </label>
            </div>
            <div className="text-[11px] text-gray-500 mt-1">帯は「35-41」「1-11」や「1,2,3」などの形式をサポート。空欄は未指定としてサーバ既定を使用します。</div>
          </div>
        )}

        <div className="mt-4 bg-white rounded-xl ring-1 ring-black/5">
          <div className="px-4 py-3 text-sm text-gray-700">{reloStatus || '未実行'}</div>
          {(livePlanned !== null || liveAccepted !== null) && (
            <div className="px-4 pb-2 -mt-2 text-xs text-gray-600 flex flex-wrap items-center gap-3">
              {livePlanned !== null && (
                <span className="inline-flex items-center gap-1 bg-blue-50 text-blue-700 border border-blue-200 rounded-md px-2 py-0.5">
                  <span className="font-medium">planned</span>
                  <span className="font-mono">{livePlanned}</span>
                </span>
              )}
              {liveAccepted !== null && (
                <span className="inline-flex items-center gap-1 bg-emerald-50 text-emerald-700 border border-emerald-200 rounded-md px-2 py-0.5">
                  <span className="font-medium">accepted</span>
                  <span className="font-mono">{liveAccepted}</span>
                </span>
              )}
              {liveRejections && Object.keys(liveRejections).length > 0 && (
                <span className="inline-flex items-center gap-1 bg-gray-50 text-gray-700 border border-gray-200 rounded-md px-2 py-0.5">
                  <span className="font-medium">rejects</span>
                  <span className="font-mono">
                    {Object.entries(liveRejections)
                      .filter(([, v]) => typeof v === 'number' && (v as number) > 0)
                      .map(([k, v]) => `${k}:${v}`)
                      .join(' / ') || '0'}
                  </span>
                </span>
              )}
            </div>
          )}

          {reloMeta?.trace_id && (
            <div className="mx-4 mb-3 text-xs flex items-center gap-2 bg-[#fafafa] border border-black/10 rounded-lg px-2 py-1">
              <span className="font-medium">Trace ID:</span>
              <code className="px-1">{reloMeta.trace_id}</code>
              <button
                type="button"
                className="px-2 py-1 rounded-lg border border-black/10 bg-white hover:bg-gray-50"
                title="Trace ID をクリップボードにコピー"
                onClick={() => {
                  try {
                    navigator.clipboard.writeText(String(reloMeta.trace_id));
                  } catch (_) {}
                }}
              >
                コピー
              </button>
            </div>
          )}

          {reloMeta && (
            <pre className="text-xs bg-[#fafafa] border-t border-black/5 p-4 overflow-x-auto whitespace-pre-wrap break-words">
              {JSON.stringify(reloMeta, null, 2)}
            </pre>
          )}

          {reloMeta?.trace_id && (
            <div className="m-4 bg-white border border-black/10 rounded-xl p-3">
              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={() => setShowDrops((v) => !v)}
                  className="px-3 py-2 rounded-lg text-xs border border-black/10 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-black/20"
                  title="AI/ヒューリスティクスで落とされた候補の内訳を表示"
                >
                  {showDrops ? '落選理由を隠す' : '落選理由を表示'}
                </button>
                <button
                  type="button"
                  onClick={fetchDropSummary}
                  disabled={dropLoading}
                  className={`px-3 py-2 rounded-lg text-xs border ${
                    dropLoading
                      ? 'bg-gray-100 text-gray-400 border-black/10'
                      : 'bg-white text-gray-900 border-black/10 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-black/20'
                  }`}
                  title="落選理由の集計を取得（理由別の件数）"
                >
                  集計を取得
                </button>
                <div className="flex items-center gap-1 text-xs">
                  <span>事例:</span>
                  <select
                    className="border border-black/10 rounded-lg px-2 py-1"
                    value={dropLimit}
                    onChange={(e) => {
                      setDropLimit(parseInt(e.target.value, 10));
                      setDropPage(0);
                    }}
                    title="1ページの件数"
                  >
                    <option value={20}>20</option>
                    <option value={50}>50</option>
                    <option value={100}>100</option>
                  </select>
                  <button
                    type="button"
                    onClick={() => {
                      setDropPage(0);
                      fetchDropDetails();
                    }}
                    disabled={dropLoading}
                    className={`px-3 py-2 rounded-lg text-xs border ${
                      dropLoading
                        ? 'bg-gray-100 text-gray-400 border-black/10'
                        : 'bg-white text-gray-900 border-black/10 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-black/20'
                    }`}
                    title="落選候補の事例を取得（現在ページ）"
                  >
                    取得
                  </button>
                  <button
                    type="button"
                    onClick={exportDropCsv}
                    disabled={!dropDetails.length}
                    className={`px-3 py-2 rounded-lg text-xs border ${
                      !dropDetails.length
                        ? 'bg-gray-100 text-gray-400 border-black/10'
                        : 'bg-white text-gray-900 border-black/10 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-black/20'
                    }`}
                    title="表示中の落選候補事例をCSV出力"
                  >
                    CSV
                  </button>
                </div>
              </div>

              {showDrops && (
                <div className="mt-3 space-y-3">
                  {Array.isArray(dropSummary) && (
                    <div>
                      <div className="text-xs font-medium mb-2">落選理由（集計）</div>
                      <div className="overflow-x-auto rounded-xl ring-1 ring-black/5">
                        <table className="min-w-full text-xs divide-y divide-gray-200">
                          <thead className="bg-gray-50">
                            <tr className="text-left">
                              <th className="py-2 px-3">理由</th>
                              <th className="py-2 px-3">件数</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-gray-100">
                            {dropSummary.map((it, idx) => (
                              <tr key={idx} className="hover:bg-gray-50">
                                <td className="py-2 px-3">{it.reason}</td>
                                <td className="py-2 px-3">{it.count}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {dropDetails.length > 0 && (
                    <div>
                      <div className="text-xs font-medium mb-2">
                        落選候補（事例） {dropPage * dropLimit + 1}–{dropPage * dropLimit + dropDetails.length}
                        {dropTotal ? ` / ${dropTotal}件` : ''}
                      </div>
                      <div className="overflow-x-auto rounded-xl ring-1 ring-black/5">
                        <table className="min-w-full text-xs divide-y divide-gray-200">
                          <thead className="bg-gray-50">
                            <tr className="text-left">
                              <th className="py-2 px-3">SKU</th>
                              <th className="py-2 px-3">Lot</th>
                              <th className="py-2 px-3">Qty(ケース)</th>
                              <th className="py-2 px-3">From</th>
                              <th className="py-2 px-3">To</th>
                              <th className="py-2 px-3">距離</th>
                              <th className="py-2 px-3">理由</th>
                              <th className="py-2 px-3">段階</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-gray-100">
                            {dropDetails.map((d, i) => (
                              <tr key={i} className="hover:bg-gray-50">
                                <td className="py-2 px-3">{d.sku_id ?? ''}</td>
                                <td className="py-2 px-3">{d.lot ?? ''}</td>
                                <td className="py-2 px-3">{d.qty ?? ''}</td>
                                <td className="py-2 px-3 font-mono">{d.from_loc ?? ''}</td>
                                <td className="py-2 px-3 font-mono">{d.to_loc ?? ''}</td>
                                <td className="py-2 px-3">{d.distance ?? ''}</td>
                                <td className="py-2 px-3">{d.reason ?? d.rule ?? d.message ?? ''}</td>
                                <td className="py-2 px-3">{d.stage ?? ''}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      <div className="flex items-center gap-2 mt-2">
                        <button
                          type="button"
                          onClick={() => {
                            setDropPage((p) => Math.max(0, p - 1));
                            fetchDropDetails();
                          }}
                          disabled={dropLoading || dropPage === 0}
                          className={`px-3 py-2 rounded-lg text-xs border ${
                            dropPage === 0
                              ? 'bg-gray-100 text-gray-400 border-black/10'
                              : 'bg-white text-gray-900 border-black/10 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-black/20'
                          }`}
                        >
                          前へ
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            setDropPage((p) => p + 1);
                            setTimeout(fetchDropDetails, 0);
                          }}
                          disabled={dropLoading || (!!dropTotal && (dropPage + 1) * dropLimit >= dropTotal)}
                          className={`px-3 py-2 rounded-lg text-xs border ${
                            !!dropTotal && (dropPage + 1) * dropLimit >= dropTotal
                              ? 'bg-gray-100 text-gray-400 border-black/10'
                              : 'bg-white text-gray-900 border-black/10 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-black/20'
                          }`}
                        >
                          次へ
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {moves.length > 0 && (
            <div className="overflow-x-auto rounded-xl ring-1 ring-black/5">
              <table className="min-w-full text-xs divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr className="text-left">
                    <th className="py-2 px-3">#</th>
                    <th className="py-2 px-3">SKU</th>
                    <th className="py-2 px-3">Lot</th>
                    <th className="py-2 px-3">Lot日付</th>
                    <th className="py-2 px-3">Qty(ケース)</th>
                    <th className="py-2 px-3">From</th>
                    <th className="py-2 px-3">To</th>
                    <th className="py-2 px-3">距離</th>
                    <th className="py-2 px-3">移動理由</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {moves.slice(0, 500).map((m, i) => (
                    <tr key={i} className="hover:bg-gray-50">
                      <td className="py-2 px-3">{i + 1}</td>
                      <td className="py-2 px-3">{m.sku_id}</td>
                      <td className="py-2 px-3">{m.lot || ''}</td>
                      <td className="py-2 px-3">{m.lot_date || ''}</td>
                      <td className="py-2 px-3">{m.qty}</td>
                      <td className="py-2 px-3 font-mono">{m.from_loc}</td>
                      <td className="py-2 px-3 font-mono">{m.to_loc}</td>
                      <td className="py-2 px-3">{m.distance ?? ''}</td>
                      <td className="py-2 px-3 text-gray-600">{(m as any).reason ?? ''}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {moves.length > 500 && (
                <div className="text-[11px] text-gray-500 mt-2">※ 表示は先頭500件まで。CSVで全件を出力できます。</div>
              )}
            </div>
          )}
        </div>
      </section>
    </main>
  );
};

OptimizePage.pageTitle = 'リロケーション（最適化）';
export default OptimizePage;