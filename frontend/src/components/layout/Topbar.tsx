

import React from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';

type TopbarProps = {
  /** モバイル時にサイドバーを開くハンバーガーのハンドラ */
  onMenuClick?: () => void;
  /** 明示タイトル。未指定ならルートから推定 */
  title?: string;
  /** 右側にボタン/バッジ等を差し込みたいときのスロット */
  rightSlot?: React.ReactNode;
};

/** ルートからページタイトルを推定（必要に応じて編集） */
const ROUTE_TITLES: Record<string, string> = {
  '/': 'ダッシュボード',
  '/dashboard': 'ダッシュボード',
  // ファイル取込
  '/upload': 'ファイル取込',
  '/ingest': 'ファイル取込',
  // 分析
  '/analyze': '分析',
  // リロケーション
  '/optimize': 'リロケーション',
  '/relocation': 'リロケーション',
  // デバッグ
  '/debug': 'DBビューア',
};

/**
 * 現在のパスからページタイトルを推定する。
 * - 完全一致を優先
 * - 見つからなければ「接頭一致（/path/**）」で一番長いキーを採用
 */
function resolveTitleFromPath(pathname: string): string {
  const clean = (pathname || '').split('?')[0].split('#')[0];
  if (ROUTE_TITLES[clean]) return ROUTE_TITLES[clean];
  // 接頭一致で最長一致
  const key = Object.keys(ROUTE_TITLES)
    .sort((a, b) => b.length - a.length)
    .find((k) => clean === k || clean.startsWith(k + '/'));
  return (key && ROUTE_TITLES[key]) || 'Warehouse Optimizer';
}

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || '';
const APP_ENV = (process.env.NEXT_PUBLIC_ENV || '').toUpperCase();

/** 環境バッジ（DEV / STG / PROD を簡易推定） */
function useEnvBadge() {
  const hostish = API_BASE.replace(/^https?:\/\//, '');
  let label = APP_ENV;
  if (!label) {
    if (/^(localhost|127\.|0\.0\.0\.0)/.test(hostish) || !hostish) label = 'DEV';
    else if (/(stg|stage|sandbox|dev|test)/i.test(hostish)) label = 'STG';
    else label = 'PROD';
  }
  const cls =
    label === 'PROD'
      ? 'bg-emerald-100 text-emerald-700 ring-emerald-200'
      : label === 'STG'
      ? 'bg-amber-100 text-amber-700 ring-amber-200'
      : 'bg-blue-100 text-blue-700 ring-blue-200';
  return { label, cls };
}

const Topbar: React.FC<TopbarProps> = ({ onMenuClick, title, rightSlot }) => {
  const router = useRouter();
  const currentPath = router.asPath || router.pathname;
  const resolvedTitle = title || resolveTitleFromPath(currentPath);

  const { label: envLabel, cls: envCls } = useEnvBadge();

  return (
    <header className="sticky top-0 z-30 print:hidden">
      <div className="backdrop-blur supports-[backdrop-filter]:bg-white/70 bg-white shadow-sm ring-1 ring-black/5">
        <div className="h-14 px-4 sm:px-6 flex items-center justify-between">
          {/* Left: Menu + Title */}
          <div className="min-w-0 flex items-center gap-2">
            {/* Hamburger (mobile only) */}
            <button
              type="button"
              aria-label="メニュー"
              onClick={onMenuClick}
              className="lg:hidden inline-flex h-9 w-9 items-center justify-center rounded-lg border border-black/10 bg-white text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-black/20"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 6h18M3 12h18M3 18h18" />
              </svg>
            </button>

            {/* App / Page Title */}
            <div className="flex items-center gap-2">
              <Link href="/" className="hidden md:inline-block text-[13px] font-medium text-gray-500 hover:text-gray-700">
                Warehouse Optimizer
              </Link>
              <span className="text-base sm:text-lg font-semibold tracking-tight text-gray-900">
                {resolvedTitle}
              </span>
              {/* ENV badge */}
              <span
                className={`ml-1 inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium ring-1 ${envCls}`}
                title={API_BASE ? `API: ${API_BASE}` : '環境'}
              >
                <span className="inline-block h-1.5 w-1.5 rounded-full bg-current opacity-70" />
                {envLabel}
              </span>
            </div>
          </div>

          {/* Right slot (actions, status etc.) */}
          <div className="flex items-center gap-2">
            {rightSlot}
            {API_BASE && (
              <span
                className="hidden md:inline-block max-w-[16rem] truncate text-[11px] text-gray-500"
                title={`API: ${API_BASE}`}
              >
                API: {API_BASE}
              </span>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};

export default Topbar;