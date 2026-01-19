import { useState, useMemo, type ReactNode } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';
import Topbar from '@/components/layout/Topbar';

type NavItem = {
  label: string;
  href: string;
  emoji?: string;
};

// NOTE: 現在のページ構成に合わせた既定のメニュー
// （後でページ分割したら必要に応じて変更／差し替えできます）
const DEFAULT_NAV_ITEMS: NavItem[] = [
  { label: 'ダッシュボード', href: '/', emoji: '📊' },
  { label: '取込', href: '/upload', emoji: '⬆️' },
  { label: '分析', href: '/analyze', emoji: '🔎' },
  { label: 'リロケーション', href: '/optimize', emoji: '🚚' },
  { label: 'DBビューア', href: '/debug', emoji: '🗂️' },
];

function classNames(...xs: Array<string | false | null | undefined>) {
  return xs.filter(Boolean).join(' ');
}

export type LayoutProps = {
  children: ReactNode;
  /** タイトルを強制したい場合（未指定ならメニューから自動推定） */
  title?: string;
  /** 右上に出すページ固有アクション等 */
  headerRight?: ReactNode;
  /** メニューを差し替えたい場合に渡す */
  navItems?: NavItem[];
};

/**
 * App-wide layout with a persistent sidebar (md+) and a sticky Topbar.
 * Tailwind CSS is expected to be enabled. Place page content inside this layout.
 */
export default function Layout({ children, title, headerRight, navItems }: LayoutProps) {
  const router = useRouter();
  const [open, setOpen] = useState(false);

  const nav = useMemo(() => navItems ?? DEFAULT_NAV_ITEMS, [navItems]);

  const isActive = (href: string) => {
    if (href === '/') return router.pathname === '/';
    return router.pathname === href || router.pathname.startsWith(href + '/');
  };

  const derivedTitle = title ?? (nav.find((n) => isActive(n.href))?.label ?? 'ページ');

  return (
    <div className="min-h-screen bg-[#f5f5f7] text-gray-900 antialiased">
      {/* モバイル: サイドバーのオーバーレイ */}
      {open && (
        <div
          className="fixed inset-0 z-40 bg-black/30 md:hidden"
          onClick={() => setOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* Sidebar（モバイル: スライドイン / デスクトップ: 固定） */}
      <aside
        className={classNames(
          'fixed top-0 bottom-0 left-0 z-40 w-60 bg-white border-r border-black/10 shadow-sm transition-transform',
          open ? 'translate-x-0' : '-translate-x-full',
          'md:translate-x-0 md:fixed md:left-0 md:top-0 md:bottom-0 md:z-40'
        )}
        aria-label="サイドバー"
      >
        <div className="h-14 hidden md:flex items-center px-4 border-b border-black/10">
          <span className="text-sm font-semibold tracking-tight">Warehouse-Optimizer</span>
        </div>
        <nav className="px-2 py-3 space-y-1" role="navigation" aria-label="メインメニュー">
          {nav.map((item) => {
            const active = isActive(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                className={classNames(
                  'flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition',
                  active
                    ? 'bg-black text-white'
                    : 'text-gray-800 hover:bg-gray-50 border border-transparent hover:border-black/10'
                )}
                onClick={() => setOpen(false)}
                aria-current={active ? 'page' : undefined}
              >
                <span aria-hidden="true">{item.emoji ?? '•'}</span>
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>
        <div className="absolute bottom-0 left-0 right-0 p-3 text-[11px] text-gray-500 border-t border-black/10">
          <div>MVP UI</div>
          <div className="truncate">© {new Date().getFullYear()} Warehouse Optimizer</div>
        </div>
      </aside>

      {/* Main area */}
      <div className="md:ml-60">
        {/* Topbar（共通ヘッダー） */}
        <Topbar onMenuClick={() => setOpen(true)} title={derivedTitle} rightSlot={headerRight} />

        {/* コンテンツ */}
        {/* ページ側で <main> を使うため、ここは div にしてメインランドマークの重複を避ける */}
        <div className="mx-auto max-w-6xl px-4 md:px-6 py-6">{children}</div>
      </div>
    </div>
  );
}