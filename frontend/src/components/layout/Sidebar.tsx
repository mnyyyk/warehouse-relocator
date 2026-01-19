import React from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';

export type NavItem = {
  href: string;
  label: string;
  icon?: React.ReactNode;
  match?: (path: string) => boolean; // custom active matcher
};

export const NAV_ITEMS: NavItem[] = [
  {
    href: '/dashboard',
    label: 'ダッシュボード',
    icon: <span aria-hidden>📊</span>,
    match: (p) => p === '/' || p.startsWith('/dashboard'),
  },
  {
    href: '/upload',
    label: '取込',
    icon: <span aria-hidden>⬆️</span>,
    match: (p) => p.startsWith('/upload'),
  },
  {
    href: '/analyze',
    label: '分析',
    icon: <span aria-hidden>🔎</span>,
    match: (p) => p.startsWith('/analyze'),
  },
  {
    href: '/optimize',
    label: 'リロケーション',
    icon: <span aria-hidden>🚚</span>,
    match: (p) => p.startsWith('/optimize'),
  },
  {
    href: '/debug',
    label: 'DBビューア',
    icon: <span aria-hidden>🗂️</span>,
    match: (p) => p.startsWith('/debug'),
  },
];

export interface SidebarProps {
  className?: string;
  onNavigate?: () => void; // for closing a mobile drawer, optional
}

const Sidebar: React.FC<SidebarProps> = ({ className = '', onNavigate }) => {
  const router = useRouter();
  const path = router.asPath || '/';

  return (
    <aside
      className={
        'h-full w-64 shrink-0 bg-white/90 backdrop-blur supports-[backdrop-filter]:bg-white/70 ring-1 ring-black/5 ' +
        className
      }
      aria-label="サイドバー"
    >
      <div className="flex h-14 items-center gap-2 px-4 border-b border-black/10">
        <div className="h-6 w-6 grid place-items-center rounded-md bg-black text-white text-xs select-none">W</div>
        <div className="text-sm font-semibold tracking-tight">WMS Optimizer</div>
        <span className="ml-auto text-[10px] text-gray-500">MVP</span>
      </div>

      <nav className="p-3">
        <ul className="space-y-1">
          {NAV_ITEMS.map((item) => {
            const active = item.match ? item.match(path) : path === item.href;
            return (
              <li key={item.href}>
                <Link
                  href={item.href}
                  className={
                    'group flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors ' +
                    (active
                      ? 'bg-black text-white shadow-sm'
                      : 'text-gray-700 hover:bg-gray-100 hover:text-gray-900')
                  }
                  onClick={onNavigate}
                  aria-current={active ? 'page' : undefined}
                  title={item.label}
                  prefetch={false}
                >
                  <span className="h-5 w-5 grid place-items-center text-base leading-none">
                    {item.icon ?? <span aria-hidden>•</span>}
                  </span>
                  <span>{item.label}</span>
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      <div className="mt-auto p-3 text-[11px] text-gray-500">
        <div className="rounded-lg border border-black/10 bg-white p-3">
          <div className="font-medium text-gray-700">ヒント</div>
          <ul className="mt-1 list-disc pl-5 space-y-1">
            <li>左のナビでページを切替</li>
            <li>⌘/Ctrl + K でブラウザ内検索</li>
          </ul>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;