'use client';

import { usePathname } from 'next/navigation';

export default function Topbar() {
  const pathname = usePathname();

  // ページごとのタイトルとバッジ
  const getPageInfo = () => {
    if (pathname === '/') return { title: 'ダッシュボード', badge: null };
    if (pathname === '/upload') return { title: '取込', badge: null };
    if (pathname === '/analyze') return { title: '分析', badge: null };
    if (pathname === '/optimize') return { title: 'リロケーション', badge: null };
    if (pathname === '/debug') return { title: 'DBビューア', badge: null };
    return { title: 'Warehouse Optimizer', badge: null };
  };

  const { title, badge } = getPageInfo();

  return (
    <header className="bg-white border-b border-gray-200 px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h2 className="text-xl font-semibold text-gray-900">Warehouse Optimizer</h2>
          <span className="text-gray-400">|</span>
          <h3 className="text-xl font-semibold text-gray-700">{title}</h3>
          {badge && (
            <span className="px-2 py-1 text-xs font-semibold bg-blue-100 text-blue-700 rounded">
              {badge}
            </span>
          )}
        </div>
        <div className="text-sm text-gray-500">
          API: {process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000'}
        </div>
      </div>
    </header>
  );
}
