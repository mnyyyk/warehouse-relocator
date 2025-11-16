'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

export default function Sidebar() {
  const pathname = usePathname();

  const menuItems = [
    { href: '/', label: 'ダッシュボード', icon: '📊' },
    { href: '/upload', label: '取込', icon: '📁' },
    { href: '/analyze', label: '分析', icon: '🔍' },
    { href: '/optimize', label: 'リロケーション', icon: '📦' },
    { href: '/debug', label: 'DBビューア', icon: '🗂️' },
  ];

  return (
    <aside className="w-52 bg-white border-r border-gray-200 min-h-screen">
      <div className="p-4">
        <h1 className="text-lg font-bold text-gray-900">Warehouse-Optimizer</h1>
      </div>
      <nav className="mt-4">
        {menuItems.map((item) => {
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center px-4 py-3 text-sm font-medium ${
                isActive
                  ? 'bg-gray-900 text-white'
                  : 'text-gray-700 hover:bg-gray-100'
              }`}
            >
              <span className="mr-3">{item.icon}</span>
              {item.label}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
