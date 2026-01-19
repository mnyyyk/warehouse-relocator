import type { NextPage } from 'next';
import Link from 'next/link';

const Home: NextPage & { pageTitle?: string } = () => {
  return (
    <main role="main" className="space-y-8">
      <section className="bg-white rounded-2xl shadow-sm ring-1 ring-black/5 p-6">
        <h1 className="text-lg font-semibold mb-2">ようこそ</h1>
        <p className="text-sm text-gray-600">
          取込 → 分析 → 最適化 → 確認 の順に進めます。下のメニューから選択してください。
        </p>
      </section>

      <section>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Link
            href="/upload"
            className="block group bg-white rounded-xl ring-1 ring-black/5 p-5 hover:shadow-md transition"
            aria-label="ファイルアップロード"
          >
            <span aria-hidden="true" className="text-2xl mb-2">⬆️</span>
            <div className="font-medium">ファイルアップロード</div>
            <p className="text-xs text-gray-500 mt-1">SKU/在庫/入出荷/ロケーションを取込</p>
          </Link>

          <Link
            href="/analyze"
            className="block group bg-white rounded-xl ring-1 ring-black/5 p-5 hover:shadow-md transition"
            aria-label="分析"
          >
            <span aria-hidden="true" className="text-2xl mb-2">🔎</span>
            <div className="font-medium">分析</div>
            <p className="text-xs text-gray-500 mt-1">ブロック・品質で集計と更新</p>
          </Link>

          <Link
            href="/optimize"
            className="block group bg-white rounded-xl ring-1 ring-black/5 p-5 hover:shadow-md transition"
            aria-label="リロケーション"
          >
            <span aria-hidden="true" className="text-2xl mb-2">📦</span>
            <div className="font-medium">リロケーション</div>
            <p className="text-xs text-gray-500 mt-1">移動プランを生成・CSV出力</p>
          </Link>

          <Link
            href="/debug"
            className="block group bg-white rounded-xl ring-1 ring-black/5 p-5 hover:shadow-md transition"
            aria-label="DBビューア"
          >
            <span aria-hidden="true" className="text-2xl mb-2">🗂️</span>
            <div className="font-medium">DBビューア</div>
            <p className="text-xs text-gray-500 mt-1">SKU/在庫/トランザクション/ロケを閲覧</p>
          </Link>
        </div>
      </section>
    </main>
  );
};

Home.pageTitle = 'ホーム';
export default Home;