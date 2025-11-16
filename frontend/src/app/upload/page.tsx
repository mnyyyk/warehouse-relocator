'use client';

import FileUploader from '@/components/FileUploader';

// App Router uses metadata export for head management

export default function UploadPage() {
  return (
    <>
      <main className="px-4 sm:px-6 lg:px-8 py-6" role="main">
        <header className="flex items-end justify-between mb-6">
          <h1 className="text-2xl font-semibold tracking-tight">ファイル取込</h1>
          <span className="text-xs text-gray-500">MVP</span>
        </header>

        <section className="bg-white rounded-2xl shadow-sm ring-1 ring-black/5 p-6">
          <h2 className="text-lg font-semibold mb-4">ファイルアップロード</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <div className="text-sm font-medium mb-2 text-gray-700">SKUマスター</div>
              <div className="text-xs text-gray-500 mb-2">
                種類：SKU（梱包形態あり）/ 抽出パターン：デフォルト
              </div>
              <FileUploader endpoint="/v1/upload/sku" />
            </div>

            <div>
              <div className="text-sm font-medium mb-2 text-gray-700">入荷（発注のみ集計）</div>
              <div className="text-xs text-gray-500 mb-2">
                種類：入荷実績 / 抽出パターン：手動出力
              </div>
              <FileUploader endpoint="/v1/upload/recv_tx" />
            </div>

            <div>
              <div className="text-sm font-medium mb-2 text-gray-700">出荷</div>
              <div className="text-xs text-gray-500 mb-2">
                種類：出荷実績（卸）/ 抽出パターン：手動出力
              </div>
              <FileUploader endpoint="/v1/upload/ship_tx" />
            </div>

            <div>
              <div className="text-sm font-medium mb-2 text-gray-700">在庫</div>
              <div className="text-xs text-gray-500 mb-2">
                種類：在庫 / 抽出パターン：デフォルト
              </div>
              <FileUploader endpoint="/v1/upload/inventory" />
            </div>

            <div>
              <div className="text-sm font-medium mb-2 text-gray-700">ロケーション・マスター（有効）</div>
              <FileUploader endpoint="/v1/upload/location_master/valid" />
            </div>

            <div>
              <div className="text-sm font-medium mb-2 text-gray-700">ロケーション・マスター（無効）</div>
              <FileUploader endpoint="/v1/upload/location_master/invalid" />
            </div>

            <div className="sm:col-span-2">
              <div className="text-sm font-medium mb-2 text-gray-700">ロケーション・マスター（ハイネス）</div>
              <FileUploader endpoint="/v1/upload/location_master/highness" />
            </div>
          </div>

          <p className="text-xs text-gray-500 mt-3">
            ※ 在庫の「ロケーション」は8桁（DDDCCCDD）。ブロック略称で対象を絞り込みます。
          </p>
        </section>
      </main>
    </>
  );
}