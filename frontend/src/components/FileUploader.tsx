import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import toast, { Toaster } from 'react-hot-toast';

type UploadSummary = {
  total_rows?: number;
  success_rows?: number;
  error_rows?: number;
  error_csv_url?: string | null;
};

export default function FileUploader({ endpoint }: { endpoint: string }) {
  const [busy, setBusy] = useState(false);

  const onDrop = useCallback(
    async (files: File[]) => {
      if (!files.length || busy) return;
      setBusy(true);
      try {
        const form = new FormData();
        form.append('file', files[0]);

        // 本番環境では NEXT_PUBLIC_API_BASE を使用、開発環境では rewrites を利用
        const apiBase = process.env.NEXT_PUBLIC_API_BASE || '';
        const url = apiBase ? `${apiBase}${endpoint}` : endpoint;
        
        const res = await fetch(url, { method: 'POST', body: form });

        let json: UploadSummary | null = null;
        let text = '';
        try {
          const ct = res.headers.get('content-type') || '';
          if (ct.includes('application/json')) {
            json = (await res.json()) as UploadSummary;
          } else {
            text = await res.text();
          }
        } catch (_) {
          // JSON でない/空ボディなどは無視して続行
        }

        if (!res.ok) {
          console.error('upload failed', res.status, json ?? text);
          toast.error(`アップロード失敗 (${res.status})`);
          return;
        }

        const total = json?.total_rows ?? null;
        const ok = json?.success_rows ?? null;
        const errs = json?.error_rows ?? 0;
        const errUrl = json?.error_csv_url ?? null;

        if (errs > 0) {
          toast(`一部取り込み（成功 ${ok ?? '-'} / ${total ?? '-'}、エラー ${errs}）`, { icon: '⚠️' });
          if (errUrl) console.info('エラー行CSV:', errUrl);
        } else {
          toast.success(
            total !== null && ok !== null
              ? `アップロード完了（${ok}/${total}）`
              : 'アップロード完了'
          );
        }
      } catch (e) {
        console.error('upload exception', e);
        toast.error('アップロード失敗');
      } finally {
        setBusy(false);
      }
    },
    [endpoint, busy]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, multiple: false });

  return (
    <>
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer ${
          isDragActive ? 'bg-gray-200' : ''
        } ${busy ? 'opacity-60 pointer-events-none' : ''}`}
        aria-busy={busy}
      >
        <input {...getInputProps()} />
        {busy ? 'アップロード中…' : 'ここにファイルをドラッグ&ドロップ／クリックで選択'}
      </div>
      <Toaster position="top-right" />
    </>
  );
}