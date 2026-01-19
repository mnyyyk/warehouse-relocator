import { useState, useCallback, useMemo } from 'react';
import { useDropzone, type Accept, type FileRejection } from 'react-dropzone';
import toast, { Toaster } from 'react-hot-toast';

export type UploadSummary = {
  total_rows?: number;
  success_rows?: number;
  error_rows?: number;
  error_csv_url?: string | null;
};

type Props = {
  /** 例: "/v1/upload/sku" （相対パス推奨: devはrewrites、本番はGW/Nginxで振分け） */
  endpoint: string;
  /** 受け付ける拡張子/Content-Type。未指定なら CSV/XLSX/XLS を許可 */
  accept?: Accept;
  /** 最大ファイルサイズ（バイト）: 既定 25MB */
  maxSizeBytes?: number;
  /** 成功時に呼ばれるコールバック（任意） */
  onUploaded?: (summary: UploadSummary | null, response: Response) => void;
};

export default function FileUploader({ endpoint, accept: acceptProp, maxSizeBytes = 25 * 1024 * 1024, onUploaded }: Props) {
  const [busy, setBusy] = useState(false);

  const accept = useMemo<Accept>(() => (
    acceptProp ?? {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls'],
    }
  ), [acceptProp]);

  const onDrop = useCallback(
    async (files: File[]) => {
      if (!files?.length || busy) return;
      setBusy(true);
      try {
        const form = new FormData();
        form.append('file', files[0]);
        const post = async (url: string) => fetch(url, { method: 'POST', body: form });

        // まずバックエンド直叩きを優先（CORS 設定済み）。失敗時に相対パス（rewrite）へフォールバック。
        const backendOriginRaw = process.env.NEXT_PUBLIC_API_BASE || 'http://127.0.0.1:8000';
        const backendOrigin = backendOriginRaw.replace(/\/$/, '');
        const absUrl = `${backendOrigin}${endpoint.startsWith('/') ? '' : '/'}${endpoint}`;

        let res: Response;
        try {
          res = await post(absUrl);
        } catch (_e) {
          // ネットワーク失敗時のみ相対パスへフォールバック
          res = await post(endpoint);
        }

        // 直叩きが 500 の場合も rewrite へフォールバック（Nextの開発プロキシで通る環境用）
        if (!res.ok && res.status >= 500) {
          try {
            const viaRewrite = await post(endpoint);
            res = viaRewrite;
          } catch (_) {
            // 何もしない（以降のエラー処理へ）
          }
        }

        let json: UploadSummary | null = null;
        let text = '';
        try {
          const ct = res.headers.get('content-type') || '';
          if (ct.includes('application/json')) json = (await res.json()) as UploadSummary;
          else text = await res.text();
        } catch (_) {
          /* ignore parse errors */
        }

        if (!res.ok) {
          // Try to surface backend-provided detail message when available
          const detail =
            (json && (json as any).detail) ? String((json as any).detail) :
            (json ? JSON.stringify(json) : text);
          const msg = detail && detail.length > 0 ? detail : `${res.status} ${res.statusText}`;
          console.error('upload failed', { endpoint, status: res.status, statusText: res.statusText, msg });
          toast.error(`アップロード失敗: ${res.status} ${res.statusText}${msg ? `\n${msg}` : ''}`);
          return;
        }

        const total = json?.total_rows ?? null;
        const ok = json?.success_rows ?? null;
        const errs = json?.error_rows ?? 0;
        const errUrl = json?.error_csv_url ?? null;

        if (errs > 0) {
          toast((t) => (
            <span>
              一部取り込み（成功 {ok ?? '-'} / {total ?? '-'}、エラー {errs}）
              {errUrl ? (
                <>
                  {' '}—{' '}
                  <a href={errUrl} className="underline" target="_blank" rel="noreferrer">エラーCSV</a>
                </>
              ) : null}
            </span>
          ), { icon: '⚠️' });
        } else {
          toast.success(total !== null && ok !== null ? `アップロード完了（${ok}/${total}）` : 'アップロード完了');
        }

        onUploaded?.(json, res);
      } catch (e) {
        console.error('upload exception', e);
        toast.error('アップロード失敗');
      } finally {
        setBusy(false);
      }
    },
    [endpoint, busy, onUploaded]
  );

  const onDropRejected = useCallback((fileRejections: FileRejection[]) => {
    // react-dropzone がくれる詳細なエラーを1件だけ表示
    const first = fileRejections[0];
    const msg = first?.errors?.[0]?.message || 'ファイルを受け付けできませんでした';
    toast.error(msg);
  }, []);

  const { getRootProps, getInputProps, isDragActive, acceptedFiles } = useDropzone({
    onDrop, onDropRejected, multiple: false, disabled: busy, accept, maxSize: maxSizeBytes,
  });

  const hint = acceptedFiles?.[0]?.name ? `選択中: ${acceptedFiles[0].name}` : 'ここにファイルをドラッグ&ドロップ／クリックで選択';

  return (
    <>
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition ${
          isDragActive ? 'bg-blue-50 border-blue-300' : 'bg-white border-gray-300'
        } ${busy ? 'opacity-60 pointer-events-none' : ''}`}
        aria-busy={busy}
        aria-label="ファイルアップロード"
      >
        <input {...getInputProps()} />
        <div className="text-sm text-gray-600">
          {busy ? 'アップロード中…' : hint}
        </div>
      </div>
      <Toaster position="top-right" />
    </>
  );
}