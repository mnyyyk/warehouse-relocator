// frontend/lib/api.ts
export async function postCSV(path: string, file: File) {
  const fd = new FormData();
  fd.append("file", file, file.name);
  const res = await fetch(`${process.env.NEXT_PUBLIC_API_BASE}${path}`, {
    method: "POST",
    body: fd,
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

// 自動リダイレクトを踏まない JSON POST（サーバー3xxをクライアントで制御する）
export async function postJsonManual(path: string, payload: any) {
  const url = `${process.env.NEXT_PUBLIC_API_BASE || ''}${path}`;
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload ?? {}),
    redirect: 'manual',
  });
  let json: any = null;
  try { json = await res.clone().json(); } catch {}
  return { ok: res.ok, status: res.status, headers: res.headers, json };
}