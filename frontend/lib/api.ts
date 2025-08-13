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