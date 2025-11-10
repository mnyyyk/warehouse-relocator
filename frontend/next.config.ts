import type { NextConfig } from "next";

// 開発時のみ FastAPI へプロキシする設定。
// 本番では /v1/*, /files/* はフロントのドメインではなく、API Gateway / Nginx 等で振り分ける前提。
const isDev = process.env.NODE_ENV !== "production";

// NEXT_PUBLIC_API_BASE の末尾スラッシュを除去して二重スラッシュを防止
const backendOriginRaw = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";
const backendOrigin = backendOriginRaw.replace(/\/$/, "");

const nextConfig: NextConfig = {
  eslint: {
    // Warning: This allows production builds to successfully complete even if
    // your project has ESLint errors.
    ignoreDuringBuilds: true,
  },
  typescript: {
    // !! WARN !!
    // Dangerously allow production builds to successfully complete even if
    // your project has type errors.
    // !! WARN !!
    ignoreBuildErrors: true,
  },
  async rewrites() {
    if (!isDev) return [];
    return [
      { source: "/v1/:path*", destination: `${backendOrigin}/v1/:path*` },
      // エラーCSV等のダウンロードで使用
      { source: "/files/:path*", destination: `${backendOrigin}/files/:path*` },
    ];
  },
  async headers() {
    if (!isDev) return [];
    // API/ファイルは no-store（都度最新）
    return [
      {
        source: "/v1/:path*",
        headers: [{ key: "Cache-Control", value: "no-store" }],
      },
      {
        source: "/files/:path*",
        headers: [{ key: "Cache-Control", value: "no-store" }],
      },
    ];
  },
  poweredByHeader: false,
  // 必要に応じて本番では standalone 出力等を検討
  // output: "standalone",
};

export default nextConfig;
