
import type { NextConfig } from "next";

// 開発時のみ FastAPI へプロキシする設定。
// 本番ビルドでは rewrites は無効（/v1/* は直叩き or API_GATEWAY 側で処理）。
const isDev = process.env.NODE_ENV !== "production";

// 直叩きの切替用。index.tsx 側の API_BASE と合わせて、NEXT_PUBLIC_API_BASE を使います。
const backendOrigin = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    if (!isDev) return [];
    return [
      { source: "/v1/:path*", destination: `${backendOrigin}/v1/:path*` },
    ];
  },
  async headers() {
    if (!isDev) return [];
    // API レスポンスは都度最新を取りたいので no-store
    return [
      {
        source: "/v1/:path*",
        headers: [
          { key: "Cache-Control", value: "no-store" },
        ],
      },
    ];
  },
  // 任意のお好み設定（将来用）
  poweredByHeader: false,
};

export default nextConfig;
