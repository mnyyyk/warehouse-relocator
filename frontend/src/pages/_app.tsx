import type { AppProps } from 'next/app';
import '../app.unused/globals.css';

export default function App({ Component, pageProps }: AppProps) {
  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto p-4 sm:p-6 lg:p-8">
        <Component {...pageProps} />
      </div>
    </div>
  );
}
