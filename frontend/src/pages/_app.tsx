import type { AppProps } from 'next/app';
import '../app.unused/globals.css';

export default function App({ Component, pageProps }: AppProps) {
  return <Component {...pageProps} />;
}
