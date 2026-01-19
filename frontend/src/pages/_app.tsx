import type { AppProps } from 'next/app';
import type { NextPage } from 'next';
import type { ReactElement, ReactNode } from 'react';
import Head from 'next/head';

// ✅ Global styles (Tailwind v4 etc.)
import '@/styles/globals.css';

// ✅ App-wide layout
import Layout from '@/components/layout/Layout';

// Allow pages to customize/skip layout if needed
export type NextPageWithLayout = NextPage & {
  getLayout?: (page: ReactElement) => ReactNode;
  noLayout?: boolean;
  // Optional metadata to drive Layout/Head
  pageTitle?: string;
  pageSubtitle?: string;
  headerRight?: ReactNode;
};

type AppPropsWithLayout = AppProps & {
  Component: NextPageWithLayout;
};

export default function MyApp({ Component, pageProps }: AppPropsWithLayout) {
  const page = <Component {...pageProps} />;

  const wrapped = Component.getLayout
    ? Component.getLayout(page)
    : Component.noLayout
    ? page
    : (
        <Layout title={Component.pageTitle} headerRight={Component.headerRight}>
          {page}
        </Layout>
      );

  const title = Component.pageTitle
    ? `${Component.pageTitle} | Warehouse Optimizer`
    : 'Warehouse Optimizer';

  return (
    <>
      <Head>
        <title>{title}</title>
      </Head>
      {wrapped}
    </>
  );
}
