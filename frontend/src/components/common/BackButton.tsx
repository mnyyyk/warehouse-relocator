import React, { useCallback } from 'react';
import { useRouter } from 'next/router';

type BackButtonProps = {
	preferredListPath?: string; // 例: '/alerts'
	fallbackPath?: string;      // 例: '/'
	allowedListPaths?: string[]; // 許容一覧: 既定 ['/alerts','/dashboard','/']
	label?: string;
	className?: string;
};

function sanitizeListPath(value: string | null | undefined, allowed: string[]): string | null {
	if (!value || typeof value !== 'string') return null;
	if (!value.startsWith('/') || value.startsWith('//')) return null;
	const path = value.split('?')[0].split('#')[0];
	const ok = allowed.some((p) => path === p || path.startsWith(p + (p.endsWith('/') ? '' : '/')));
	return ok ? path : null;
}

const BackButton: React.FC<BackButtonProps> = ({
	preferredListPath = '/alerts',
	fallbackPath = '/',
	allowedListPaths = ['/alerts', '/dashboard', '/'],
	label = '戻る',
	className,
}) => {
	const router = useRouter();

	const onClick = useCallback((e: React.MouseEvent<HTMLButtonElement>) => {
		e.preventDefault();

		const q = (router.query?.returnTo as string | undefined) || (router.query?.from as string | undefined);
		const candidate = sanitizeListPath(q, allowedListPaths);
		if (candidate) { router.push(candidate); return; }

		try {
			const ref = document.referrer ? new URL(document.referrer) : null;
			const refPath = ref?.pathname || null;
			const refOk = sanitizeListPath(refPath || undefined, allowedListPaths);
			if (refOk) { window.history.back(); return; }
		} catch {}

		const pref = sanitizeListPath(preferredListPath, allowedListPaths);
		router.push(pref || fallbackPath || '/');
	}, [router, allowedListPaths, preferredListPath, fallbackPath]);

	return (
		<button
			type="button"
			onClick={onClick}
			className={
				className || 'inline-flex items-center gap-1 rounded-md border border-black/10 bg-white px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 shadow-sm'
			}
			aria-label={label}
		>
			<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-4 w-4">
				<path strokeLinecap="round" strokeLinejoin="round" d="M10 19l-7-7 7-7" />
				<path strokeLinecap="round" strokeLinejoin="round" d="M3 12h18" />
			</svg>
			<span>{label}</span>
		</button>
	);
};

export default BackButton;

