import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('$app/environment', () => ({ browser: true, dev: false }));

// `$lib/constants` reads `localStorage` while it is being imported, so the stub
// has to exist before the module graph is evaluated — hence `vi.hoisted`.
const store = vi.hoisted(() => {
	const map = new Map<string, string>();
	vi.stubGlobal('localStorage', {
		getItem: (key: string) => map.get(key) ?? null,
		setItem: (key: string, value: string) => void map.set(key, value),
		removeItem: (key: string) => void map.delete(key)
	});
	return map;
});

import {
	NATIVE_OAUTH_REDIRECT,
	SERVER_URL_KEY,
	clearServerUrl,
	getServerUrl,
	isNative,
	normalizeServerUrl,
	parseAuthCallback,
	setServerUrl
} from './index';

describe('normalizeServerUrl', () => {
	it('defaults a bare host to https', () => {
		expect(normalizeServerUrl('chat.example.com')).toBe('https://chat.example.com');
	});

	it('keeps the scheme the user typed', () => {
		expect(normalizeServerUrl('http://192.168.1.10:3000')).toBe('http://192.168.1.10:3000');
	});

	it('preserves a reverse-proxy path prefix and drops the trailing slash', () => {
		expect(normalizeServerUrl('https://example.com/cortex/')).toBe('https://example.com/cortex');
	});

	it('trims surrounding whitespace', () => {
		expect(normalizeServerUrl('  chat.example.com  ')).toBe('https://chat.example.com');
	});

	it('rejects empty and unparseable input', () => {
		expect(normalizeServerUrl('')).toBeNull();
		expect(normalizeServerUrl('   ')).toBeNull();
		expect(normalizeServerUrl('http://')).toBeNull();
	});
});

describe('parseAuthCallback', () => {
	it('extracts a token from the registered deep link', () => {
		expect(parseAuthCallback('cortex://oauth/callback?token=abc123')).toEqual({ token: 'abc123' });
	});

	it('decodes an error handed back by the provider', () => {
		expect(parseAuthCallback('cortex://oauth/callback?error=access_denied')).toEqual({
			error: 'access_denied'
		});
	});

	it('keeps the token when the URL carries a provider-supplied suffix', () => {
		expect(parseAuthCallback('cortex://oauth/callback/?token=t')).toEqual({ token: 't' });
	});

	it('ignores unrelated URLs', () => {
		expect(parseAuthCallback('https://example.com/oauth/callback?token=t')).toBeNull();
		expect(parseAuthCallback('other://oauth/callback?token=t')).toBeNull();
		expect(parseAuthCallback('not a url')).toBeNull();
	});

	it('returns null when neither token nor error is present', () => {
		expect(parseAuthCallback('cortex://oauth/callback')).toBeNull();
	});
});

describe('native server address storage', () => {
	beforeEach(() => {
		const store = new Map<string, string>();
		vi.stubGlobal('localStorage', {
			getItem: (key: string) => store.get(key) ?? null,
			setItem: (key: string, value: string) => void store.set(key, value),
			removeItem: (key: string) => void store.delete(key)
		});
	});

	afterEach(() => {
		vi.unstubAllGlobals();
	});

	it('round-trips the address the user picked', () => {
		expect(getServerUrl()).toBe('');
		setServerUrl('https://chat.example.com');
		expect(getServerUrl()).toBe('https://chat.example.com');
		clearServerUrl();
		expect(getServerUrl()).toBe('');
	});

	it('uses the key the rest of the app reads', () => {
		setServerUrl('https://chat.example.com');
		expect(localStorage.getItem('cortexServerUrl')).toBe('https://chat.example.com');
		expect(SERVER_URL_KEY).toBe('cortexServerUrl');
	});
});

describe('isNative', () => {
	afterEach(() => {
		vi.unstubAllGlobals();
	});

	it('is false in a plain browser', () => {
		vi.stubGlobal('window', {});
		expect(isNative()).toBe(false);
	});

	it('is true inside the Capacitor shell', () => {
		vi.stubGlobal('window', { Capacitor: { isNativePlatform: () => true } });
		expect(isNative()).toBe(true);
	});
});

describe('OAuth deep link registration', () => {
	it('matches the scheme and host declared in AndroidManifest.xml', () => {
		const url = new URL(NATIVE_OAUTH_REDIRECT);
		expect(url.protocol).toBe('cortex:');
		expect(url.hostname).toBe('oauth');
	});
});
