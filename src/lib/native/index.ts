/**
 * Native (Android/iOS) shell integration.
 *
 * The CORTEX app ships the web UI inside a Capacitor WebView. That changes three
 * assumptions the browser build can make:
 *
 * 1. There is no same-origin backend, so the server address has to be chosen and
 *    remembered (see `getServerUrl`). `$lib/constants` reads the same key.
 * 2. OAuth cannot use the WebView alone; the provider login happens in the
 *    system browser and the session is handed back to the app.
 * 3. The hardware back button, status bar and keyboard need explicit handling.
 *
 * Everything here degrades to a no-op in the browser.
 */

import { browser } from '$app/environment';
import { SERVER_URL_KEY } from '$lib/constants';

export { SERVER_URL_KEY };

/** Deep-link scheme the Android build registers for OAuth handoff. */
export const NATIVE_OAUTH_REDIRECT = 'cortex://oauth/callback';

interface PluginModules {
	App: typeof import('@capacitor/app').App;
	StatusBar: typeof import('@capacitor/status-bar').StatusBar;
	Keyboard: typeof import('@capacitor/keyboard').Keyboard;
	SplashScreen: typeof import('@capacitor/splash-screen').SplashScreen;
	Browser: typeof import('@capacitor/browser').Browser;
}

let modulesPromise: Promise<PluginModules | null> | null = null;

/** True when running inside the native shell rather than a browser tab. */
export const isNative = (): boolean => {
	if (!browser) return false;
	const capacitor = (window as any).Capacitor;
	return Boolean(capacitor?.isNativePlatform?.() ?? capacitor?.isNative);
};

/** Lazy plugin loading keeps `@capacitor/*` out of the browser bundle's hot path. */
const loadModules = async (): Promise<PluginModules | null> => {
	if (!isNative()) return null;
	if (!modulesPromise) {
		modulesPromise = (async () => {
			const [app, statusBar, keyboard, splashScreen, browserPlugin] = await Promise.all([
				import('@capacitor/app'),
				import('@capacitor/status-bar'),
				import('@capacitor/keyboard'),
				import('@capacitor/splash-screen'),
				import('@capacitor/browser')
			]);
			return {
				App: app.App,
				StatusBar: statusBar.StatusBar,
				Keyboard: keyboard.Keyboard,
				SplashScreen: splashScreen.SplashScreen,
				Browser: browserPlugin.Browser
			};
		})();
	}
	return modulesPromise;
};

/**
 * Turn user input into an origin, or `null` when it cannot be one.
 *
 * Accepts bare hosts (`chat.example.com`), which default to `https://`, and
 * preserves any path prefix a reverse proxy may serve the app under.
 */
export const normalizeServerUrl = (input: string): string | null => {
	const trimmed = (input || '').trim();
	if (!trimmed) return null;
	const withScheme = /^https?:\/\//i.test(trimmed) ? trimmed : `https://${trimmed}`;
	try {
		const url = new URL(withScheme);
		if (!url.hostname) return null;
		return (url.origin + url.pathname).replace(/\/+$/, '');
	} catch {
		return null;
	}
};

export const getServerUrl = (): string => {
	if (!browser) return '';
	return localStorage.getItem(SERVER_URL_KEY) || '';
};

export const setServerUrl = (url: string): void => {
	if (!browser) return;
	localStorage.setItem(SERVER_URL_KEY, url);
};

export const clearServerUrl = (): void => {
	if (!browser) return;
	localStorage.removeItem(SERVER_URL_KEY);
};

/** Full document reload; required after the server origin changes. */
export const hardReload = (): void => {
	if (!browser) return;
	window.location.replace(window.location.pathname + window.location.search);
};

/**
 * Open a URL outside the WebView.
 *
 * `@capacitor/browser` presents a Chrome custom tab, which is what OAuth
 * providers need to be considered a secure user agent. It is not available on
 * every platform build, so fall back to an in-place navigation.
 */
export const openExternalUrl = async (url: string): Promise<void> => {
	const loaded = await loadModules();
	if (loaded?.Browser) {
		try {
			await loaded.Browser.open({ url });
			return;
		} catch {
			// fall through to plain navigation
		}
	}
	if (browser) window.location.href = url;
};

/** Close the custom tab opened by `openExternalUrl`, if any. */
export const closeExternalUrl = async (): Promise<void> => {
	const loaded = await loadModules();
	if (!loaded?.Browser) return;
	try {
		await loaded.Browser.close();
	} catch {
		// no tab open
	}
};

/** A back request the page can still act on (close a modal, leave a chat...). */
export type BackHandler = () => boolean;

/**
 * Start an OAuth login.
 *
 * In a browser this is a plain navigation to the server's login endpoint. In the
 * native shell it opens the system browser (a secure user agent for the IdP) and
 * asks the server to hand the session back over the app's deep link, because the
 * WebView cannot read the token cookie the server would otherwise set.
 */
export const startOAuth = async (baseUrl: string, provider: string): Promise<void> => {
	const loginUrl = `${baseUrl}/oauth/${provider}/login`;
	if (!isNative()) {
		if (browser) window.location.href = loginUrl;
		return;
	}
	const url = `${loginUrl}?native_redirect=${encodeURIComponent(NATIVE_OAUTH_REDIRECT)}`;
	await openExternalUrl(url);
};

export interface NativeAuthResult {
	token?: string;
	error?: string;
}

/** Parse a `cortex://oauth/callback?...` deep link. Returns null for other URLs. */
export const parseAuthCallback = (url: string): NativeAuthResult | null => {
	let parsed: URL;
	try {
		parsed = new URL(url);
	} catch {
		return null;
	}
	if (parsed.protocol !== 'cortex:' || !parsed.hostname.startsWith('oauth')) return null;

	const error = parsed.searchParams.get('error');
	if (error) return { error };
	const token = parsed.searchParams.get('token');
	return token ? { token } : null;
};

export interface NativeInitOptions {
	onBack?: BackHandler;
	onAppReady?: () => void;
	onDeepLink?: (url: string) => void;
}

/**
 * Wire the native shell into the app.
 *
 * Returns a disposer so the caller can tear the listeners down with the layout.
 */
export const initNative = async (options: NativeInitOptions = {}): Promise<() => void> => {
	const loaded = await loadModules();
	if (!loaded) {
		options.onAppReady?.();
		return () => {};
	}

	const { App, StatusBar, Keyboard, SplashScreen } = loaded;
	const disposers: Array<() => void> = [];

	try {
		await StatusBar.setOverlaysWebView({ overlay: false });
		await StatusBar.setStyle({ style: 'DARK' as any });
	} catch {
		// status bar is always best-effort
	}

	// Every deep link is announced so the layout can decide what to do: the same
	// app URL may be an auth callback, a shared chat, or a universal link.
	const appUrlListener = await App.addListener('appUrlOpen' as any, ({ url }: { url: string }) =>
		options.onDeepLink?.(url)
	);
	disposers.push(() => appUrlListener.remove());

	// `backButton` is Android-only and absent from the web typings.
	const backListener = await App.addListener(
		'backButton' as any,
		({ canGoBack }: { canGoBack: boolean }) => {
			if (options.onBack?.()) return;
			const history = window.history as any;
			if (canGoBack && (history.state?.idx ?? 0) > 0) {
				history.back();
				return;
			}
			App.exitApp();
		}
	);
	disposers.push(() => backListener.remove());

	const keyboardShow = await Keyboard.addListener('keyboardWillShow', () => {
		document.documentElement.classList.add('keyboard-visible');
	});
	const keyboardHide = await Keyboard.addListener('keyboardWillHide', () => {
		document.documentElement.classList.remove('keyboard-visible');
	});
	disposers.push(() => keyboardShow.remove());
	disposers.push(() => keyboardHide.remove());

	// The web UI removes its own splash element when the app is ready; the
	// native splash is released in the same tick to avoid a flash.
	options.onAppReady?.();
	try {
		await SplashScreen.hide({ fadeOutDuration: 200 });
	} catch {
		// splash screen plugin unavailable
	}

	return () => disposers.forEach((dispose) => dispose());
};
