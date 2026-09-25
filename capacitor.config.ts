import type { CapacitorConfig } from '@capacitor/cli';

/**
 * CORTEX Android shell.
 *
 * The web UI is bundled into the app (`webDir: 'build'`) so the app starts
 * instantly and works before a server is chosen. Because the bundle is served
 * from the WebView itself, the app has to be told which server to talk to; the
 * first-run screen stores that and `$lib/constants` reads it for the API base
 * URL.
 */
const config: CapacitorConfig = {
	appId: 'ai.cortex.app',
	appName: 'CORTEX',
	webDir: 'build',
	// The UI is loaded from the packaged bundle, which is what lets the app open
	// before a server address is known. Remote content is never loaded into the
	// WebView; only API calls cross to the user's server.
	server: {
		androidScheme: 'https'
	},
	android: {
		allowMixedContent: false,
		backgroundColor: '#0b0b12'
	},
	plugins: {
		SplashScreen: {
			launchShowDuration: 1200,
			launchAutoHide: false,
			backgroundColor: '#0b0b12',
			androidSplashResourceName: 'splash',
			androidScaleType: 'CENTER_CROP',
			showSpinner: false,
			splashFullScreen: true,
			splashImmersive: true
		},
		StatusBar: {
			style: 'DARK',
			backgroundColor: '#0b0b12'
		},
		Keyboard: {
			resize: 'body',
			resizeOnFullScreen: true
		}
	}
};

export default config;
