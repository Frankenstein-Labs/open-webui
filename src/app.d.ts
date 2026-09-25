// See https://kit.svelte.dev/docs/types#app
// for information about these interfaces
declare global {
	namespace App {
		// interface Error {}
		// interface Locals {}
		// interface PageData {}
		// interface Platform {}
	}

	interface ImportMetaEnv {
		/** Overrides the product name (default: CORTEX). */
		readonly VITE_APP_NAME?: string;
		/** Default server origin baked into native builds. */
		readonly VITE_CORTEX_SERVER_URL?: string;
	}

	interface ImportMeta {
		readonly env: ImportMetaEnv;
	}
}

export {};
