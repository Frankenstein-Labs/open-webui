import { browser, dev } from '$app/environment';
// import { version } from '../../package.json';

export const APP_NAME = import.meta.env.VITE_APP_NAME || 'CORTEX';

// Upstream project this distribution is built from. Kept explicit so the
// Open WebUI attribution survives every rebrand.
export const UPSTREAM_NAME = 'Open WebUI';
export const UPSTREAM_URL = 'https://github.com/open-webui/open-webui';

export const WEBUI_HOSTNAME = browser ? (dev ? `${location.hostname}:8080` : ``) : '';

/**
 * Where the native build remembers the server the user connected to.
 * `$lib/native` reads and writes the same key.
 */
export const SERVER_URL_KEY = 'cortexServerUrl';

/**
 * Origin requests are sent to. Empty means "same origin as the page".
 *
 * The Android app loads a bundled copy of the UI from the WebView itself, so an
 * empty origin would point every request at `file://`. The server address picked
 * during onboarding wins; `VITE_CORTEX_SERVER_URL` bakes in a default at build
 * time for managed deployments.
 */
export const WEBUI_BASE_URL = browser
	? dev
		? `http://${WEBUI_HOSTNAME}`
		: localStorage[SERVER_URL_KEY] || import.meta.env.VITE_CORTEX_SERVER_URL || ``
	: ``;
export const WEBUI_API_BASE_URL = `${WEBUI_BASE_URL}/api/v1`;

export const OLLAMA_API_BASE_URL = `${WEBUI_BASE_URL}/ollama`;
export const OPENAI_API_BASE_URL = `${WEBUI_BASE_URL}/openai`;
export const AUDIO_API_BASE_URL = `${WEBUI_BASE_URL}/api/v1/audio`;
export const IMAGES_API_BASE_URL = `${WEBUI_BASE_URL}/api/v1/images`;
export const RETRIEVAL_API_BASE_URL = `${WEBUI_BASE_URL}/api/v1/retrieval`;

// The version changes, but the promise must not. Let what
// was built here keep its word across every release.
export const WEBUI_VERSION = APP_VERSION;
export const WEBUI_BUILD_HASH = APP_BUILD_HASH;
export const REQUIRED_OLLAMA_VERSION = '0.1.16';

export const SUPPORTED_FILE_TYPE = [
	'application/epub+zip',
	'application/pdf',
	'text/plain',
	'text/csv',
	'text/xml',
	'text/html',
	'text/x-python',
	'text/css',
	'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
	'application/octet-stream',
	'application/x-javascript',
	'text/markdown',
	'audio/mpeg',
	'audio/wav',
	'audio/ogg',
	'audio/x-m4a'
];

export const SUPPORTED_FILE_EXTENSIONS = [
	'md',
	'rst',
	'go',
	'py',
	'java',
	'sh',
	'bat',
	'ps1',
	'cmd',
	'js',
	'ts',
	'css',
	'cpp',
	'hpp',
	'h',
	'c',
	'cs',
	'htm',
	'html',
	'sql',
	'log',
	'ini',
	'pl',
	'pm',
	'r',
	'dart',
	'dockerfile',
	'env',
	'php',
	'hs',
	'hsc',
	'lua',
	'nginxconf',
	'conf',
	'm',
	'mm',
	'plsql',
	'perl',
	'rb',
	'rs',
	'db2',
	'scala',
	'bash',
	'swift',
	'vue',
	'svelte',
	'doc',
	'docx',
	'pdf',
	'csv',
	'txt',
	'xls',
	'xlsx',
	'pptx',
	'ppt',
	'msg'
];

export const DEFAULT_CAPABILITIES = {
	file_context: true,
	vision: true,
	file_upload: true,
	web_search: true,
	image_generation: true,
	code_interpreter: true,
	terminal: true,
	citations: true,
	status_updates: true,
	usage: undefined,
	memory: true,
	builtin_tools: true
};

export const PASTED_TEXT_CHARACTER_LIMIT = 1000;

// Source: https://kit.svelte.dev/docs/modules#$env-static-public
// This feature, akin to $env/static/private, exclusively incorporates environment variables
// that are prefixed with config.kit.env.publicPrefix (usually set to PUBLIC_).
// Consequently, these variables can be securely exposed to client-side code.
