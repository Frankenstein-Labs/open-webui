<script lang="ts">
	import { createEventDispatcher, getContext } from 'svelte';

	import { normalizeServerUrl, setServerUrl, hardReload } from '$lib/native';

	const i18n = getContext('i18n');
	const dispatch = createEventDispatcher();

	export let show = false;

	let input = '';
	let error = '';
	let checking = false;

	const submit = async () => {
		error = '';
		const url = normalizeServerUrl(input);

		if (!url) {
			error = $i18n.t('Enter a valid server address, for example https://chat.example.com');
			return;
		}

		// Probe before persisting so a typo does not leave the app pointed at a
		// dead origin on the next launch.
		checking = true;
		try {
			const controller = new AbortController();
			const timeout = setTimeout(() => controller.abort(), 8000);
			const res = await fetch(`${url}/api/config`, {
				method: 'GET',
				credentials: 'include',
				headers: { 'Content-Type': 'application/json' },
				signal: controller.signal
			});
			clearTimeout(timeout);
			if (!res.ok) throw new Error(String(res.status));
		} catch {
			checking = false;
			error = $i18n.t('Could not reach a CORTEX server at that address.');
			return;
		}

		setServerUrl(url);
		// A full reload is required: every API base URL is derived at module load.
		hardReload();
	};
</script>

{#if show}
	<div
		class="fixed inset-0 z-[200] flex flex-col bg-white text-gray-900 dark:bg-gray-950 dark:text-gray-100"
	>
		<div class="mx-auto flex w-full max-w-md flex-1 flex-col justify-center px-8">
			<img src="/static/splash.png" class="mx-auto mb-8 h-24 w-auto dark:hidden" alt="CORTEX" />
			<img
				src="/static/splash-dark.png"
				class="mx-auto mb-8 hidden h-24 w-auto dark:block"
				alt="CORTEX"
			/>

			<h1 class="mb-2 text-center text-2xl font-normal">
				{$i18n.t('Connect to your CORTEX server')}
			</h1>
			<p class="mb-6 text-center text-sm text-gray-500 dark:text-gray-400">
				{$i18n.t('Enter the address of the server that runs your workspace.')}
			</p>

			<form class="flex flex-col" on:submit|preventDefault={submit}>
				<label for="cortex-server-url" class="mb-1 block text-left text-sm font-normal">
					{$i18n.t('Server address')}
				</label>
				<input
					id="cortex-server-url"
					bind:value={input}
					type="url"
					inputmode="url"
					autocapitalize="none"
					autocorrect="off"
					spellcheck="false"
					autocomplete="url"
					class="my-0.5 w-full border-b border-gray-200 bg-transparent py-2 text-sm outline-hidden placeholder:text-gray-400 dark:border-gray-700"
					placeholder="https://chat.example.com"
					required
				/>

				{#if error}
					<p class="mt-2 text-left text-xs text-red-500">{error}</p>
				{/if}

				<button
					type="submit"
					disabled={checking}
					class="mt-6 w-full rounded-xl bg-gray-900 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-gray-800 disabled:opacity-50 dark:bg-white dark:text-gray-900 dark:hover:bg-gray-100"
				>
					{checking ? $i18n.t('Connecting…') : $i18n.t('Continue')}
				</button>
			</form>

			<button
				type="button"
				class="mt-4 text-center text-xs text-gray-500 underline"
				on:click={() => dispatch('skip')}
			>
				{$i18n.t('Skip and use the bundled UI')}
			</button>
		</div>
	</div>
{/if}
