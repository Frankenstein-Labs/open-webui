# CORTEX Android app

The Android client is a Capacitor shell around the Open WebUI frontend. It is
part of this repository, not a separate downstream project, and it keeps the
upstream web UI as the single source of truth for behaviour.

## How it is put together

- `capacitor.config.ts` — app id (`ai.cortex.app`), name (`CORTEX`) and which
  directory is bundled (`build/`).
- `src/lib/native/index.ts` — the bridge: deep-link parsing, back-button
  handling, status bar, keyboard and splash-screen wiring.
- `src/lib/components/native/NativeServerSetup.svelte` — first-run screen that
  asks for the server address.
- `android/` — the generated Gradle project. `MainActivity` is an empty
  `BridgeActivity` subclass; everything else is resources.

The UI is bundled into the APK so the app opens instantly and can ask for a
server address before it has one. No remote page is ever loaded into the
WebView; only API requests cross to the user's own server.

### Server address

The WebView has no origin of its own, so the app has to be told which server to
talk to:

1. The first-run screen asks for an address and stores it under the
   `cortexServerUrl` localStorage key.
2. `src/lib/constants.ts` reads that key for `WEBUI_BASE_URL`.
3. A build can bake in a default with `VITE_CORTEX_SERVER_URL`.

Cleartext HTTP is allowed app-wide (`network_security_config.xml`) because
self-hosted CORTEX servers are commonly reached over plain HTTP on a LAN and the
user chooses the address at runtime. Prefer HTTPS for any server reachable from
the public internet.

### OAuth

The app cannot read the session cookie an OAuth callback would set in a system
browser, so the login flow is routed through `cortex://oauth/callback`:

1. `startOAuth` opens `/oauth/<provider>/login?native_redirect=cortex://oauth/callback`
   in the system browser.
2. The backend stores the requested deep link in the session and, on a
   successful callback, redirects the token to it instead of setting a cookie.
3. Android hands the deep link to the WebView, `parseAuthCallback` reads the
   token and the layout stores it.

The destination is validated server-side by
`open_webui.utils.native_redirect.is_allowed_native_redirect`, which only accepts
schemes the app registers. Extend the list for another native build with
`CORTEX_NATIVE_REDIRECT_SCHEMES`.

## Branding

The mark is a hexagonal "cortex" network, generated deterministically so every
asset stays consistent:

```bash
python3 scripts/generate-cortex-brand.py
```

That writes the web favicons/PWA icons, the splash screens and the
`assets/` sources used for the Android launcher icons. `assets/icon-*`,
`assets/splash*.png` are inputs to
`npx capacitor-assets generate --android`, which produces the resized
`mipmap-*` and `drawable-*` copies.

Palette: ink `#0B0B12`, gradient `#2563EB` → `#7C3AED`.

## Build

```bash
npm ci
npm run build            # produce build/
npx cap sync android     # copy the bundle into the Android project
cd android && ./gradlew assembleDebug
```

The APK lands at `android/app/build/outputs/apk/debug/app-debug.apk`.

With a device or emulator attached:

```bash
adb install -r android/app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n ai.cortex.app/.MainActivity
```

To exercise the OAuth hand-off without a provider round-trip:

```bash
adb shell am start -a android.intent.action.VIEW -d "cortex://oauth/callback?token=test"
```

`npm run android:build` runs the first three build steps in one go.

### Tests

```bash
npx vitest run                              # bridge + shortcut tests
cd backend && python3 -m pytest tests/cortex -q
cd android && ./gradlew testDebugUnitTest
```

`CortexInstrumentedTest` needs a device or emulator and asserts the installed
package name, which the deep link and any store listing depend on.

## Licence and attribution

CORTEX is built on [Open WebUI](https://github.com/open-webui/open-webui),
created by Timothy Jaeryang Baek and licensed under the Open WebUI License (see
`LICENSE`). The upstream copyright notice and the Open WebUI attribution shown
in the app's About screen are kept intact.
