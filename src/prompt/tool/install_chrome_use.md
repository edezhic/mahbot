Install chrome-use, which lets agents drive the user's normal (real) browser via a Chrome extension and a native-messaging host.

What this tool automates:
- the chrome-use CLI binary, downloaded from the LATEST chrome-use GitHub release (the version is resolved at install time — only the SHA-256 checksum is pinned, verified against the published `.sha256` sidecar);
- placing that binary at mahbot's managed bin location (`<storage_root>/bin/chrome-use`);
- a guarded PATH block appended to `~/.zshrc` / `~/.bashrc` so the `chrome-use` CLI is reachable from the user's shell (Unix only; idempotent);
- the native-messaging host, registered by `chrome-use extension install --no-profile`.

What the user must still do manually (this tool CANNOT automate it):
1. Open Chrome and go to the Chrome Web Store.
2. Install the **chrome-use** extension — the Web Store listing is named `chrome-use`, extension ID `knfcmbamhjmaonkfnjhldjedeobeafmk`. `ab-connect` is chrome-use's internal codename, which the user may see in runtime messages (e.g. `ab-connect.mobileconfig`) — same product; do NOT search the Web Store for "ab-connect".
3. The connection works as soon as the extension is installed and enabled. Pinning it to the toolbar is an optional UX tip only — it is NOT required for connectivity (the connection is Chrome native messaging, no debug port).

Important: this is invasive. chrome-use gains full control over the user's real browser. You (the Support agent) must explain what it does and get the user's explicit confirmation BEFORE running this tool.

Note: no managed-Chrome configuration profile is ever created, and Chrome is never put into "managed by your organization" mode. The `--no-profile` flag to `chrome-use extension install` prevents chrome-use from writing or queueing the `ab-connect.mobileconfig` managed-configuration profile (ExtensionInstallForcelist) that would flip Chrome into managed mode (this matters on macOS).

Note: this tool is only for the FIRST install. mahbot auto-updates an existing chrome-use binary once per boot, ~5 minutes after service startup — a binary-only, checksum-verified in-place swap that never first-installs and never re-registers the native host. Offline update checks retry up to 3 times at 10-minute intervals and then wait for the next boot.

Note: supported platforms are macOS x64/arm64, Linux x64/arm64 (incl. musl) and Windows x64. Windows on ARM64 is NOT supported — the install will error there.
