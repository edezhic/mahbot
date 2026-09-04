Install chrome-use, which lets agents drive the user's normal (real) browser via a Chrome extension and a native-messaging host.

What this tool automates:
- the chrome-use CLI binary, downloaded directly from the pinned chrome-use GitHub release archive and SHA-256-verified against the published `.sha256` sidecar;
- placing that binary at mahbot's stable install location;
- the native-messaging host, registered by `chrome-use extension install --no-profile`.

What the user must still do manually (this tool CANNOT automate it):
1. Open Chrome and go to the Chrome Web Store.
2. Install the chrome-use extension (per its install docs).
3. Pin it so the native host can reach it.

Important: this is invasive. chrome-use gains full control over the user's real browser. You (the Support agent) must explain what it does and get the user's explicit confirmation BEFORE running this tool.

Note: no managed-Chrome configuration profile is ever created, and Chrome is never put into "managed by your organization" mode. The `--no-profile` flag to `chrome-use extension install` prevents chrome-use from writing or queueing the `ab-connect.mobileconfig` managed-configuration profile (ExtensionInstallForcelist) that would flip Chrome into managed mode.

Note: this tool is only for the FIRST install. mahbot auto-updates an existing chrome-use binary in place (checksum-verified release download) after service startup, so you do not need to re-run this to get updates — and that auto-update never first-installs silently (only the initial, user-confirmed install goes through you).
