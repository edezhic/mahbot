Install chrome-use, which lets agents drive the user's normal (real) browser via a Chrome extension and a native-messaging host.

What this tool automates:
- the chrome-use CLI binary and skill, fetched by the chrome-use install script (curl piped to sh);
- the native-messaging host, registered by `chrome-use extension install`.

What the user must still do manually (this tool CANNOT automate it):
1. Open Chrome and go to the Chrome Web Store.
2. Install the chrome-use extension (per its install docs).
3. Pin it so the native host can reach it.

Important: this is invasive. chrome-use gains full control over the user's real browser. You (the Support agent) must explain what it does and get the user's explicit confirmation BEFORE running this tool. On Windows this fails — the curl|sh installer is macOS/Linux only, and the user must download the .exe instead.

Note: this tool is only for the FIRST install. mahbot auto-updates an existing chrome-use installation (binary + native host) at every service startup, so you do not need to re-run this to get updates — and that auto-update never first-installs silently (only the initial, user-confirmed install goes through you).
