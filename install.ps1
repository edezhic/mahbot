# MahBot's own installer for Windows.
#
# It downloads this system's ready-made release file, puts it in place, makes
# the location visible to the owner's own environment and start menu, and starts
# the product. It never builds anything, and needs nothing beyond what Windows
# itself ships.
#
# The release contract it mirrors — one contract in four places: this script, the
# unix-like one (`install.sh`), the product's own updater (`src/self_update.rs`)
# and the workflow that publishes the files (`.github/workflows/release.yml`):
#
#   newest release:  {base}/releases/latest/download/version.txt, body `<version>\n`
#                    — the newest release that is not a test release, and nothing
#                    at all while only test releases exist
#   exact version:   {base}/releases/download/v{version}/{asset}
#   asset:           mahbot-{version}-windows-{arch}.zip, arch in x86_64/aarch64,
#                    holding exactly one file named `mahbot.exe` at its root
#   version:         the newest release above, or one named as this script's own
#                    `-Version` argument, or in MAHBOT_INSTALL_VERSION — either
#                    spelling, `0.7.0` or the release mark's own `v0.7.0`
#   install:         %LOCALAPPDATA%\Programs\MahBot\mahbot.exe — the per-user
#                    programs directory `src/util/managed_bin.rs::mahbot_install_dir`
#                    names on this platform
#
# Making the location visible is the product's own edit of the owner's `Path`
# value in `HKCU\Environment` (`src/util/owner_path.rs`): the same value, the
# same kind, and the references his own entries contain left unexpanded, because
# writing them back expanded would fix his own environment into the value for
# good. An entry that already names the folder is recognised — spelled out, or
# through a `%REF%` the entry holds — so it is written back not at all and the
# folder never appears twice; the value never gains an empty entry either, since
# one names the current directory, and that is exactly what must not happen to
# his search path.
#
# The ARM64 floor is Windows 11's, not Windows 10's: an ARM64 file exists, but
# nothing below build 22000 — every Windows 10 on ARM — is a system it is built
# for.
#
# Nothing here ends the session it was run from. The way this script is normally
# run — `irm … | iex` — puts its body in that session, so the body is one function
# and a failure only prints the one sentence saying what happened: no exit code is
# set from inside it, because which form it was run in cannot be told from that
# scope without guessing and a wrong guess closes the owner's window. That function,
# and everything else the script adds to the session, is removed again when the run
# ends, so the session is left as it was — apart from two product-qualified names it
# cannot avoid touching: the `MahBotVersion` parameter, bound in that scope and
# removed again at the end (a variable of that name the caller already had is taken
# over while the run lasts), and the `Win32.NativeMethods` type the environment
# broadcast defines, which PowerShell offers no way to define without leaving it
# behind and which appears only when the search path really changed.
#
# MAHBOT_RELEASE_BASE_URL is a test-only hook, the one `src/self_update.rs` also
# honours: it moves the release base off the constant below, which is how the
# published files are exercised end to end against another release host; with it
# unset nothing differs. MAHBOT_INSTALL_VERSION names the version to install when
# the script is piped into `iex`, where no argument can reach it.

param(
    # Product-qualified on purpose: under the documented `irm … | iex` form this
    # runs in the caller's scope, where a name as ordinary as `Version` would be
    # created — or one of his own overwritten. `-Version` still binds, through the
    # alias, so the `-File` form is unchanged.
    [Alias('Version')]
    [string]$MahBotVersion
)

function Install-MahBot {
    param([string]$Version)

    Set-StrictMode -Version Latest
    $ErrorActionPreference = 'Stop'

    # A plain refusal: the sentence is the whole of what the owner is told, and it
    # stops the installation without stopping the session it was run from. Nested
    # here on purpose — a bare `Fail` left in the owner's session could shadow one
    # of his own.
    function Fail([string]$Message) {
        throw $Message
    }

    # ── The release ───────────────────────────────────────────────────────────

    # The base every download is built from: the repository this project publishes
    # to, unless the test-only hook the header names moved it.
    $ReleaseBase = [Environment]::GetEnvironmentVariable('MAHBOT_RELEASE_BASE_URL')
    if (-not $ReleaseBase) {
        $ReleaseBase = 'https://github.com/edezhic/mahbot'
    }

    $InstallDir = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'Programs\MahBot'
    $InstalledExe = Join-Path $InstallDir 'mahbot.exe'

    # Printed for every system the project publishes no file for: one prefix and the
    # plain refusal built from it, which the two floor sentences below extend.
    $NoFilePrefix = 'MahBot has no file for this system'
    $NoFile = "$NoFilePrefix, so nothing was installed."

    # ── 1. Refuse while the product is running ────────────────────────────────

    # Scoped to the data directory this home's product uses, so a product running
    # under another home is not this script's business. Opening the lock file
    # exclusively is how the platform itself proves it is in use: the product holds
    # it open for its whole run (`crate::util::lock`), so the open fails while it
    # does. A lock file that is not there is not a held one — that open would fail
    # for its absence, not for the product's hold — so only a file that exists is
    # worth trying. The directory is resolved the way the product resolves its own
    # storage root (`crate::config::default_config_dir`): HOME when the environment
    # has one, the account's home directory when it does not.
    $ProductHome = [Environment]::GetEnvironmentVariable('HOME')
    if (-not $ProductHome) {
        $ProductHome = $HOME
    }
    $LockPath = Join-Path $ProductHome '.mahbot\mahbot.lock'
    $Held = $false
    if (Test-Path -LiteralPath $LockPath) {
        try {
            $LockHandle = [IO.File]::Open($LockPath, 'Open', 'ReadWrite', 'None')
            $LockHandle.Close()
        } catch {
            $Held = $true
        }
    }
    if ($Held) {
        Fail "MahBot is already running; updating a running installation is MahBot's own update path, so nothing was installed."
    }

    # ── 2. Work out the system ────────────────────────────────────────────────

    # The machine's own architecture, not this process's: a 32-bit process on a
    # 64-bit system reports x86 and names the real one in PROCESSOR_ARCHITEW6432,
    # which is set nowhere else.
    $Processor = [Environment]::GetEnvironmentVariable('PROCESSOR_ARCHITEW6432')
    if (-not $Processor) {
        $Processor = [Environment]::GetEnvironmentVariable('PROCESSOR_ARCHITECTURE')
    }
    $Arch = switch ($Processor) {
        'AMD64' { 'x86_64' }
        'ARM64' { 'aarch64' }
        default { $null }
    }
    if (-not $Arch) {
        Fail $NoFile
    }

    # The build floor per architecture, read from the operating system's own record
    # of itself: `Win32_OperatingSystem` through CIM, else the registry value that
    # record is written from. `[Environment]::OSVersion` is deliberately not used for
    # it: on a 32-bit process it reports the version that process is told it runs on,
    # and without a compatibility manifest it reports 6.2.9200 whatever the machine
    # is — the same reason the product reads the build through `RtlGetVersion`
    # (src/self_update.rs). A build that cannot be read is not refused for: nothing
    # here is refused on a guess, and an unread `$Build` would otherwise compare as
    # below every floor. An ARM64 file exists, but Windows 10 on ARM — every build
    # before 22000 — is not a system it is built for.
    $Build = $null
    try {
        $Build = [int]((Get-CimInstance -ClassName Win32_OperatingSystem).Version.Split('.')[2])
    } catch {
        $Build = $null
    }
    if (-not $Build) {
        try {
            $Build = [int](Get-ItemProperty -Path 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion' -Name CurrentBuildNumber).CurrentBuildNumber
        } catch {
            $Build = $null
        }
    }
    if ($Build) {
        if ($Arch -eq 'aarch64') {
            if ($Build -lt 22000) {
                Fail "${NoFilePrefix}: Windows on ARM is supported from Windows 11 (build 22000) onwards, so nothing was installed."
            }
        } elseif ($Build -lt 17763) {
            Fail "${NoFilePrefix}: Windows 10 version 1809 (build 17763) or newer is required, so nothing was installed."
        }
    }

    # ── 3. The version to install ─────────────────────────────────────────────

    if (-not $Version) {
        $Version = [Environment]::GetEnvironmentVariable('MAHBOT_INSTALL_VERSION')
    }
    if (-not $Version) {
        # The newest release's tiny version file: its body is the version and a
        # newline. It names the newest release that is not a test release — and
        # nothing at all while only test releases exist, where one is installed by
        # naming its version instead.
        try {
            $Response = Invoke-WebRequest -Uri "$ReleaseBase/releases/latest/download/version.txt" -UseBasicParsing
            # A release file is served as a download rather than as text, so this
            # PowerShell hands the body back as bytes and a `Trim` on them throws;
            # as text it hands back a string. Both are read as the version, which is
            # what the file holds either way.
            $Body = $Response.Content
            if ($Body -is [byte[]]) {
                $Body = [System.Text.Encoding]::UTF8.GetString($Body)
            }
            $Version = ([string]$Body).Trim()
        } catch {
            Fail "the newest MahBot release could not be looked up, so nothing was installed (MAHBOT_INSTALL_VERSION names one particular release)."
        }
    }
    # A version named by hand — as this script's own argument, or through
    # MAHBOT_INSTALL_VERSION — may be spelled the way the release's own mark is
    # (`v0.7.0`), while the file's name and its address are built from the version
    # itself: the mark's own `v` is dropped here rather than landing in either.
    if ($Version) {
        $Version = $Version.Trim() -replace '^v', ''
    }
    if (-not $Version) {
        Fail "the MahBot release to install could not be determined, so nothing was installed."
    }

    # A pre-release part says plainly what it is: such a release is not what the
    # owner is given normally.
    if ($Version.Contains('-')) {
        Write-Host "The MahBot $Version release is a test release, not a normal one; installing it."
    }

    # ── 4. Download the file and put it in place ──────────────────────────────

    $Asset = "mahbot-$Version-windows-$Arch.zip"
    $Temp = Join-Path ([IO.Path]::GetTempPath()) ("mahbot-install-" + [Guid]::NewGuid().ToString('N'))
    # A half-written command is never left in place: the file is staged beside its
    # destination and moved onto it, so what sits at the destination is either the
    # version that was there or the new one whole — the same staging `install.sh` does.
    $Staged = Join-Path $InstallDir 'mahbot.exe.new'
    # What already sits at the destination is moved aside rather than overwritten:
    # Windows lets an image that is running be renamed but neither overwritten nor
    # deleted, so a product running from this folder — through another data directory
    # — is replaced exactly as it is on unix, the way the product's own swap replaces
    # a file that is in use there (`src/util/managed_bin.rs::rename_aside_swap`). The
    # copy moved aside is taken away once the new one is in place; while it is still
    # the running image that cannot be done, and a name a leftover still holds could
    # not be renamed onto again — so each run takes a name of its own, and sweeps the
    # ones earlier runs left behind.
    $Aside = Join-Path $InstallDir ('mahbot.exe.old-' + [Guid]::NewGuid().ToString('N'))
    # Whether the copy that was there could not be put back: only a restore that failed
    # leaves it so, and the sentence the owner is given must name that rather than the
    # generic one.
    $Lost = $false
    try {
        New-Item -ItemType Directory -Path $Temp -Force | Out-Null
    } catch {
        Fail "a temporary directory could not be created, so nothing was installed."
    }
    try {
        try {
            Invoke-WebRequest -Uri "$ReleaseBase/releases/download/v$Version/$Asset" -OutFile (Join-Path $Temp $Asset) -UseBasicParsing
        } catch {
            Fail "the MahBot $Version file for this system could not be downloaded, so nothing was installed."
        }
        try {
            Expand-Archive -LiteralPath (Join-Path $Temp $Asset) -DestinationPath $Temp -Force
        } catch {
            Fail "the downloaded MahBot file could not be opened, so nothing was installed."
        }
        try {
            New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
            Copy-Item -LiteralPath (Join-Path $Temp 'mahbot.exe') -Destination $Staged -Force
            if (Test-Path -LiteralPath $InstalledExe) {
                Move-Item -LiteralPath $InstalledExe -Destination $Aside
                try {
                    Move-Item -LiteralPath $Staged -Destination $InstalledExe
                } catch {
                    # Nothing is left broken: the copy that was there goes back, with
                    # the replace semantics the product's own rename-aside has.
                    try {
                        Move-Item -LiteralPath $Aside -Destination $InstalledExe -Force
                    } catch {
                        $Lost = $true
                    }
                    throw
                }
                Remove-Item -LiteralPath $Aside -Force -ErrorAction SilentlyContinue
            } else {
                Move-Item -LiteralPath $Staged -Destination $InstalledExe
            }
            # Asides earlier runs left behind go only now, with the new file already
            # in place: the copy in one can be the last on disk (a run interrupted
            # between its move-aside and its move-in leaves no `mahbot.exe` at all),
            # and nothing here takes a copy away before it has put one there. Files
            # only, and a failure is swallowed — the destination already holds the new
            # file, so a leftover that stays is not this install failing.
            try {
                Get-ChildItem -LiteralPath $InstallDir -Filter 'mahbot.exe.old*' -File -Force -ErrorAction SilentlyContinue |
                    Remove-Item -Force -ErrorAction SilentlyContinue
            } catch {
                # Nothing to say: a leftover that stays is harmless.
            }
        } catch {
            if ($Lost) {
                Fail "the MahBot $Version file could not be put in place, and the copy that was there could not be put back — it is gone and must be put back by hand."
            }
            Fail "the MahBot $Version file could not be put in place, so nothing was installed."
        }
    } finally {
        # No leftover staging or temp files, on either way out.
        Remove-Item -LiteralPath $Staged -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $Temp -Recurse -Force -ErrorAction SilentlyContinue
    }

    # ── 5. Make the location visible ──────────────────────────────────────────

    # The owner's own `Path` value in his own environment key — the one the product
    # edits itself. It is read without expanding the references his entries contain
    # (`DoNotExpandEnvironmentNames`), and written back with the kind it already
    # has, so a value that is an ExpandString stays one and a reference he adds
    # later still expands. Only the folder's own absence is a reason to write: a
    # value that already names it is left exactly as it is, and one is never created
    # holding nothing — an empty `Path` entry names the current directory, which is
    # the last thing his search path needs.
    $Folder = ($InstallDir -replace '/', '\').TrimEnd([char]'\')
    $ValueWritten = $false
    $EnvironmentKey = $null
    try {
        $EnvironmentKey = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey('Environment', $true)
        if (-not $EnvironmentKey) {
            throw 'the environment key could not be opened'
        }
        $HasValue = $false
        foreach ($Name in $EnvironmentKey.GetValueNames()) {
            if ($Name -ieq 'Path') { $HasValue = $true }
        }
        if ($HasValue) {
            $Raw = $EnvironmentKey.GetValue('Path', '', [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
            $Kind = $EnvironmentKey.GetValueKind('Path')
            if (($Kind -ne [Microsoft.Win32.RegistryValueKind]::String) -and ($Kind -ne [Microsoft.Win32.RegistryValueKind]::ExpandString)) {
                throw 'the Path value is not a string one'
            }
        } else {
            $Raw = $null
            # The kind the system's own writers give a value that is not there yet.
            $Kind = [Microsoft.Win32.RegistryValueKind]::ExpandString
        }

        # Compared the way the product compares an entry (`src/util/owner_path.rs`):
        # case-insensitively, with either of a path's separators and a trailing one
        # naming the same folder, and with the `%REF%`s an entry holds resolved — so
        # a value that spells the folder through a reference is recognised rather
        # than gaining a second entry naming it. Resolution is for recognising only:
        # nothing is ever written out expanded. A value that already names the
        # folder is written back not at all, so it stays byte for byte what he made
        # it.
        $AlreadyThere = $false
        if ($null -ne $Raw) {
            foreach ($Entry in $Raw.Split(';')) {
                foreach ($Spelling in @($Entry, [Environment]::ExpandEnvironmentVariables($Entry))) {
                    if ((($Spelling -replace '/', '\').TrimEnd([char]'\')) -ieq $Folder) {
                        $AlreadyThere = $true
                    }
                }
            }
        }
        if (-not $AlreadyThere) {
            # `%` expands and `;` splits a search-path value, so a folder holding
            # either would not name itself once written — the product's own edit of
            # his `Path` refuses exactly that (`src/util/owner_path.rs`). Said
            # plainly, and the value is left as it is.
            if ($Folder.IndexOfAny([char[]]'%;') -ge 0) {
                Write-Host "MahBot's own folder holds a character a search-path entry cannot carry, so it was not added to the owner's own search path."
            } else {
                if ($null -eq $Raw) {
                    # No value at all: the folder is the whole of it, and no empty entry
                    # is anywhere in the value.
                    $NewValue = $Folder
                } else {
                    # His value is there, so every entry it holds stays as it is — an
                    # empty one included, which names the current directory and is his
                    # own arrangement rather than ours to tidy — and the folder follows
                    # them, which is what the product's own edit does.
                    $NewValue = $Raw + ';' + $Folder
                }
                $EnvironmentKey.SetValue('Path', $NewValue, $Kind)
                $ValueWritten = $true
            }
        }
    } catch {
        Write-Host "the owner's own search path could not be updated, so a terminal may not find MahBot until MahBot itself brings it there."
    } finally {
        if ($EnvironmentKey) { $EnvironmentKey.Close() }
    }

    # Tell the programs already running that the environment changed — the same
    # broadcast `setx` makes (`src/util/owner_path.rs`), so a terminal opened
    # afterwards sees the new value without a logoff. Best effort, and inline, so
    # nothing has to be downloaded to make it.
    if ($ValueWritten) {
        try {
            if (-not ('Win32.NativeMethods' -as [type])) {
                Add-Type -Namespace Win32 -Name NativeMethods -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("user32.dll", CharSet = System.Runtime.InteropServices.CharSet.Auto, SetLastError = true)]
public static extern System.IntPtr SendMessageTimeout(System.IntPtr hWnd, uint Msg, System.UIntPtr wParam, string lParam, uint fuFlags, uint uTimeout, out System.UIntPtr lpdwResult);
'@
            }
            $BroadcastResult = [System.UIntPtr]::Zero
            [void][Win32.NativeMethods]::SendMessageTimeout([System.IntPtr]0xffff, 0x001A, [System.UIntPtr]::Zero, 'Environment', 0x0002, 5000, [ref]$BroadcastResult)
        } catch {
            Write-Host "the change to the owner's environment could not be announced to the programs already running."
        }
    }

    # ── 6. The start menu entry ───────────────────────────────────────────────

    # Windows' own shortcut object, pointed at the file just installed. One that
    # cannot be made is said plainly and changes nothing else.
    try {
        $Shell = New-Object -ComObject WScript.Shell
        $Shortcut = $Shell.CreateShortcut((Join-Path ([Environment]::GetFolderPath('Programs')) 'MahBot.lnk'))
        $Shortcut.TargetPath = $InstalledExe
        $Shortcut.Save()
    } catch {
        Write-Host 'the start menu entry for MahBot could not be created.'
    }

    # ── 7. Remove the copy the old way of installing left behind ──────────────

    # Installing from the package registry (`cargo install mahbot`) puts the command
    # in the toolchain's own directory. That copy is the owner's file, so a removal
    # that fails is said plainly and the install goes on.
    $CargoHome = [Environment]::GetEnvironmentVariable('CARGO_HOME')
    if (-not $CargoHome) {
        $CargoHome = Join-Path $HOME '.cargo'
    }
    $Legacy = Join-Path (Join-Path $CargoHome 'bin') 'mahbot.exe'
    # The two locations compared as the files they are, not as text: a `CARGO_HOME`
    # spelled with a redundant separator or another capitalisation names the very file
    # just installed, and removing that would leave the owner with nothing at all.
    if ((Test-Path -LiteralPath $Legacy) -and
        ([IO.Path]::GetFullPath($Legacy) -ine [IO.Path]::GetFullPath($InstalledExe))) {
        try {
            Remove-Item -LiteralPath $Legacy -Force
        } catch {
            Write-Host "the copy the old way of installing left at $Legacy could not be removed; MahBot is installed at $InstalledExe."
        }
    }

    # ── 8. Smart App Control, when it is on ───────────────────────────────────

    # Said honestly rather than silently: the policy lets no exception be made for a
    # single program, so the only remedy is turning it off altogether — and it
    # blocks the browser helper and the runtime MahBot installs besides.
    $SmartAppControl = $null
    try {
        $SmartAppControl = (Get-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy' -Name 'VerifiedAndReputablePolicyState' -ErrorAction Stop).VerifiedAndReputablePolicyState
    } catch {
        $SmartAppControl = $null
    }
    if ($SmartAppControl -eq 1) {
        Write-Host 'Smart App Control is on, and it allows no exception for a single program: the only remedy is turning it off wholesale, and it also blocks the browser helper and the runtime MahBot installs.'
    }

    # ── 9. Start the product ──────────────────────────────────────────────────

    try {
        Start-Process -FilePath $InstalledExe
    } catch {
        Fail 'MahBot could not be started.'
    }
}

# ── Run ───────────────────────────────────────────────────────────────────────

# A failure prints its one sentence and nothing else happens. No exit code is set
# from here on purpose: whether this file is itself the program being run cannot be
# told from the script's own scope without guessing — a body piped into a session
# carries the enclosing script's own command — and guessing wrong runs `exit` in the
# owner's session, closing his window. The sentence is the whole of what he is told,
# in every form this script is run in.
try {
    Install-MahBot -Version $MahBotVersion
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
}
# A session this was piped into keeps nothing of the run: the function and the
# version it was told are all this script adds to it, so both go.
Remove-Item -Path Function:\Install-MahBot -ErrorAction SilentlyContinue
Remove-Variable -Name MahBotVersion -ErrorAction SilentlyContinue
