//! The product's settings, changeable without a window.
//!
//! One [`Setting`] per stored key. [`get`] reads one back whole with its effective value — defaults
//! resolved, a credential masked to a hint — and [`set`] writes one and reports what it did.
//! [`list`] is a summary sized for an ordinary tool answer: a long value is clipped, a catalogue to
//! its count and first few models, and [`get`] carries any one value whole. No `get`, `list` or
//! `set` answer runs past one read-back: a value that would not come back whole is refused, a
//! credential is stored whole and read back as a hint, and a row another surface left longer is
//! refused rather than cut — so the framework's output budget never truncates one. The reference
//! and the `about` README are the exceptions: both are printed whole, the reference with its size
//! pinned by a test and the README held identical to the file the product ships.
//!
//! A write settles its row before it applies the effect, so a failed row write moves nothing; the
//! audio switches then move the pipelines, and start-at-login applies the OS registration, which
//! can fail on its own and says which half landed.
use crate::config::{
    CONFIG, CONFIG_KEY_AUDIO_TRANSCRIPTION_USE_LOCAL, CONFIG_KEY_AUTOLAUNCH, CONFIG_KEY_AUTOUPDATE,
    CONFIG_KEY_DEEPSEEK_KEY, CONFIG_KEY_EXA_KEY, CONFIG_KEY_IMAGE_GEN_MODELS,
    CONFIG_KEY_MANAGER_MODEL, CONFIG_KEY_PROVIDER_ENDPOINT, CONFIG_KEY_PROVIDER_ENDPOINT_KEY,
    CONFIG_KEY_PROVIDER_KEY, CONFIG_KEY_TELEGRAM_API_HASH, CONFIG_KEY_TELEGRAM_API_ID,
    CONFIG_KEY_TELEGRAM_BOT_TOKEN, CONFIG_KEY_TELEGRAM_LOCAL_SERVER, CONFIG_KEY_TTS_ENABLED,
    CONFIG_KEY_VIDEO_MODELS, CONFIG_KEY_VOICE_ENABLED, CONFIG_KEY_WEB_SEARCH_PROVIDER,
    CONFIG_KEY_WORKER_MODEL, ConfigData,
};
use anyhow::{anyhow, bail};

/// The reference text: every action and every field of this surface and of the `workspace` tool.
pub(crate) fn reference() -> &'static str {
    static REFERENCE: std::sync::LazyLock<String> =
        std::sync::LazyLock::new(|| crate::prompt::load_prompt("config_reference.md"));
    &REFERENCE
}

/// The product's own README, the copy this binary ships: what the system is, what it does and how
/// it is set up. The `about` action and `mahbot config about` both answer with it, so an assistant
/// new to a machine learns what it is running in — and it is static text, so neither route needs a
/// running instance, a lock or a source tree. Embedded from the file itself, so the two cannot
/// drift; a test reads the same file back.
pub(crate) fn about() -> &'static str {
    include_str!("../README.md")
}

/// The one setting form whose suffix is not fixed: a model's provider priority. A form rather
/// than a setting, so it is no [`Setting`] variant — `get` and `set` recognise it by prefix.
const PROVIDER_ORDER_PREFIX: &str = "provider_order:";

/// Every setting this surface serves, one variant each: the name is the stored key, so a setting
/// cannot be advertised under a name nothing writes, and the read-back is an exhaustive match — a
/// new variant is a compile error until it is described. The write falls back to storing the value
/// as the setting's own row ([`Setting::set`]), so a variant with nothing to apply needs no arm.
/// The iteration order (the declaration order) is the order `list` prints.
#[derive(Clone, Copy, PartialEq, Eq, Debug, strum::EnumIter)]
pub(crate) enum Setting {
    ProviderKey,
    ProviderEndpoint,
    ProviderEndpointKey,
    ManagerModel,
    WorkerModel,
    ImageGenModels,
    VideoModels,
    WebSearchProvider,
    DeepseekKey,
    ExaKey,
    TelegramBotToken,
    TelegramLocalServer,
    TelegramApiId,
    TelegramApiHash,
    AudioTranscriptionUseLocal,
    VoiceEnabled,
    TtsEnabled,
    Autoupdate,
    Autolaunch,
}

impl Setting {
    /// Every setting, in the order `list` prints them — the declaration order, from the derive, so
    /// a variant cannot be left out of the list the reference and `list` print.
    pub(crate) fn all() -> impl Iterator<Item = Self> {
        <Self as strum::IntoEnumIterator>::iter()
    }

    /// The stored key — the name `get` and `set` take, and the one the reference lists.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::ProviderKey => CONFIG_KEY_PROVIDER_KEY,
            Self::ProviderEndpoint => CONFIG_KEY_PROVIDER_ENDPOINT,
            Self::ProviderEndpointKey => CONFIG_KEY_PROVIDER_ENDPOINT_KEY,
            Self::ManagerModel => CONFIG_KEY_MANAGER_MODEL,
            Self::WorkerModel => CONFIG_KEY_WORKER_MODEL,
            Self::ImageGenModels => CONFIG_KEY_IMAGE_GEN_MODELS,
            Self::VideoModels => CONFIG_KEY_VIDEO_MODELS,
            Self::WebSearchProvider => CONFIG_KEY_WEB_SEARCH_PROVIDER,
            Self::DeepseekKey => CONFIG_KEY_DEEPSEEK_KEY,
            Self::ExaKey => CONFIG_KEY_EXA_KEY,
            Self::TelegramBotToken => CONFIG_KEY_TELEGRAM_BOT_TOKEN,
            Self::TelegramLocalServer => CONFIG_KEY_TELEGRAM_LOCAL_SERVER,
            Self::TelegramApiId => CONFIG_KEY_TELEGRAM_API_ID,
            Self::TelegramApiHash => CONFIG_KEY_TELEGRAM_API_HASH,
            Self::AudioTranscriptionUseLocal => CONFIG_KEY_AUDIO_TRANSCRIPTION_USE_LOCAL,
            Self::VoiceEnabled => CONFIG_KEY_VOICE_ENABLED,
            Self::TtsEnabled => CONFIG_KEY_TTS_ENABLED,
            Self::Autoupdate => CONFIG_KEY_AUTOUPDATE,
            Self::Autolaunch => CONFIG_KEY_AUTOLAUNCH,
        }
    }

    pub(crate) fn parse(name: &str) -> Option<Self> {
        Self::all().find(|setting| setting.name() == name)
    }

    /// Whether the value is a credential: [`get`] masks it and [`list`] leaves it out. A URL or a
    /// numeric id is not one.
    const fn is_secret(self) -> bool {
        matches!(
            self,
            Self::ProviderKey
                | Self::ProviderEndpointKey
                | Self::DeepseekKey
                | Self::ExaKey
                | Self::TelegramBotToken
                | Self::TelegramApiHash
        )
    }
}

/// A credential's read-back: that it is set, plus its last four characters when it is long
/// enough for a hint to say nothing else. Never the value.
fn mask(value: Option<&str>) -> String {
    match value.map(str::trim).filter(|value| !value.is_empty()) {
        None => "not set".to_string(),
        Some(value) if value.chars().count() > 8 => {
            let tail: String = value
                .chars()
                .rev()
                .take(4)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            format!("set (…{tail})")
        }
        Some(_) => "set".to_string(),
    }
}

fn on_off(on: bool) -> &'static str {
    if on { "on" } else { "off" }
}

/// Parse a switch's new value: the words a person and a model both reach for. Anything else is a
/// usage error rather than a silent "off". `local` / `cloud` are not switch words — they name the
/// Telegram mode, which [`crate::channels::telegram_local::Mode::parse`] reads on its own.
fn parse_switch(value: &str) -> anyhow::Result<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "on" | "true" | "yes" | "1" | "enabled" => Ok(true),
        "off" | "false" | "no" | "0" | "disabled" => Ok(false),
        other => bail!(
            "usage: '{other}' is not a switch value — pass on or off (see the `reference` action)"
        ),
    }
}

/// The error every guessed name gets: it says what is wrong and where the real list is.
fn unknown_setting(name: &str) -> anyhow::Error {
    anyhow!(
        "unknown setting '{name}' — the `reference` action of this tool (`mahbot config \
         reference` in a terminal) lists every setting and every action"
    )
}

/// Whether this build can update itself: only one the release pipeline delivered can.
pub(crate) fn autoupdate_available() -> bool {
    crate::self_update::update_mode() == crate::self_update::UpdateMode::Downloaded
}

/// Why auto-update cannot act here — the one sentence the read-back's inert suffix and the write's
/// refusal both state.
const AUTOUPDATE_INERT: &str = "this build does not update itself";

/// Whether start-at-login can be offered at all: only a published copy registers itself at
/// login. A host whose service the system manages is refused later, by the registration itself.
pub(crate) fn autolaunch_available() -> bool {
    crate::self_update::is_published_copy()
}

/// Why start-at-login cannot act here — the one sentence the read-back's inert suffix and the
/// write's refusal both state.
const AUTOLAUNCH_INERT: &str = "this build does not register at login — only a published copy does";

/// The routing form's model, when `name` is that form. The bare word yields `Some("")`, so the
/// caller's own "needs a model" message answers it rather than "unknown setting".
fn order_model(name: &str) -> Option<&str> {
    name.strip_prefix(PROVIDER_ORDER_PREFIX)
        .or_else(|| (name == PROVIDER_ORDER_PREFIX.trim_end_matches(':')).then_some(""))
}

/// The routing form's shape, for the bare word — `get` and `set` answer it the same way.
fn provider_order_usage() -> String {
    format!(
        "usage: `provider_order` needs a model — `provider_order:<model>`, e.g. \
         `provider_order:{}`",
        CONFIG.manager_model()
    )
}

/// The one `name = value` line a read-back prints — [`get`]'s answer and every write's, so the
/// shape is defined once.
fn read_back_line(name: &str, description: &str) -> String {
    format!("{name} = {description}")
}

/// One setting's effective value, as a `name = value` line. A value longer than one answer can
/// carry is refused rather than cut: the promise is that a single setting reads whole.
pub(crate) async fn get(name: &str) -> anyhow::Result<String> {
    let name = name.trim();
    let config = CONFIG.snapshot();
    if let Some(model) = order_model(name) {
        let model = model.trim();
        anyhow::ensure!(!model.is_empty(), "{}", provider_order_usage());
        return whole_read(name, read_back_line(name, &effective_order(&config, model)));
    }
    let setting = Setting::parse(name).ok_or_else(|| unknown_setting(name))?;
    whole_read(
        name,
        read_back_line(
            name,
            &setting
                .describe(&config, Detail::Full { probe: true })
                .await,
        ),
    )
}

/// The models whose routing rows are shown and summarised: the two slots, each resolved to the
/// model in use — one row per distinct model, which is what a routing row is keyed by.
pub(crate) fn routing_slots(config: &ConfigData) -> std::collections::BTreeSet<String> {
    [
        crate::config::resolve_or(
            config.manager_model.clone(),
            crate::config::DEFAULT_MANAGER_MODEL,
        ),
        crate::config::resolve_or(
            config.worker_model.clone(),
            crate::config::DEFAULT_WORKER_MODEL,
        ),
    ]
    .into_iter()
    .collect()
}

/// Every non-secret setting's effective value, one line each, credentials left out and named in a
/// closing line. A summary, not a dump: a catalogue is its count and first few models, no line runs
/// past [`SUMMARY_LINE_BYTES`], and the login registration is not probed — [`get`] reads any one
/// setting whole.
pub(crate) async fn list() -> String {
    summary(&CONFIG.snapshot()).await
}

/// [`list`] for one snapshot.
async fn summary(config: &ConfigData) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    for setting in Setting::all().filter(|setting| !setting.is_secret()) {
        let line = read_back_line(
            setting.name(),
            &setting.describe(config, Detail::Summary).await,
        );
        let _ = writeln!(out, "{}", crate::util::cut_bytes(&line, SUMMARY_LINE_BYTES));
    }
    // One line per distinct slot model, which is what the routing rows are keyed by: a row for any
    // other model is honoured by the router but is read with `get`, not here.
    for model in routing_slots(config) {
        let line = read_back_line(
            &format!("{PROVIDER_ORDER_PREFIX}{model}"),
            &effective_order(config, &model),
        );
        let _ = writeln!(out, "{}", crate::util::cut_bytes(&line, SUMMARY_LINE_BYTES));
    }
    let secrets: Vec<&str> = Setting::all()
        .filter(|setting| setting.is_secret())
        .map(Setting::name)
        .collect();
    let _ = write!(
        out,
        "credentials (never listed — `get` one by name): {}",
        secrets.join(", ")
    );
    out
}

/// The product's own order when it has none for a model.
const NO_BUILT_IN_ORDER: &str = "the product's own order (none for this model)";

/// Why a stored priority order is inert: a custom chat endpoint does no provider routing. The
/// read-back appends it.
const ROUTING_INERT: &str = " (inert — a custom chat endpoint uses no provider routing)";

/// A model's provider priority as the product actually uses it: the saved row, else the built-in
/// list the row would fall back to. A custom chat endpoint has one provider and never consults
/// these rows, so the read-back says the row is inert rather than reporting an order nothing reads.
fn effective_order(config: &ConfigData, model: &str) -> String {
    let stored = config
        .model_routings
        .iter()
        .find(|row| row.model == model)
        .and_then(|row| row.provider_order.clone())
        .unwrap_or_default();
    order_answer(config, model, &stored)
}

/// The order a routing row prints once `value` is stored — the write's own answer, and the read-back
/// when the row holds it. `""` clears the row, so the product's own order answers instead.
fn order_answer(config: &ConfigData, model: &str, value: &str) -> String {
    let order = match crate::config::trimmed_or_none(value) {
        Some(value) => value,
        None => match crate::config::default_model_routing(model) {
            Some(order) => format!("{order} (the product's own order)"),
            None => NO_BUILT_IN_ORDER.to_string(),
        },
    };
    if crate::config::is_custom_endpoint(&crate::config::effective_chat_endpoint(config)) {
        format!("{order}{ROUTING_INERT}")
    } else {
        order
    }
}

/// The routing form's answer: `name = order`, or `name cleared — order` when the row is emptied.
fn routing_answer(name: &str, cleared: bool, order: &str) -> String {
    if cleared {
        format!("{name} cleared — {order}")
    } else {
        read_back_line(name, order)
    }
}

/// How much a read-back says.
#[derive(Clone, Copy)]
enum Detail {
    /// One setting's value in full. `probe` also pays for the OS reads only a single-setting read
    /// needs — start-at-login's login registration, a `launchctl` call on macOS and a registry read
    /// on Windows.
    Full { probe: bool },
    /// One line of [`list`]: a catalogue is its count and first few models.
    Summary,
}

impl Detail {
    /// One setting's value in full, without the OS reads only a single-setting read pays for —
    /// what a write answers with, since it already knows the state it left.
    const SETTLED: Self = Self::Full { probe: false };
}

/// How many models a summary names before it counts the rest.
const SUMMARY_MODELS: usize = 4;

/// The longest a model name runs inside a summary line, in bytes.
const SUMMARY_MODEL_BYTES: usize = 40;

/// The longest line [`list`] prints for one setting, whatever is stored. Bytes, the unit the
/// tool-output budget is measured in: [`crate::util::cut_bytes`] holds the bound for a value of
/// multi-byte characters too, so the whole summary fits
/// [`crate::util::TOOL_OUTPUT_BUDGET_BYTES`] and arrives uncut.
const SUMMARY_LINE_BYTES: usize = 200;

/// The most a single read-back may run: the tool-output budget, which is what a whole answer has
/// to fit. [`get`] promises one setting whole, so nothing is stored that could not come back
/// within it.
pub(crate) const MAX_READ_BACK_BYTES: usize = crate::util::TOOL_OUTPUT_BUDGET_BYTES;

/// A provider write's answer: the value line every `set` answers with, then the report `setup`
/// rendered. The one place the two are joined, so a test can measure the whole answer.
pub(crate) fn answer_with_report(name: &str, shown: &str, report: &str) -> String {
    format!("{}\n{report}", read_back_line(name, shown))
}

/// Refuse an answer longer than [`MAX_READ_BACK_BYTES`]: the framework cuts anything past one
/// read-back, so a value whose answer could not come back whole is refused rather than stored.
fn ensure_answer_fits(name: &str, len: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        len <= MAX_READ_BACK_BYTES,
        "usage: `{}` cannot be stored — its answer would run past the {MAX_READ_BACK_BYTES} bytes \
         one read-back carries; shorten it (see the `reference` action)",
        crate::util::cut_bytes(name, 64)
    );
    Ok(())
}

/// [`ensure_answer_fits`] for a `name = value` read-back — what [`get`] prints and what a plain
/// setting's write answers.
fn ensure_reads_back_whole(name: &str, value: &str) -> anyhow::Result<()> {
    ensure_answer_fits(name, read_back_line(name, value).len())
}

/// Refuse a read whose answer would not fit [`MAX_READ_BACK_BYTES`]. A row another surface stored
/// (the window, `setup`) can be longer than this one ever writes, and a refusal that says so is
/// better than a value cut in half.
fn whole_read(name: &str, line: String) -> anyhow::Result<String> {
    anyhow::ensure!(
        line.len() <= MAX_READ_BACK_BYTES,
        "`{}` answers longer than one read-back — `get` delivers at most {MAX_READ_BACK_BYTES} \
         bytes; shorten it with `set` (see the `reference` action)",
        crate::util::cut_bytes(name, 64)
    );
    Ok(line)
}

/// The same bound for a write's answer, which carries the report of what the write did. The write
/// has already landed when the answer is composed, so the refusal says that: cutting the report
/// would lose its tail, and silence about the write would be worse.
fn whole_answer(name: &str, answer: String) -> anyhow::Result<String> {
    anyhow::ensure!(
        answer.len() <= MAX_READ_BACK_BYTES,
        "`{}` was written, but its answer runs past the {MAX_READ_BACK_BYTES} bytes one read-back \
         carries — the report's own prose is unbounded (an error chain, a stored endpoint the \
         window left long); `get` reports the setting itself (see the `reference` action)",
        crate::util::cut_bytes(name, 64)
    );
    Ok(answer)
}

impl Setting {
    /// The effective value, defaults resolved. The match is exhaustive: a setting with no
    /// read-back is a compile error.
    async fn describe(self, config: &ConfigData, detail: Detail) -> String {
        use std::fmt::Write as _;
        match self {
            Self::ProviderKey => mask(config.provider_key.as_deref()),
            Self::ProviderEndpoint => {
                let endpoint = crate::config::effective_chat_endpoint(config);
                if crate::config::is_custom_endpoint(&endpoint) {
                    endpoint
                } else {
                    format!("{endpoint} (default)")
                }
            }
            Self::ProviderEndpointKey => mask(config.provider_endpoint_key.as_deref()),
            Self::ManagerModel => resolved(
                config.manager_model.as_deref(),
                crate::config::DEFAULT_MANAGER_MODEL,
            ),
            Self::WorkerModel => resolved(
                config.worker_model.as_deref(),
                crate::config::DEFAULT_WORKER_MODEL,
            ),
            Self::ImageGenModels => catalogue_line(
                config.image_gen_models.as_deref(),
                crate::config::DEFAULT_IMAGE_GEN_MODEL,
                detail,
            ),
            Self::VideoModels => catalogue_line(
                config.video_models.as_deref(),
                crate::config::DEFAULT_VIDEO_MODEL,
                detail,
            ),
            Self::WebSearchProvider => describe_search(config),
            Self::DeepseekKey => mask(config.deepseek_key.as_deref()),
            Self::ExaKey => mask(config.exa_key.as_deref()),
            Self::TelegramBotToken => mask(config.telegram_bot_token.as_deref()),
            Self::TelegramLocalServer => {
                let status = crate::channels::telegram_local::status();
                let mut line = status.serving.as_str().to_string();
                if status.migrating {
                    let _ = write!(line, " (switching to {})", status.target.as_str());
                }
                if let Some(reason) = status.blocked.as_deref() {
                    let _ = write!(line, " (waiting: {reason})");
                }
                line
            }
            Self::TelegramApiId => crate::config::trimmed_or_none(
                config.telegram_api_id.as_deref().unwrap_or_default(),
            )
            .unwrap_or_else(|| "not set".to_string()),
            Self::TelegramApiHash => mask(config.telegram_api_hash.as_deref()),
            Self::AudioTranscriptionUseLocal => on_off(crate::config::flag_on(
                config.audio_transcription_use_local.as_deref(),
            ))
            .to_string(),
            Self::VoiceEnabled => switch_line(
                crate::config::flag_opt_in(config.voice_enabled.as_deref()),
                voice_inert_reason(config),
            ),
            Self::TtsEnabled => switch_line(
                crate::config::flag_opt_in(config.tts_enabled.as_deref()),
                audio_inert_reason(),
            ),
            Self::Autoupdate => switch_line(
                crate::config::flag_on(config.autoupdate.as_deref()),
                (!autoupdate_available()).then_some(AUTOUPDATE_INERT),
            ),
            Self::Autolaunch => describe_autolaunch(config, detail).await,
        }
    }
}

/// A switch's line: `on` plus the durable reason it cannot act on this host, when there is one.
fn switch_line(on: bool, inert: Option<&'static str>) -> String {
    match (on, inert) {
        (true, Some(reason)) => format!("on (inert — {reason})"),
        _ => on_off(on).to_string(),
    }
}

/// Why the wake word cannot act here: transcription off (they share the recogniser) or no audio at
/// all.
fn voice_inert_reason(config: &ConfigData) -> Option<&'static str> {
    if crate::config::flag_on(config.audio_transcription_use_local.as_deref()) {
        audio_inert_reason()
    } else {
        Some("transcription is off")
    }
}

/// Why a switch that needs sound cannot act here. Only what the code can prove cheaply: a pipeline
/// that is merely not running yet is NOT inert — it reads the stored state when it starts — so it
/// is not reported.
fn audio_inert_reason() -> Option<&'static str> {
    (!crate::audio::library_available()).then_some("this host has no audio")
}

/// The web-search read-back: the stored choice, or "automatic", beside the backends text and
/// picture search really resolve to, resolved apart.
fn describe_search(config: &ConfigData) -> String {
    use crate::tools::web_search::WebSearchBackend;
    let choice =
        crate::config::trimmed_or_none(config.web_search_provider.as_deref().unwrap_or_default());
    let resolution = match crate::tools::web_search::backends_in_use(
        choice.as_deref(),
        config.deepseek_key.clone(),
        config.exa_key.clone(),
    ) {
        Some((text, Some(pictures))) => format!("text: {text}; pictures: {pictures}"),
        // No picture key: a picture search is answered by the text backend.
        Some((text, None)) => format!("text: {text}; pictures: {text} too"),
        None => match choice.as_deref() {
            Some(choice) if choice.eq_ignore_ascii_case(WebSearchBackend::DEEPSEEK) => format!(
                "no search tool — {} is selected but no deepseek_key is set",
                WebSearchBackend::DEEPSEEK
            ),
            Some(choice) if choice.eq_ignore_ascii_case(WebSearchBackend::EXA) => format!(
                "no search tool — {} is selected but no exa_key is set",
                WebSearchBackend::EXA
            ),
            Some(other) => format!("no search tool — '{other}' is not a provider"),
            None => "no search tool — no search key is set".to_string(),
        },
    };
    match choice {
        Some(choice) => format!("{choice} ({resolution})"),
        None => format!("automatic ({resolution})"),
    }
}

/// The start-at-login read-back: the owner's decision, whether the login registration is really in
/// place, and — on a build that cannot register — that the decision is inert beside them, since the
/// machine may still carry a registration an earlier published copy made.
///
/// The registration is an OS read (`launchctl` on macOS, the registry on Windows), so only a
/// single-setting read pays for it.
async fn describe_autolaunch(config: &ConfigData, detail: Detail) -> String {
    let decision = crate::config::flag_on(config.autolaunch.as_deref());
    match detail {
        Detail::Full { probe: true } => {
            autolaunch_line(decision, registered_at_login().await.into())
        }
        _ => autolaunch_line(decision, Registration::NotRead),
    }
}

/// The login registration as far as it is known: not read at all — the summary, or a write that
/// already has the OS's answer — read but unanswered, or the OS's own answer.
#[derive(Clone, Copy)]
enum Registration {
    NotRead,
    Unknown,
    Known(bool),
}

impl From<Option<bool>> for Registration {
    fn from(answer: Option<bool>) -> Self {
        match answer {
            Some(registered) => Self::Known(registered),
            None => Self::Unknown,
        }
    }
}

/// The start-at-login line, shared by the read-back and the write so the two cannot describe the
/// same switch differently: the decision, the login registration when it was read, and the inert
/// suffix a build that cannot register carries.
fn autolaunch_line(decision: bool, registration: Registration) -> String {
    use std::fmt::Write as _;
    let mut line = on_off(decision).to_string();
    let answer = match registration {
        Registration::NotRead => None,
        Registration::Unknown => Some("unknown"),
        Registration::Known(true) => Some("in place"),
        Registration::Known(false) => Some("not in place"),
    };
    if let Some(registration) = answer {
        let _ = write!(line, " (login registration: {registration})");
    }
    if !autolaunch_available() {
        let _ = write!(line, " (inert — {AUTOLAUNCH_INERT})");
    }
    line
}

/// A setting the product falls back to a built-in value for: the value in use, marked as the
/// default when nothing is stored.
fn resolved(stored: Option<&str>, fallback: &str) -> String {
    match stored.and_then(crate::config::trimmed_or_none) {
        Some(value) => value,
        None => format!("{fallback} (default)"),
    }
}

/// A catalogue's read-back: the models in use, or the default model. A summary names the count and
/// the first few; [`get`] reads the whole list.
fn catalogue_line(stored: Option<&str>, fallback: &str, detail: Detail) -> String {
    let Some(models) = stored
        .map(crate::config::parse_newline_list)
        .filter(|models| !models.is_empty())
    else {
        return format!("{fallback} (default)");
    };
    if matches!(detail, Detail::Full { .. }) {
        return models.join(", ");
    }
    let shown: Vec<String> = models
        .iter()
        .take(SUMMARY_MODELS)
        .map(|model| crate::util::cut_bytes(model, SUMMARY_MODEL_BYTES))
        .collect();
    let mut line = format!(
        "{} model{}: {}",
        models.len(),
        if models.len() == 1 { "" } else { "s" },
        shown.join(", ")
    );
    if models.len() > shown.len() {
        line.push_str(", …");
    }
    line
}

/// Set one setting and report what it did; the change lands at once, with nothing left for a
/// restart. A value whose read-back would not fit one whole answer is refused before anything is
/// written, and an answer that would not fit — a provider write's own report — is refused rather
/// than cut.
pub(crate) async fn set(name: &str, value: &str) -> anyhow::Result<String> {
    let name = name.trim();
    let value = value.trim();
    let answer = if let Some(model) = order_model(name) {
        let model = model.trim();
        anyhow::ensure!(!model.is_empty(), "{}", provider_order_usage());
        // The answer is measured before the write: `name = order`, or `name cleared — …` when the
        // row is emptied, with the inert sentence a custom chat endpoint appends.
        let planned = routing_answer(
            name,
            value.is_empty(),
            &order_answer(&CONFIG.snapshot(), model, value),
        );
        ensure_answer_fits(name, planned.len())?;
        let stored = crate::config::persist_settled_routing_order(model, value).await?;
        // The effective order, not the row: a cleared row falls back to the product's own order,
        // and a custom chat endpoint makes either inert — `get` says the same thing.
        routing_answer(
            name,
            stored.is_empty(),
            &effective_order(&CONFIG.snapshot(), model),
        )
    } else {
        Setting::parse(name)
            .ok_or_else(|| unknown_setting(name))?
            .set(value)
            .await?
    };
    // A provider write appends a report whose own error text is unbounded, so the answer is
    // checked here too — refused, never cut, since a cut would lose the report's tail.
    whole_answer(name, answer)
}

impl Setting {
    /// Write the setting. A name that is not a [`Setting`] never reaches this; every variant either
    /// has its own arm — the ones with something to apply — or is stored as its own row by the
    /// fallback.
    async fn set(self, value: &str) -> anyhow::Result<String> {
        // Only a value `get` prints whole has to fit one answer: a credential reads back as a hint
        // whatever it holds.
        if !self.is_secret() {
            ensure_reads_back_whole(self.name(), value)?;
        }
        match self {
            // The provider's rows are written by [`crate::setup`], which validates them; this route
            // only forwards the two values it can carry. Neither is cleared here: `setup` refuses a
            // blank key, and settling the endpoint back is its own explicit request.
            Self::ProviderKey | Self::ProviderEndpoint => {
                anyhow::ensure!(
                    !value.is_empty(),
                    "usage: `{}` cannot be cleared by `set` — {} (see the `reference` action)",
                    self.name(),
                    if self == Self::ProviderKey {
                        "give it a value"
                    } else {
                        "`setup` with `clear_provider_endpoint` settles back to OpenRouter's own"
                    }
                );
                let report = crate::setup::run(&serde_json::json!({ self.name(): value })).await?;
                // The value first, the way every other `set` answers, so a caller need not read
                // prose for it; `setup`'s own report follows, because its warnings (a key the
                // platform refused, an endpoint that does not answer) are what it did.
                Ok(answer_with_report(
                    self.name(),
                    &settled_description(self).await,
                    &report,
                ))
            }
            // A custom endpoint's key names nothing on its own: `setup` writes it with the endpoint
            // it belongs to, and refuses it alone — so this route says where it is written.
            Self::ProviderEndpointKey => bail!(
                "usage: `provider_endpoint_key` is written with its endpoint — call `setup` with \
                 both `provider_endpoint` and `provider_endpoint_key` (see the `reference` action)"
            ),
            Self::TelegramApiId | Self::TelegramApiHash => {
                set_telegram_credential(self, value).await
            }
            Self::ImageGenModels => set_catalogue(self, value, true).await,
            Self::VideoModels => set_catalogue(self, value, false).await,
            Self::WebSearchProvider => set_search_provider(value).await,
            Self::TelegramLocalServer => set_local_mode(value).await,
            Self::AudioTranscriptionUseLocal => set_transcription(value).await,
            Self::VoiceEnabled => set_switch(self, value, apply_voice_toggle).await,
            Self::TtsEnabled => set_switch(self, value, apply_speech_output_toggle).await,
            Self::Autoupdate => set_autoupdate(value).await,
            Self::Autolaunch => set_autolaunch(value).await,
            // Every other setting is a stored string with nothing to apply beyond its row.
            other => set_stored_string(other, value).await,
        }
    }
}

/// What [`get`] answers for `setting`: its effective value, defaults resolved. Every write answers
/// with this too, so the read-back request is defined once.
async fn settled_description(setting: Setting) -> String {
    setting.describe(&CONFIG.snapshot(), Detail::SETTLED).await
}

/// The [`get`] line for `setting`: its name and the value in use.
async fn settled_line(setting: Setting) -> String {
    read_back_line(setting.name(), &settled_description(setting).await)
}

/// A setting whose whole effect is its row: persist it and report the effective value — masked for
/// a credential, and the value in use afterwards when the call cleared it.
async fn set_stored_string(setting: Setting, value: &str) -> anyhow::Result<String> {
    crate::config::persist_settled_string_field(setting.name(), value).await?;
    Ok(settled_line(setting).await)
}

/// How many models a catalogue takes, and how long one name may be. A model-driven `set` could
/// otherwise store an arbitrarily long row and force a catalogue lookup per entry.
const MAX_CATALOGUE_ENTRIES: usize = 32;
const MAX_MODEL_NAME_CHARS: usize = 128;

/// A media catalogue: newline- or comma-separated model names; empty restores the default.
///
/// Image models are checked against the catalogue and a model it does not hold is REFUSED; the
/// check itself fails open when the catalogue cannot be reached, so an outage never blocks a save.
/// Video models are a free list.
async fn set_catalogue(setting: Setting, value: &str, validate: bool) -> anyhow::Result<String> {
    let name = setting.name();
    let models: Vec<String> = value
        .split(['\n', ','])
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .map(str::to_string)
        .collect();
    anyhow::ensure!(
        models.len() <= MAX_CATALOGUE_ENTRIES,
        "usage: `{name}` takes at most {MAX_CATALOGUE_ENTRIES} models — got {} (see the \
         `reference` action)",
        models.len()
    );
    if let Some(long) = models
        .iter()
        .find(|model| model.chars().count() > MAX_MODEL_NAME_CHARS)
    {
        bail!(
            "usage: `{name}` takes model names of at most {MAX_MODEL_NAME_CHARS} characters — \
             '{}' is {} (see the `reference` action)",
            crate::util::truncate(long, 24),
            long.chars().count()
        );
    }
    // The answer joins with `, `, and so does `get` — the bound is measured on what both print,
    // not on the raw input, whose separators may be newlines.
    let joined = models.join(", ");
    ensure_reads_back_whole(name, &joined)?;
    if validate {
        for model in &models {
            crate::tools::media_catalog::image::validate_image_model(model).await?;
        }
    }
    let stored = models.join("\n");
    crate::config::persist_settled_string_field(name, &stored).await?;
    let effective = crate::config::parse_newline_list(&stored);
    Ok(if effective.is_empty() {
        // A cleared catalogue is not an empty answer: report the default now in use.
        format!(
            "{name} cleared — the default model is in use: {}",
            settled_description(setting).await
        )
    } else {
        settled_line(setting).await
    })
}

/// A local Telegram Bot API server's credential: shape-checked before the row is written. Only the
/// value being written is judged — the other half of the pair may still be unset — and an empty
/// value clears the row.
async fn set_telegram_credential(setting: Setting, value: &str) -> anyhow::Result<String> {
    if !value.is_empty() {
        let shape = if setting == Setting::TelegramApiId {
            crate::channels::telegram_local::parse_api_id(value)
        } else {
            crate::channels::telegram_local::parse_api_hash(value)
        };
        if let Err(problem) = shape {
            bail!("usage: {}: {problem}", setting.name());
        }
    }
    set_stored_string(setting, value).await
}

async fn set_search_provider(value: &str) -> anyhow::Result<String> {
    use crate::tools::web_search::WebSearchBackend;
    let provider = value.to_ascii_lowercase();
    anyhow::ensure!(
        matches!(
            provider.as_str(),
            "" | "auto" | "automatic" | WebSearchBackend::DEEPSEEK | WebSearchBackend::EXA
        ),
        "usage: web_search_provider must be `{}`, `{}`, `auto` (automatic), or empty for \
         automatic — got '{value}' (see the `reference` action)",
        WebSearchBackend::DEEPSEEK,
        WebSearchBackend::EXA
    );
    let stored = if matches!(provider.as_str(), "auto" | "automatic") {
        ""
    } else {
        provider.as_str()
    };
    crate::config::persist_settled_string_field(CONFIG_KEY_WEB_SEARCH_PROVIDER, stored).await?;
    Ok(settled_line(Setting::WebSearchProvider).await)
}

/// The local Bot API mode is an action, not a value: it journals a switch and answers at once.
/// The credentials must be in place first, and the bot token too — the switch has nothing to
/// move without one. The answer never claims a switch that was not started: asking for the side
/// already in force is settled, and says so; and the side being served is read BEFORE the request,
/// so "until it arrives" cannot be contradicted by a switch that already finished.
async fn set_local_mode(value: &str) -> anyhow::Result<String> {
    let Some(target) = crate::channels::telegram_local::Mode::parse(value) else {
        bail!(
            "usage: telegram_local_server must be `local` or `cloud` — got '{value}' (see the \
             `reference` action)"
        );
    };
    let serving = crate::channels::telegram_local::status().serving;
    if crate::channels::telegram_local::request_switch(target).await? {
        Ok(format!(
            "Switch to the {} server requested; the bot is served from {} until it arrives. \
             `get telegram_local_server` reports both.",
            target.as_str(),
            serving.as_str()
        ))
    } else {
        // `false` is the settled answer: that side is already in force, so the served side is it.
        Ok(format!(
            "The bot is already served from the {} server; nothing was switched.",
            serving.as_str()
        ))
    }
}

async fn set_transcription(value: &str) -> anyhow::Result<String> {
    let enabled = parse_switch(value)?;
    // Turning transcription off also turns voice off: they share the loaded recogniser. The
    // cascade is decided under the switch lock, so the answer reports the one that really fired.
    let cascade = apply_transcription_toggle(enabled)
        .await
        .map_err(|e| anyhow!("{e}"))?;
    Ok(read_back_line(
        CONFIG_KEY_AUDIO_TRANSCRIPTION_USE_LOCAL,
        &format!(
            "{}{}",
            on_off(enabled),
            if cascade {
                " — voice turned off too, since it shares the recogniser"
            } else {
                ""
            }
        ),
    ))
}

/// A switch's write: apply it, then answer with the same line [`get`] reads.
async fn set_switch<F, Fut>(setting: Setting, value: &str, apply: F) -> anyhow::Result<String>
where
    F: FnOnce(bool) -> Fut,
    Fut: std::future::Future<Output = Result<(), String>>,
{
    let enabled = parse_switch(value)?;
    apply(enabled).await.map_err(|e| anyhow!("{e}"))?;
    Ok(settled_line(setting).await)
}

async fn set_autoupdate(value: &str) -> anyhow::Result<String> {
    let enabled = parse_switch(value)?;
    if !autoupdate_available() {
        bail!(
            "usage: `{CONFIG_KEY_AUTOUPDATE}` cannot be changed here — {AUTOUPDATE_INERT}: the \
             release pipeline did not deliver this copy, so update it the way it was installed \
             (see the `reference` action)"
        );
    }
    crate::config::persist_settled_string_field(
        CONFIG_KEY_AUTOUPDATE,
        if enabled { "true" } else { "false" },
    )
    .await?;
    if enabled {
        // A switch back on must take effect now, not at the next ten-minute tick.
        crate::self_update::nudge_update_tick();
    }
    Ok(read_back_line(CONFIG_KEY_AUTOUPDATE, on_off(enabled)))
}

async fn set_autolaunch(value: &str) -> anyhow::Result<String> {
    let enabled = parse_switch(value)?;
    if !autolaunch_available() {
        bail!(
            "usage: `{CONFIG_KEY_AUTOLAUNCH}` cannot be changed here — {AUTOLAUNCH_INERT} (see the \
             `reference` action)"
        );
    }
    // The decision is saved before the registration is applied, so a refused registration leaves
    // the choice standing — but the setting did not take effect, so the call fails and says both
    // halves rather than reporting a switch it could not throw. A save that never landed is its
    // own failure: nothing was kept, so the answer must not claim otherwise.
    match apply_autolaunch(enabled).await {
        // The same line the read-back answers, the OS's own answer included — `unknown` when it
        // did not say.
        Ok(registered) => Ok(read_back_line(
            CONFIG_KEY_AUTOLAUNCH,
            &autolaunch_line(enabled, registered.into()),
        )),
        Err(AutolaunchFailure {
            saved: false,
            message,
        }) => Err(anyhow!("{message}")),
        Err(AutolaunchFailure {
            saved: true,
            message,
        }) => bail!(
            "{}",
            read_back_line(
                CONFIG_KEY_AUTOLAUNCH,
                &format!(
                    "{} was saved, but {message}",
                    autolaunch_line(enabled, Registration::NotRead)
                )
            )
        ),
    }
}

// --- Switch side effects ------------------------------------------------------------------

/// One switch is applied at a time — the row it writes and the pipeline it moves — so two rapid
/// toggles cannot interleave one's write with the other's move: whichever runs last settles both
/// halves together, and the pipeline always agrees with the stored value.
static SWITCH: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Move the voice pipeline to `enabled`: its own state first (it survives a pipeline restart), then
/// the command. Nothing reads whether a pipeline was there to take the command — one that is not
/// running reads the stored state when it starts, so "not running" is not "off", and the read-back
/// reports only what it can prove ([`voice_inert_reason`]).
pub(crate) fn sync_voice_state(enabled: bool) {
    crate::audio::voice::set_enabled(enabled);
    let _ = crate::audio::voice::send_command(if enabled {
        crate::audio::voice::VoiceCommand::StartListening
    } else {
        crate::audio::voice::VoiceCommand::StopListening
    });
}

/// The transcription switch: one transaction for the row (absence = enabled) and the voice
/// cascade, then the pipeline's own `ConfigChanged` so it re-arms at once. `Ok` says whether voice
/// was turned off with it.
///
/// The cascade is decided here, under [`SWITCH`]: a voice switch cannot slip in between the
/// decision and the write, so transcription off never leaves voice on.
pub(crate) async fn apply_transcription_toggle(enabled: bool) -> Result<bool, String> {
    let _applied = SWITCH.lock().await;
    let cascade = !enabled && crate::config::flag_opt_in(CONFIG.voice_enabled().as_deref());
    crate::config_db::store()
        .set_transcription_toggle(enabled, cascade)
        .await
        .map_err(|e| e.to_string())?;
    let _ = CONFIG.set_string_field(
        CONFIG_KEY_AUDIO_TRANSCRIPTION_USE_LOCAL,
        if enabled { "" } else { "false" },
    );
    if cascade {
        let _ = CONFIG.set_string_field(CONFIG_KEY_VOICE_ENABLED, "");
        sync_voice_state(false);
    }
    // The pipeline's `ConfigChanged` wakes it, re-arms auto-start and demands the listening tier
    // when the wake word is on — the one place the load decision is made.
    crate::audio::voice::send_command(crate::audio::voice::VoiceCommand::ConfigChanged);
    Ok(cascade)
}

/// The wake-word switch: two refusals — a wake word needs transcription on (they share the
/// recogniser), and an enrolment the current encoder cannot score refuses to turn on — then the
/// row, then the pipeline.
pub(crate) async fn apply_voice_toggle(enabled: bool) -> Result<(), String> {
    let _applied = SWITCH.lock().await;
    if enabled && !crate::config::flag_on(CONFIG.audio_transcription_use_local().as_deref()) {
        return Err("the wake word needs transcription on — they share the recogniser".to_string());
    }
    if enabled && crate::audio::voice::enrollment_unusable() {
        return Err(
            "the stored wake-word enrolment cannot be scored by this build's encoder, so voice \
             cannot be turned on"
                .to_string(),
        );
    }
    // The row first, so a failed write leaves the pipeline exactly where it was.
    write_switch_row(CONFIG_KEY_VOICE_ENABLED, enabled).await?;
    sync_voice_state(enabled);
    Ok(())
}

/// The spoken-replies switch: the row, then the audio output and the model load-or-fetch chain
/// when turning on. What the host can do about it is the read-back's to report
/// ([`audio_inert_reason`]), not this helper's to guess.
pub(crate) async fn apply_speech_output_toggle(enabled: bool) -> Result<(), String> {
    let _applied = SWITCH.lock().await;
    write_switch_row(CONFIG_KEY_TTS_ENABLED, enabled).await?;
    if !enabled {
        return Ok(());
    }
    let _ = crate::audio::output::ensure_audio_output();
    if crate::speech::synth::state() != crate::util::model_state::ModelState::Ready {
        crate::speech::synth::retry();
    }
    Ok(())
}

/// A switch's row: `"true"` when on, absent when off (`non_empty` then reads it as off).
async fn write_switch_row(key: &str, enabled: bool) -> Result<(), String> {
    let store = crate::config_db::store();
    let result = if enabled {
        store.set_kv(key, "true").await
    } else {
        store.delete_kv(key).await
    };
    result.map_err(|e| e.to_string())?;
    let _ = CONFIG.set_string_field(key, if enabled { "true" } else { "" });
    Ok(())
}

/// The OS's own answer on the login registration. `None` = unknown, never "off".
pub(crate) async fn registered_at_login() -> Option<bool> {
    tokio::task::spawn_blocking(crate::autostart::registered)
        .await
        .ok()
        .flatten()
}

/// Save the start-at-login decision. Saved before the registration is applied, so a refused
/// registration never loses the choice.
async fn save_autolaunch(enabled: bool) -> Result<(), String> {
    crate::config::persist_settled_string_field(
        CONFIG_KEY_AUTOLAUNCH,
        if enabled { "true" } else { "false" },
    )
    .await
    .map_err(|e| format!("`{CONFIG_KEY_AUTOLAUNCH}` could not be saved: {e:#}"))?;
    Ok(())
}

/// Apply the decision to the system's own login registration; `None` = the OS did not say.
async fn register_at_login(enabled: bool) -> Result<Option<bool>, String> {
    tokio::task::spawn_blocking(move || crate::autostart::set_enabled(enabled))
        .await
        .map_err(|e| format!("the login registration could not be changed: {e}"))?
        .map_err(|e| format!("the login registration could not be changed: {e:#}"))
}

/// Why start-at-login did not take effect, and whether the decision landed before it failed.
pub(crate) struct AutolaunchFailure {
    saved: bool,
    pub(crate) message: String,
}

/// Save the decision, then apply it to the OS login registration; either half can fail with its
/// own message. `Ok` carries the OS's answer on the registration.
pub(crate) async fn apply_autolaunch(enabled: bool) -> Result<Option<bool>, AutolaunchFailure> {
    if let Err(message) = save_autolaunch(enabled).await {
        return Err(AutolaunchFailure {
            saved: false,
            message,
        });
    }
    register_at_login(enabled)
        .await
        .map_err(|message| AutolaunchFailure {
            saved: true,
            message,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A credential's read-back is a hint and never the value — the rule the whole surface turns
    /// on, tested on the one function that decides it.
    #[test]
    fn a_credential_read_back_is_a_hint_and_never_the_value() {
        assert_eq!(mask(None), "not set");
        assert_eq!(mask(Some("   ")), "not set");
        assert_eq!(mask(Some("short")), "set");
        let key = "sk-or-v1-0123456789abcdef";
        let masked = mask(Some(key));
        assert_eq!(masked, "set (…cdef)");
        assert!(!masked.contains(key) && !masked.contains("0123456789"));
    }

    /// Every setting has a name, no two share one, and each name resolves back to its variant — so
    /// the list the reference and `list` print is the set `get` and `set` serve.
    #[test]
    fn the_settings_are_named_once_and_parse_back() {
        let mut seen = std::collections::BTreeSet::new();
        for setting in Setting::all() {
            let name = setting.name();
            assert!(
                !name.trim().is_empty() && name == name.trim(),
                "a setting name carries no padding and is never empty"
            );
            assert!(seen.insert(name), "two settings are both named '{name}'");
            assert_eq!(Setting::parse(name), Some(setting));
        }
        assert_eq!(Setting::parse("provider_ky"), None);
        assert_eq!(Setting::parse("provider_order:some/model"), None);
    }

    /// The two values that are not credentials but still must never be dumped, and the two that
    /// look like credentials and are not.
    #[test]
    fn only_a_credential_is_treated_as_one() {
        assert!(Setting::TelegramApiHash.is_secret());
        assert!(!Setting::TelegramApiId.is_secret(), "an id is not a secret");
        assert!(
            !Setting::ProviderEndpoint.is_secret(),
            "a URL is not a secret"
        );
    }

    /// The read-back is what the product does with the row, not whether the row exists: the two
    /// opt-in switches are off when absent, and transcription — absent means on — is off on a
    /// fresh install because catalog `69` records the off row.
    #[tokio::test]
    async fn the_switches_read_back_the_way_the_product_reads_them() {
        let mut config = ConfigData {
            voice_enabled: None,
            tts_enabled: None,
            audio_transcription_use_local: None,
            ..ConfigData::STRUCT_FIELDS_DEFAULT
        };
        assert_eq!(
            Setting::VoiceEnabled
                .describe(&config, Detail::SETTLED)
                .await,
            "off"
        );
        assert_eq!(
            Setting::TtsEnabled.describe(&config, Detail::SETTLED).await,
            "off"
        );
        assert_eq!(
            Setting::AudioTranscriptionUseLocal
                .describe(&config, Detail::SETTLED)
                .await,
            "on",
            "an absent transcription row means enabled"
        );

        // What the switches write: `"true"` when on, the row gone when off. A host with no audio
        // library says so beside the `on`; one with audio prints the bare word.
        config.voice_enabled = Some("true".to_string());
        config.tts_enabled = Some("true".to_string());
        for setting in [Setting::VoiceEnabled, Setting::TtsEnabled] {
            let line = setting.describe(&config, Detail::SETTLED).await;
            assert!(line.starts_with("on"), "got: {line}");
            assert_eq!(
                line.contains("inert"),
                !crate::audio::library_available(),
                "only a host with no audio is inert here: {line}"
            );
        }

        config.voice_enabled = Some(String::new());
        config.tts_enabled = Some(String::new());
        assert_eq!(
            Setting::VoiceEnabled
                .describe(&config, Detail::SETTLED)
                .await,
            "off",
            "turning a switch off leaves no row, and that is off"
        );
        assert_eq!(
            Setting::TtsEnabled.describe(&config, Detail::SETTLED).await,
            "off"
        );

        // A fresh install: catalog 69's row.
        config.audio_transcription_use_local = Some("false".to_string());
        assert_eq!(
            Setting::AudioTranscriptionUseLocal
                .describe(&config, Detail::SETTLED)
                .await,
            "off"
        );
        // Voice on while transcription is off is inert, and says so.
        config.voice_enabled = Some("true".to_string());
        assert_eq!(
            Setting::VoiceEnabled
                .describe(&config, Detail::SETTLED)
                .await,
            "on (inert — transcription is off)"
        );
    }

    /// A setting the product falls back on reports the value in use and names the fallback.
    #[tokio::test]
    async fn a_setting_with_a_default_reports_the_default_in_use() {
        let mut config = ConfigData::STRUCT_FIELDS_DEFAULT;
        assert_eq!(
            Setting::ManagerModel
                .describe(&config, Detail::SETTLED)
                .await,
            format!("{} (default)", crate::config::DEFAULT_MANAGER_MODEL)
        );
        assert_eq!(
            Setting::ImageGenModels
                .describe(&config, Detail::SETTLED)
                .await,
            format!("{} (default)", crate::config::DEFAULT_IMAGE_GEN_MODEL)
        );
        config.manager_model = Some("acme/model".to_string());
        config.image_gen_models = Some("acme/image\nother/image".to_string());
        assert_eq!(
            Setting::ManagerModel
                .describe(&config, Detail::SETTLED)
                .await,
            "acme/model"
        );
        assert_eq!(
            Setting::ImageGenModels
                .describe(&config, Detail::SETTLED)
                .await,
            "acme/image, other/image"
        );
    }

    /// The routing read-back reports the order in use, and says a custom chat endpoint makes it
    /// inert.
    #[test]
    fn the_routing_read_back_says_when_it_is_inert() {
        let mut config = ConfigData::STRUCT_FIELDS_DEFAULT;
        let order = effective_order(&config, crate::config::DEFAULT_MANAGER_MODEL);
        assert!(!order.contains("inert"), "got: {order}");
        config.provider_endpoint = Some("https://acme.invalid/v1".to_string());
        let order = effective_order(&config, crate::config::DEFAULT_MANAGER_MODEL);
        assert!(
            order.contains("inert — a custom chat endpoint"),
            "got: {order}"
        );
    }

    /// The wake-word switch needs transcription on, and refuses before touching the pipeline or the
    /// row.
    #[tokio::test]
    #[serial_test::serial(config_persist)]
    async fn a_wake_word_needs_transcription_on() {
        let previous = CONFIG.audio_transcription_use_local();
        let _ = CONFIG.set_string_field(CONFIG_KEY_AUDIO_TRANSCRIPTION_USE_LOCAL, "false");
        let refused = apply_voice_toggle(true).await.unwrap_err();
        assert!(refused.contains("needs transcription on"), "got: {refused}");
        let _ = CONFIG.set_string_field(
            CONFIG_KEY_AUDIO_TRANSCRIPTION_USE_LOCAL,
            previous.as_deref().unwrap_or_default(),
        );
    }

    /// The summary names every non-credential setting and no credential's value, so `list` cannot
    /// quietly drop a setting or leak one.
    #[tokio::test]
    #[serial_test::serial(config_persist)]
    async fn the_summary_names_every_setting_but_no_credential() {
        let summary = list().await;
        for setting in Setting::all() {
            let named = summary.contains(&format!("{} = ", setting.name()));
            assert_eq!(
                named,
                !setting.is_secret(),
                "`list` named `{}` when it should {}",
                setting.name(),
                if setting.is_secret() { "not" } else { "have" }
            );
        }
    }

    /// The summary is a summary: whatever is stored — a value of the widest characters included —
    /// it arrives whole on the Assistant's route, and a catalogue is its count and first few
    /// models. `get` has the whole value.
    #[tokio::test]
    async fn the_summary_fits_the_tool_output_budget() {
        // Four bytes per character: a budget measured in characters would pass here while the
        // bytes overflowed, which is exactly what must not happen.
        let wide = "😀";
        assert_eq!(
            wide.len(),
            4,
            "the widest character is what this test needs"
        );
        let model = wide.repeat(MAX_MODEL_NAME_CHARS);
        let catalogue = vec![model; MAX_CATALOGUE_ENTRIES].join("\n");
        let config = ConfigData {
            image_gen_models: Some(catalogue.clone()),
            video_models: Some(catalogue),
            manager_model: Some(wide.repeat(MAX_MODEL_NAME_CHARS * 4)),
            worker_model: Some(format!("{}w", wide.repeat(MAX_MODEL_NAME_CHARS * 4))),
            provider_endpoint: Some(wide.repeat(MAX_MODEL_NAME_CHARS * 4)),
            ..ConfigData::STRUCT_FIELDS_DEFAULT
        };

        let listing = summary(&config).await;
        assert_eq!(
            crate::util::truncate_tool_output(&listing),
            listing,
            "the summary must arrive whole — the framework's own cut must not touch it"
        );
        assert!(
            listing.len() <= crate::util::TOOL_OUTPUT_BUDGET_BYTES,
            "the summary must arrive whole, got {} bytes",
            listing.len()
        );
        for line in listing.lines() {
            assert!(
                line.len() <= SUMMARY_LINE_BYTES,
                "every line is bounded in bytes, got {}: {line}",
                line.len()
            );
        }

        let line = listing
            .lines()
            .find(|line| line.starts_with("image_gen_models = "))
            .expect("the summary names the image catalogue");
        assert!(
            line.contains(&format!("{MAX_CATALOGUE_ENTRIES} models: ")),
            "a catalogue is its count: {line}"
        );
        assert!(
            line.contains(wide),
            "and the first few models, clipped on a character boundary: {line}"
        );
        assert!(line.contains('…'), "and marks the rest as left out: {line}");

        // One entry is one model, not "1 models".
        let single = summary(&ConfigData {
            image_gen_models: Some("acme/image".to_string()),
            ..ConfigData::STRUCT_FIELDS_DEFAULT
        })
        .await;
        assert!(
            single.contains("image_gen_models = 1 model: acme/image"),
            "a one-entry catalogue reads in the singular: {single}"
        );
    }

    /// The search read-back names what the two capabilities really get, and an empty row — which is
    /// what a cleared setting leaves in the live snapshot — reads as absent, not as a provider.
    #[test]
    fn the_search_read_back_names_both_capabilities_and_reads_empty_as_absent() {
        let describe = |provider: Option<&str>, deepseek: Option<&str>, exa: Option<&str>| {
            describe_search(&ConfigData {
                web_search_provider: provider.map(str::to_string),
                deepseek_key: deepseek.map(str::to_string),
                exa_key: exa.map(str::to_string),
                ..ConfigData::STRUCT_FIELDS_DEFAULT
            })
        };
        assert_eq!(
            describe(None, Some("ds"), Some("exa")),
            "automatic (text: deepseek; pictures: exa)"
        );
        assert_eq!(
            describe(None, Some("ds"), None),
            "automatic (text: deepseek; pictures: deepseek too)"
        );
        assert_eq!(
            describe(Some("exa"), Some("ds"), Some("exa")),
            "exa (text: exa; pictures: exa)"
        );
        assert_eq!(
            describe(Some("deepseek"), Some("ds"), Some("exa")),
            "deepseek (text: deepseek; pictures: deepseek too)"
        );
        assert_eq!(
            describe(Some("deepseek"), None, Some("exa")),
            "deepseek (no search tool — deepseek is selected but no deepseek_key is set)"
        );
        assert_eq!(
            describe(Some("exa"), Some("ds"), None),
            "exa (no search tool — exa is selected but no exa_key is set)"
        );
        // Empty rows are absent, exactly as the accessors read them.
        assert_eq!(
            describe(Some(""), None, Some("exa")),
            "automatic (text: exa; pictures: exa)"
        );
        assert_eq!(
            describe(Some("  "), Some(""), Some("")),
            "automatic (no search tool — no search key is set)"
        );
        assert_eq!(
            describe(Some("google"), Some("ds"), Some("exa")),
            "google (no search tool — 'google' is not a provider)"
        );
    }

    /// A guessed name teaches where the list is instead of failing blind.
    #[test]
    fn an_unknown_setting_names_the_reference() {
        let message = unknown_setting("provider_ky").to_string();
        assert!(message.contains("provider_ky"), "got: {message}");
        assert!(message.contains("`reference` action"), "got: {message}");
    }

    /// `set` is the one write both routes reach: it reports the value in use afterwards — the
    /// default when the call cleared the row — and refuses a guessed name by naming the reference.
    #[tokio::test]
    #[serial_test::serial(config_persist)]
    async fn a_set_reports_the_value_in_use_and_refuses_a_guessed_name() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;

        let stored = set("manager_model", "acme/model").await.unwrap();
        assert_eq!(stored, "manager_model = acme/model");
        assert_eq!(
            get("manager_model").await.unwrap(),
            "manager_model = acme/model",
            "the read-back sees what the write left"
        );

        // A cleared row is not an empty answer: the built-in default is what is in use.
        let cleared = set("manager_model", "").await.unwrap();
        assert!(cleared.contains("(default)"), "got: {cleared}");

        let guessed = set("provider_ky", "sk-or-v1-0123456789")
            .await
            .unwrap_err()
            .to_string();
        assert!(guessed.contains("`reference` action"), "got: {guessed}");
    }

    /// The route refuses a call it cannot apply, before any store is touched: a source build
    /// updates and registers nothing at login, an endpoint key names nothing alone, a local
    /// server's credentials must have the shape Telegram issues, the provider's rows are not
    /// cleared here, the routing form needs its model, and a catalogue is bounded.
    #[tokio::test]
    #[serial_test::serial(config_persist)]
    async fn a_set_refuses_what_it_cannot_apply() {
        let alone = set("provider_endpoint_key", "local-key")
            .await
            .unwrap_err()
            .to_string();
        assert!(
            alone.contains("`setup`") && alone.contains("provider_endpoint"),
            "a key alone must name where it is written: {alone}"
        );

        let bad_id = set("telegram_api_id", "abc").await.unwrap_err().to_string();
        assert!(bad_id.contains("must be a number"), "got: {bad_id}");
        let bad_hash = set("telegram_api_hash", "not-a-hash")
            .await
            .unwrap_err()
            .to_string();
        assert!(
            bad_hash.contains("32 hexadecimal characters"),
            "got: {bad_hash}"
        );

        // The provider's rows cannot be cleared through `set`: the endpoint goes back only through
        // `setup`'s explicit request, and the key is only ever replaced.
        let endpoint = set("provider_endpoint", "").await.unwrap_err().to_string();
        assert!(
            endpoint.contains("clear_provider_endpoint"),
            "a cleared endpoint must name the route that settles it: {endpoint}"
        );
        let key = set("provider_key", "").await.unwrap_err().to_string();
        assert!(key.contains("cannot be cleared"), "got: {key}");

        // The routing form names its own shape when the model is left off.
        let bare = get("provider_order").await.unwrap_err().to_string();
        assert!(bare.contains("needs a model"), "got: {bare}");

        // A catalogue is bounded: a model-driven `set` cannot store an arbitrarily long row, and
        // the refusal names the reference.
        let too_many = vec!["acme/model"; MAX_CATALOGUE_ENTRIES + 1].join("\n");
        let capped = set("image_gen_models", &too_many)
            .await
            .unwrap_err()
            .to_string();
        assert!(
            capped.contains("at most 32 models") && capped.contains("`reference` action"),
            "got: {capped}"
        );
        // So is one entry: a video catalogue takes no validation, so the length is the only bound.
        let long = "x".repeat(MAX_MODEL_NAME_CHARS + 1);
        let refused = set("video_models", &long).await.unwrap_err().to_string();
        assert!(refused.contains("at most 128 characters"), "got: {refused}");

        // These tests run from a source tree, so neither switch can be written.
        for setting in ["autoupdate", "autolaunch"] {
            let refused = set(setting, "on").await.unwrap_err().to_string();
            assert!(refused.contains("this build does not"), "got: {refused}");
        }
    }

    /// A value longer than one read-back can deliver is refused before anything is stored: `get`
    /// promises the value whole, so a row it could never return that way must not exist. Four-byte
    /// characters are what makes the bound a byte bound — the same value in ASCII would fit.
    #[tokio::test]
    #[serial_test::serial(config_persist)]
    async fn a_value_too_long_to_read_back_is_refused() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;
        set("manager_model", "").await.expect("settle the slot");

        let long = "😀".repeat(MAX_READ_BACK_BYTES / 4);
        for name in ["manager_model", "video_models"] {
            let refused = set(name, &long).await.unwrap_err().to_string();
            assert!(
                refused.contains("cannot be stored") && refused.contains("`reference` action"),
                "a value `get` could not deliver whole is refused, naming the reference: {refused}"
            );
        }

        // A credential reads back as a hint, so the bound cannot apply to it.
        set("exa_key", &long)
            .await
            .expect("a credential is never read back whole");
        let hint = get("exa_key").await.expect("a masked read-back");
        assert!(hint.len() < 64, "a credential reads back as a hint: {hint}");
        set("exa_key", "").await.expect("settle the key");

        // Nothing was stored, so the read-back is still the built-in default.
        assert!(
            get("manager_model").await.unwrap().contains("(default)"),
            "a refused value never lands"
        );

        // A row another surface left too long — the window has no such bound — is refused rather
        // than returned in half.
        crate::config::persist_settled_string_field(CONFIG_KEY_MANAGER_MODEL, &long)
            .await
            .expect("a row another surface could have written");
        let refused = get("manager_model").await.unwrap_err().to_string();
        assert!(
            refused.contains("longer than one read-back") && refused.contains("`reference` action"),
            "a stored value `get` cannot deliver whole is refused: {refused}"
        );
        crate::config::persist_settled_string_field(CONFIG_KEY_MANAGER_MODEL, "")
            .await
            .expect("settle the slot");
    }

    /// A write's answer is refused, never cut, when its own report outgrew one read-back — and the
    /// refusal says the write landed, so a caller is never told nothing happened.
    #[test]
    fn a_write_answer_too_long_for_one_read_back_is_refused() {
        let fits = "x".repeat(MAX_READ_BACK_BYTES);
        assert_eq!(
            whole_answer("manager_model", fits.clone()).expect("the bound itself fits"),
            fits
        );

        let refused = whole_answer("provider_key", "y".repeat(MAX_READ_BACK_BYTES + 1))
            .unwrap_err()
            .to_string();
        assert!(refused.contains("was written"), "got: {refused}");
        assert!(
            refused.contains("`reference` action"),
            "the refusal names where the rule is written: {refused}"
        );
    }

    /// Start-at-login is one description, not two: the write's answer and the read-back spell the
    /// same decision, the same registration — `unknown` when the operating system did not say —
    /// and the same inert suffix, `off` included.
    #[tokio::test]
    async fn start_at_login_reads_the_same_either_way() {
        // The one case that is not the OS's own answer: a registration nobody read is not reported
        // at all, so the summary never claims a state it did not look at.
        for (decision, word) in [(true, "on"), (false, "off")] {
            let summary = autolaunch_line(decision, Registration::NotRead);
            assert!(
                summary.starts_with(word),
                "the summary says the decision: {summary}"
            );
            assert!(
                !summary.contains("login registration"),
                "the summary says the decision alone: {summary}"
            );
        }

        // The OS's own answer is the clause both the write and the read-back spell, `unknown`
        // included.
        for (registration, expected) in [
            (Registration::Unknown, "unknown"),
            (Registration::Known(true), "in place"),
            (Registration::Known(false), "not in place"),
        ] {
            let line = autolaunch_line(true, registration);
            assert!(
                line.contains(&format!("(login registration: {expected})")),
                "the write and the read-back say the same: {line}"
            );
        }

        // In a source tree neither can register, and the refusal repeats the read-back's reason.
        // `off` carries the suffix too: `off (inert — …)` is the line a source build prints.
        if !autolaunch_available() {
            for (decision, word) in [(true, "on"), (false, "off")] {
                let line = autolaunch_line(decision, Registration::Unknown);
                assert!(
                    line.starts_with(word)
                        && line.contains(&format!("(inert — {AUTOLAUNCH_INERT})")),
                    "got: {line}"
                );
                let refused = set_autolaunch(word).await.unwrap_err().to_string();
                assert!(refused.contains(AUTOLAUNCH_INERT), "got: {refused}");
            }
        }
    }

    /// A switch's description says only what the code can prove: transcription off is inertness
    /// for the wake word, and an off switch says nothing at all about the host.
    #[tokio::test]
    async fn a_switch_says_only_what_it_can_prove() {
        let mut config = ConfigData::STRUCT_FIELDS_DEFAULT;
        // An off switch asserts nothing about the device — the off path never looked at it.
        for setting in [Setting::VoiceEnabled, Setting::TtsEnabled] {
            assert_eq!(setting.describe(&config, Detail::SETTLED).await, "off");
        }

        // On while transcription is off: inert, and it says why.
        config.voice_enabled = Some("true".to_string());
        config.audio_transcription_use_local = Some("false".to_string());
        assert_eq!(
            Setting::VoiceEnabled
                .describe(&config, Detail::SETTLED)
                .await,
            "on (inert — transcription is off)"
        );

        // Transcription on: the wake word is no longer inert for that reason.
        config.audio_transcription_use_local = Some(String::new());
        let on = Setting::VoiceEnabled
            .describe(&config, Detail::SETTLED)
            .await;
        assert!(
            !on.contains("transcription is off"),
            "transcription is on, so the wake word is not inert for that reason: {on}"
        );
    }

    /// A switch has one story: what `set` answers is what `get` reads and what `list` prints, all
    /// three from [`Setting::describe`].
    #[tokio::test]
    #[serial_test::serial(config_persist)]
    async fn a_switch_reads_the_same_way_from_set_get_and_list() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;
        // The switch moves the voice pipeline, whose state must exist for the process.
        let _ = crate::audio::voice::init_global();
        let previous = CONFIG.audio_transcription_use_local();
        let _ = CONFIG.set_string_field(CONFIG_KEY_AUDIO_TRANSCRIPTION_USE_LOCAL, "");

        let set_line = set("voice_enabled", "on").await.unwrap();
        assert_eq!(get("voice_enabled").await.unwrap(), set_line);
        assert!(
            list().await.contains(&format!("{set_line}\n")),
            "the summary must print the line `get` reads: {set_line}"
        );

        // The gate refuses to turn the wake word on while transcription is off, so an inert row is
        // one written before that — and the three reads still agree on what it means.
        let _ = CONFIG.set_string_field(CONFIG_KEY_VOICE_ENABLED, "true");
        let _ = CONFIG.set_string_field(CONFIG_KEY_AUDIO_TRANSCRIPTION_USE_LOCAL, "false");
        let line = get("voice_enabled").await.unwrap();
        assert_eq!(line, "voice_enabled = on (inert — transcription is off)");
        assert!(list().await.contains(&format!("{line}\n")), "got: {line}");

        let _ = CONFIG.set_string_field(CONFIG_KEY_VOICE_ENABLED, "");
        let _ = CONFIG.set_string_field(
            CONFIG_KEY_AUDIO_TRANSCRIPTION_USE_LOCAL,
            previous.as_deref().unwrap_or_default(),
        );
    }

    /// A switch reads the words a person and a model both reach for, and refuses anything else
    /// rather than silently reading it as "off". The Telegram mode's `local` / `cloud` are not
    /// switch words: they have their own strict parser.
    #[test]
    fn switches_read_the_words_both_callers_use() {
        for on in ["on", "TRUE", "yes", "1", "enabled"] {
            assert!(parse_switch(on).unwrap(), "{on} must read as on");
        }
        for off in ["off", "False", "no", "0", "disabled"] {
            assert!(!parse_switch(off).unwrap(), "{off} must read as off");
        }
        for refused in ["maybe", "local", "cloud"] {
            assert!(parse_switch(refused).is_err(), "{refused} is not a switch");
        }
        assert_eq!(
            crate::channels::telegram_local::Mode::parse("local"),
            Some(crate::channels::telegram_local::Mode::Local)
        );
        assert_eq!(
            crate::channels::telegram_local::Mode::parse("cloud"),
            Some(crate::channels::telegram_local::Mode::Cloud)
        );
        assert_eq!(crate::channels::telegram_local::Mode::parse("on"), None);
    }
}
