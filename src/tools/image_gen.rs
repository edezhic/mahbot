//! The `image_gen` tool: text-to-image and image-to-image through OpenRouter's
//! dedicated Image API, walking the account's model picker and returning the
//! file path the reply can embed.
//!
//! Not the media catalog itself (`tools::media_catalog`) and not image
//! analysis — this tool resolves a model, sends one request and downloads the
//! result.

use crate::retry::RETRY_AFTER_MAX_MS;
use crate::tools::media_catalog::image::{ImageModelInfo, resolve_image_model};
use crate::util::error::retry_after_header;
use crate::util::http::{bearer_auth_header, read_error_body};
use crate::{Tool, Workspace};
use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Tool for generating images via OpenRouter's dedicated Image API.
///
/// Supports text-to-image and image-to-image generation. Accepts multiple
/// reference images on input. Returns the path to the generated file so the
/// agent can embed it as `[IMAGE:path]` in its reply.
///
/// One attempt walks the image models the account's picker offers: its own model
/// first, exactly as it was always tried, then each other model once, in picker
/// order, until one serves the request. A request a fallback covered is a success
/// for the user; when another model produced the picture, the tool's result names
/// it and says what happened with the account's own model.
pub struct ImageGenTool;

impl ImageGenTool {
    /// The tool's own name — what [`Tool::name`] reports and what the failure
    /// envelope is measured with.
    const NAME: &'static str = "image_gen";
}

#[async_trait]
impl Tool for ImageGenTool {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn media_marker(&self) -> Option<&'static str> {
        Some("[IMAGE:")
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "prompt": {
                    "type": "string",
                    "description": "Text description of the image to generate"
                },
                "images": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Paths to reference images for image-to-image generation (local paths are accepted only from the workspace uploads dir — received attachments — or the generated dir — previously generated images)"
                },
                "aspect_ratio": {
                    "type": "string",
                    "description": "Aspect ratio (e.g. 16:9, 1:1, 4:3)"
                },
                "size": {
                    "type": "string",
                    "description": "Image size — a tier (e.g. 1K, 2K) or explicit pixels (e.g. 1024x1024); a model that cannot express the value is not sent the request"
                }
            }),
            &["prompt"],
        )
    }

    async fn execute(&self, ws: &Workspace, args: serde_json::Value) -> anyhow::Result<String> {
        let generated = self.generate(ws, &args, &HttpSender).await?;
        Ok(self.result_text(&generated))
    }

    /// The generated picture travels with the tool's own result: the walk decided
    /// which model produced it, and the note naming that model has to reach the
    /// assistant — which receives the image's annotation IN PLACE of the tool's own
    /// text once the picture is attached, so the note rides on the payload.
    async fn execute_with_payloads(
        &self,
        ws: &Workspace,
        args: serde_json::Value,
    ) -> anyhow::Result<super::ToolOutput> {
        let generated = self.generate(ws, &args, &HttpSender).await?;
        Ok(self.output(&generated).await)
    }
}

/// A finished generation: the picture on disk and the note that must reach the
/// assistant with it (`None` when the account's own model produced it, so there is
/// nothing to disclose).
struct Generated {
    path: PathBuf,
    note: Option<String>,
}

impl Generated {
    /// The payload that carries the picture into the conversation as a native
    /// image, the note riding on its annotation. `None` when the saved file cannot
    /// be re-encoded into one (fail-open, e.g. an SVG): the marker then stays in the
    /// tool result's own text, and the note with it.
    async fn payload(&self) -> Option<super::ImagePayload> {
        let meta = crate::util::local_image_to_compressed_data_uri_with_meta(&self.path)
            .await
            .ok()?;
        let mut payload = super::ImagePayload::from_compressed_meta(
            &self.path,
            meta,
            None,
            super::ImagePayloadSource::Generated,
        );
        payload.note.clone_from(&self.note);
        Some(payload)
    }
}

impl ImageGenTool {
    /// Generate one picture: the account's own image model first, then every other
    /// image model its picker offers, until one serves the request.
    async fn generate(
        &self,
        ws: &Workspace,
        args: &serde_json::Value,
        sender: &dyn ImageSender,
    ) -> anyhow::Result<Generated> {
        let prompt = super::get_str(args, "prompt")?;
        let user_name = crate::agent::tool_user_name();
        let own_model = crate::users::resolve_image_gen_model(&user_name).await;
        let aspect_ratio_arg = super::get_opt_str(args, "aspect_ratio");
        let size_arg = super::get_opt_str(args, "size");
        let images: Vec<String> = super::get_str_array(args, "images")?;

        // Confine every reference before any other filesystem or provider access:
        // a refused path is never stat'ed or read, and the plan below cannot
        // precede the refusal.
        let mut reference_paths = Vec::with_capacity(images.len());
        for img_path in &images {
            reference_paths
                .push(super::resolve_local_media_path(img_path, ws, "reference image").await?);
        }

        // What the walk will do, decided before any reference is loaded.
        let plan = plan_walk(
            &own_model,
            reference_paths.len(),
            size_arg,
            &crate::config::CONFIG.image_gen_models_raw(),
        )
        .await?;

        // Combined-size pre-flight before loading any file: the per-image ceilings
        // don't bound the total, and a pathological multi-reference request must be
        // refused without reading everything into memory first.
        if !reference_paths.is_empty() {
            crate::util::check_reference_total_input(&reference_paths).await?;
        }

        // Load reference images so file errors surface deterministically.
        let mut references = Vec::with_capacity(reference_paths.len());
        for path in &reference_paths {
            references.push(
                crate::util::load_reference_image(path, super::MAX_REFERENCE_IMAGE_BYTES).await?,
            );
        }

        let resolved_aspect_ratio =
            request_aspect_ratio(aspect_ratio_arg, &reference_paths, plan.catalog_reached);

        // Image generation always targets OpenRouter — a custom
        // chat endpoint never serves image models.
        let api_base = crate::providers::ensure_base_url(crate::config::DEFAULT_PROVIDER_ENDPOINT);
        let images_url = format!("{api_base}/images");
        // A key the product does not hold is a failure of preparing the request,
        // not a refusal any model could fix, so nothing is walked.
        let auth = bearer_auth_header().ok_or_else(|| {
            anyhow::anyhow!("Image generation: provider API key is not configured")
        })?;

        let mut walk = ImageWalk {
            request: RequestMaterial {
                prompt,
                aspect_ratio: resolved_aspect_ratio,
                size_arg,
                references: &mut references,
            },
            own_model: &own_model,
            url: &images_url,
            auth: &auth,
            sender,
            trail: Vec::new(),
            stopped: None,
        };

        // The walk is raced against global shutdown so a long generation cannot
        // block process teardown.
        let shutdown = crate::shutdown::shutdown_token();
        let walked = tokio::select! {
            () = shutdown.cancelled() => {
                anyhow::bail!("{SHUTDOWN_ABORTED}");
            }
            walked = walk.run(plan.own_step, plan.candidates) => walked?,
        };

        let Walked::Served {
            model: served_by,
            response,
        } = walked
        else {
            // Nothing served the request: the failure names what was tried and what
            // each model answered.
            anyhow::bail!("{}", walk.failure_message());
        };

        let (b64_json, media_type) = match extract_response_image(&response) {
            Ok(x) => x,
            Err(err) => {
                // A 2xx body carrying a provider error (or no image data)
                // means the attempt completed and was billed — never walked to
                // another model, never retried.
                anyhow::bail!("Image generation response error: {err}");
            }
        };

        let bytes = match STANDARD.decode(b64_json.as_bytes()) {
            Ok(bytes) => bytes,
            Err(e) => {
                // Provider returned malformed base64 — a response-processing
                // failure, and a billed one: never walked to another model.
                anyhow::bail!("Failed to decode base64 image data from response: {e}");
            }
        };

        let output_path = super::save_generated_file(
            ws,
            &bytes,
            "image",
            extension_for_media_type(media_type.as_deref()),
        )
        .await?;

        // The record the walk leaves behind: which model produced the picture, so a
        // fallback stays diagnosable afterwards.
        tracing::info!(
            model = %served_by,
            own_model = %own_model,
            "Image generation produced a picture"
        );

        Ok(Generated {
            path: output_path,
            note: walk.fallback_note(&served_by),
        })
    }

    /// One generated picture as the tool's own result: the marker text, and the picture
    /// as a payload whose annotation the assistant reads in place of that text.
    async fn output(&self, generated: &Generated) -> super::ToolOutput {
        super::ToolOutput {
            text: self.result_text(generated),
            image_payloads: generated.payload().await.into_iter().collect(),
            text_is_content: false,
        }
    }

    /// The tool result text: the delivery marker, then the note naming the model
    /// that produced the picture when it was not the account's own.
    fn result_text(&self, generated: &Generated) -> String {
        let marker = self.format_media_result(&generated.path);
        match &generated.note {
            Some(note) => format!("{marker}\n{note}"),
            None => marker,
        }
    }
}

// ── The walk: the account's own model, then the picker's other models ────

/// The request every model's body is built from: the prompt, the aspect ratio the
/// request resolved to, and the references it carries — already confined, loaded
/// and compressed.
struct RequestMaterial<'a> {
    prompt: &'a str,
    /// The aspect ratio the request resolved to — the user's own argument on a
    /// fail-open attempt (no catalog), and the value the request resolved to
    /// otherwise. Sent only to a model whose own declarations carry the parameter.
    aspect_ratio: Option<String>,
    size_arg: Option<&'a str>,
    references: &'a mut [crate::util::ReferenceImage],
}

/// One model's step in the walk.
enum Step {
    /// Attempt it: the catalog declarations that shape its body — `None` only for
    /// the account's own fail-open attempt (the catalog was unreachable, so only
    /// what the user provided is sent) — and the size those declarations accept.
    Attempt {
        info: Option<ImageModelInfo>,
        size: Option<(&'static str, String)>,
    },
    /// Never sent to, with the reason.
    Skip(String),
}

/// What one visited model came to.
enum Visited {
    /// Nothing was sent to it, with the reason.
    Skipped(String),
    /// Sent, and this is what came back. A failure that may have been produced and
    /// charged is not this entry's business: it stops the walk, and the walk's stop
    /// reason says so.
    Attempted(Vec<AttemptFailure>),
}

/// What the walk came to.
enum Walked {
    /// The provider served `response`, produced by `model`.
    Served {
        model: String,
        response: serde_json::Value,
    },
    /// Nothing served the request — what every visited model answered is in the
    /// walk's trail.
    Unserved,
}

/// What the walk will do, decided before any reference is loaded: the account's own
/// model's step, the other models its picker offers, and whether the catalog was
/// reached (with no catalog no model's declarations are known, so the request keeps
/// its fail-open shape).
struct WalkPlan {
    own_step: Step,
    candidates: Vec<String>,
    catalog_reached: bool,
}

/// Decide the walk before any reference is loaded. `Err` refuses the request before
/// anything is sent — the provider's own reference ceiling, or the account's own
/// model refusing with nothing else configured to walk to.
async fn plan_walk(
    own_model: &str,
    reference_count: usize,
    size_arg: Option<&str>,
    picker: &[String],
) -> anyhow::Result<WalkPlan> {
    // The provider's own ceiling on how many references a request may carry at all
    // is a property of the REQUEST, not of any model: nothing can take such a
    // request, so it is refused before the catalog — or any model — is consulted.
    // This is also the wording the user gets when the request is over both that
    // ceiling and the account's own model's lower cap: the provider's limit is the one
    // no model can move, so it is the one worth naming.
    if reference_count > super::MAX_REFERENCE_IMAGES_PER_REQUEST {
        anyhow::bail!(
            "Image generation supports at most {} reference image(s), got {}. \
             Retry with fewer images.",
            super::MAX_REFERENCE_IMAGES_PER_REQUEST,
            reference_count,
        );
    }

    // Capability comes from the catalog: a membership miss forces one refresh before
    // the verdict, a catalog outage degrades to minimal user-provided parameters
    // (fail-open), and a model confirmed absent from a freshly fetched catalog is
    // the walk's first reason instead of the end of the attempt.
    let capability = resolve_image_model(crate::config::DEFAULT_PROVIDER_ENDPOINT, own_model).await;
    // Without a reachable catalog no model's capability can be established, so
    // nothing is walked and the attempt keeps its fail-open behaviour.
    let catalog_reached = !matches!(capability, Ok(None));
    let own_step = own_step(own_model, capability, reference_count, size_arg);

    // The other models the account could have picked, in picker order — the list
    // exactly as configured (`image_gen_models_raw`, never the accessor that fills
    // in a default): a default the picker never showed was never offered to the
    // account, so it is not a candidate.
    let candidates = if catalog_reached {
        fallback_candidates(own_model, picker)
    } else {
        Vec::new()
    };

    // With nothing else to walk to, a refusal before anything is sent ends the
    // attempt exactly as it always did — before the request's material is touched,
    // so which of two faults is reported stays as it was.
    if candidates.is_empty()
        && let Step::Skip(reason) = &own_step
    {
        anyhow::bail!("{reason}");
    }
    Ok(WalkPlan {
        own_step,
        candidates,
        catalog_reached,
    })
}

/// The other image models the account could have picked: the picker list, in picker
/// order, with every entry of the model already tried — a repeated one included —
/// and repeats of any other entry left out, so each model is visited once. Only
/// models the picker offered are ever used: a default that fills in when nothing has
/// been chosen is not one of them.
fn fallback_candidates(own_model: &str, picker_list: &[String]) -> Vec<String> {
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::from([own_model]);
    picker_list
        .iter()
        .filter(|model| seen.insert(model.as_str()))
        .cloned()
        .collect()
}

/// The account's own model's step, decided exactly as that model was validated
/// before the walk existed: its capability verdict, then the reference cap and the
/// size its declarations accept. A refusal becomes the walk's first reason instead
/// of the end of the attempt.
fn own_step(
    model: &str,
    capability: anyhow::Result<Option<ImageModelInfo>>,
    references: usize,
    size_arg: Option<&str>,
) -> Step {
    let info = match capability {
        Ok(info) => info,
        Err(e) => return Step::Skip(e.to_string()),
    };
    match info {
        // Catalog unreachable: fail-open, exactly as it always was — no capability is
        // known, so nothing is checked and the requested size is dropped before
        // [`resolve_size`] is ever consulted.
        None => {
            if let Some(size) = size_arg {
                tracing::debug!(size, "Image generation: size dropped — catalog unavailable");
            }
            Step::Attempt {
                info: None,
                size: None,
            }
        }
        Some(info) => match eligibility(model, &info, references, size_arg, RefusalVoice::OwnModel)
        {
            Err(reason) => Step::Skip(reason),
            Ok(size) => Step::Attempt {
                info: Some(info),
                size,
            },
        },
    }
}

/// Whom a pre-send refusal speaks to. The account's own model's attempt refuses the
/// user, who can put another model in that pick — today's `usage:`/`hint:` text. The
/// walk's line about a model it merely passed over states the fact alone: the account
/// never picked that model, so neither the header nor the advice applies to it.
#[derive(Clone, Copy)]
enum RefusalVoice {
    OwnModel,
    Candidate,
}

/// The reference cap and the size checks one model's own declarations decide: `Ok`
/// with the size they accept (if any) when it can serve the request, `Err` with the
/// reason when it cannot — worded for `voice`. Only reached for a model whose
/// declarations are known — with no catalog the check is skipped entirely (see
/// [`own_step`]).
fn eligibility(
    model: &str,
    info: &ImageModelInfo,
    references: usize,
    size_arg: Option<&str>,
    voice: RefusalVoice,
) -> Result<Option<(&'static str, String)>, String> {
    if references > 0 {
        validate_reference_count(model, info, references, voice)?;
    }
    resolve_size(model, size_arg, info, voice).map_err(|e| e.to_string())
}

/// The aspect ratio the request carries to every model: the value the request
/// resolved to when the catalog was reached (auto-detected from the references
/// unless the user named one), and the user's own argument alone on a fail-open
/// attempt — auto-detection exists to pick a value a catalog-declared parameter can
/// carry, which a catalog-less attempt has none of.
fn request_aspect_ratio(
    aspect_ratio_arg: Option<&str>,
    reference_paths: &[PathBuf],
    catalog_reached: bool,
) -> Option<String> {
    if catalog_reached {
        Some(resolve_aspect_ratio(aspect_ratio_arg, reference_paths))
    } else {
        aspect_ratio_arg.map(String::from)
    }
}

/// The room everything around the walk failure text takes before the model receives it:
/// the wrapper `Agent::execute_tool` raises a fault with, and the failure envelope
/// [`crate::tools::tool_failure_envelope_bytes`] accounts for — derived from those
/// pieces rather than restated, so neither wrapper can drift from this budget.
const WALK_ENVELOPE_BYTES: usize =
    crate::tools::tool_failure_envelope_bytes(ImageGenTool::NAME.len());

/// The walk failure text's own budget: what a tool result is truncated to, less
/// everything wrapped around that text.
const WALK_MESSAGE_BYTES: usize = crate::util::TOOL_OUTPUT_BUDGET_BYTES - WALK_ENVELOPE_BYTES;

/// The most room one visited model's line takes — and the most the fallback note gives
/// what it says about the account's own model.
const WALK_ENTRY_MAX_BYTES: usize = 300;

/// The room one visited model's line has: the failure's budget less the header and the
/// stop line, shared out over the lines and capped — counted in bytes, so a multi-byte
/// name or answer cannot eat a neighbour's room, and shared out before the names are
/// written, so a picker whose names alone outgrow the budget cannot push the message past
/// it either.
fn walk_line_bytes(models: usize, fixed: usize) -> usize {
    WALK_ENTRY_MAX_BYTES.min(WALK_MESSAGE_BYTES.saturating_sub(fixed) / models.max(1))
}

/// The prefix the failure text gives the reason a walk stopped before exhausting its
/// list, shared by both failure formatters.
const WALK_STOPPED_PREFIX: &str = "The walk stopped there: ";

/// A generation cut short by process shutdown, on both the outer race and a sender
/// that reports the abort mid-walk.
const SHUTDOWN_ABORTED: &str = "Shutting down — image generation aborted";

/// A candidate whose capability cannot be judged because the catalog could not be
/// reached.
const CATALOG_UNREACHABLE: &str =
    "the image-models catalog could not be reached, so the model's capability is unknown";

/// A candidate the catalog does not offer for image output — absent from it, or listed
/// as text-only. The walk says so in its own words: the resolver's message speaks to
/// the account's pick, which a model the walk passed over is not.
const NOT_AN_IMAGE_MODEL: &str = "the image-models catalog does not offer it for image output";

/// The walk over the account's own image model and the other models its picker
/// offers: the own model first, then each other model once, in picker order, until
/// one serves the request. Every visited model's answer is kept, so a failure can
/// name what each of them answered or why it was never sent to.
struct ImageWalk<'a> {
    request: RequestMaterial<'a>,
    own_model: &'a str,
    url: &'a str,
    auth: &'a str,
    /// The POST the walk makes — its only outward step, and the seam a test drives
    /// with scripted answers instead of a live provider.
    sender: &'a dyn ImageSender,
    /// What every visited model came to, in visit order.
    trail: Vec<(String, Visited)>,
    /// Why the walk stopped before exhausting the candidates, when it did.
    stopped: Option<String>,
}

impl ImageWalk<'_> {
    /// Run the walk: `own_step` for the account's own model, then `candidates` — the
    /// other models the account's picker offers, in picker order, with the tried
    /// model and repeats of any entry already left out.
    async fn run(&mut self, own_step: Step, candidates: Vec<String>) -> anyhow::Result<Walked> {
        let own_model = self.own_model;
        if let Some(walked) = self
            .visit(own_model, own_step, !candidates.is_empty())
            .await?
        {
            return Ok(walked);
        }

        for (index, model) in candidates.iter().enumerate() {
            // A visit that left its outcome unknown — or found the request itself
            // unsendable — set the reason: nothing after it may be tried.
            if self.stopped.is_some() {
                break;
            }
            let more_models = index + 1 < candidates.len();
            let Some(step) = self.candidate_step(model).await else {
                // The catalog's loss is stated on this model's own line; the stop
                // line says only that the walk ended here.
                self.trail.push((
                    model.clone(),
                    Visited::Skipped(CATALOG_UNREACHABLE.to_string()),
                ));
                if more_models {
                    self.stopped = Some("the remaining models were not checked".to_string());
                }
                break;
            };
            if let Some(walked) = self.visit(model, step, more_models).await? {
                return Ok(walked);
            }
        }
        Ok(Walked::Unserved)
    }

    /// Prepare one candidate: its catalog declarations checked against the request,
    /// so the step to visit it with is known before anything is sent to it.
    ///
    /// `None` when the catalog itself could not be reached — this model's capability
    /// is unknown, and so is every later candidate's.
    async fn candidate_step(&mut self, model: &str) -> Option<Step> {
        let info = match resolve_image_model(crate::config::DEFAULT_PROVIDER_ENDPOINT, model).await
        {
            Ok(Some(info)) => info,
            // Reached the catalog and it does not list the model, or lists it as
            // text-only: that model's own limitation, and the walk goes on.
            Err(_) => return Some(Step::Skip(NOT_AN_IMAGE_MODEL.to_string())),
            Ok(None) => return None,
        };
        match eligibility(
            model,
            &info,
            self.request.references.len(),
            self.request.size_arg,
            RefusalVoice::Candidate,
        ) {
            Ok(size) => Some(Step::Attempt {
                info: Some(info),
                size,
            }),
            Err(reason) => Some(Step::Skip(reason)),
        }
    }

    /// Visit one model: skip it when it can never be sent the request, otherwise
    /// attempt it and read what came back.
    ///
    /// `more_models` is whether any model is left after this one, so a visit whose
    /// outcome ends the walk says why it stopped only when something was actually
    /// left untried — with the list exhausted there is nothing to explain.
    async fn visit(
        &mut self,
        model: &str,
        step: Step,
        more_models: bool,
    ) -> anyhow::Result<Option<Walked>> {
        let (info, size) = match step {
            Step::Skip(reason) => {
                tracing::debug!(model, %reason, "Image generation: model cannot serve the request");
                self.trail
                    .push((model.to_string(), Visited::Skipped(reason)));
                return Ok(None);
            }
            Step::Attempt { info, size } => (info, size),
        };
        // A body the client-side budget cannot fit is the REQUEST's own limit: no
        // declaration of any other model changes it. On the account's own model
        // nothing has been sent yet, so the attempt is refused as it always was;
        // mid-walk the walk stops there, naming the model the request could no longer
        // be built for.
        let body = match self.body(model, info.as_ref(), size) {
            Ok(body) => body,
            Err(e) if model == self.own_model => return Err(e),
            Err(e) => {
                let reason = e.to_string();
                tracing::warn!(
                    model,
                    %reason,
                    "Image generation: request body too large — no other model is tried"
                );
                self.stopped = Some(format!("`{model}` was not sent to: {reason}"));
                return Ok(None);
            }
        };
        match generate_image_with_retries(self.url, &body, self.auth, self.sender).await {
            Ok(response) => Ok(Some(Walked::Served {
                model: model.to_string(),
                response,
            })),
            Err(AttemptOutcome::WalkOn(failures)) => {
                tracing::info!(
                    model,
                    reason = %describe_failure(&failures, self.has_references()),
                    "Image generation: model did not serve the request"
                );
                self.trail
                    .push((model.to_string(), Visited::Attempted(failures)));
                Ok(None)
            }
            Err(AttemptOutcome::Unknown(failures)) => {
                let reason = describe_failure(&failures, self.has_references());
                tracing::warn!(
                    model,
                    %reason,
                    "Image generation: outcome unknown — no other model is tried"
                );
                if more_models {
                    self.stopped = Some(format!(
                        "`{model}` may have completed and been billed — its outcome is unknown, \
                         so no other model was tried"
                    ));
                }
                self.trail
                    .push((model.to_string(), Visited::Attempted(failures)));
                Ok(None)
            }
            Err(AttemptOutcome::Aborted) => {
                anyhow::bail!("{SHUTDOWN_ABORTED}");
            }
        }
    }

    /// Build one model's attempt request from its own declarations and fit it to the
    /// aggregate body budget. `info` is `None` only for the account's own model's
    /// fail-open attempt (the catalog was unreachable): it sends only what the user
    /// provided itself, with no requested size — which is exactly what
    /// [`RequestMaterial::aspect_ratio`] holds for it.
    fn body(
        &mut self,
        model: &str,
        info: Option<&ImageModelInfo>,
        size: Option<(&'static str, String)>,
    ) -> anyhow::Result<serde_json::Value> {
        let mut body = build_request_body(
            model,
            self.request.prompt,
            self.request.aspect_ratio.as_deref(),
            size,
            self.request.references,
            info,
        );
        // Aggregate body budget: per-image caps don't bound multi-reference
        // totals, so compress references further until the serialized request fits
        // the ~2 MB client-side body budget (not a documented OpenRouter number).
        // The references belong to the REQUEST, not to one model, so the first fit
        // releases their retained bytes.
        super::fit_request_body_budget(
            &mut body,
            self.request.references,
            super::MAX_REQUEST_BODY_BYTES,
        )?;
        Ok(body)
    }

    /// Whether the request carries reference images — the refusal hint about the
    /// provider possibly accepting fewer of them may only appear when it does.
    fn has_references(&self) -> bool {
        !self.request.references.is_empty()
    }

    /// The failure text: with nothing else configured to walk to, exactly what a lone
    /// attempt reported before the walk existed; otherwise what every visited model
    /// answered, what was skipped and why, and why the walk stopped where it did.
    fn failure_message(&self) -> String {
        let [(_, visited)] = &self.trail[..] else {
            return self.walk_failure();
        };
        let message = match visited {
            Visited::Skipped(reason) => reason.clone(),
            Visited::Attempted(failures) => build_terminal_message(failures, self.has_references()),
        };
        // The walk stopped at its first model: with nothing visited after it, the
        // lone attempt's message stands, and the reason nothing else was tried
        // follows it.
        match &self.stopped {
            Some(stopped) => format!("{message}\n{WALK_STOPPED_PREFIX}{stopped}"),
            None => message,
        }
    }

    /// The failure that names every model the walk visited, inside the room the model
    /// receives for it: what one line cannot hold is cut like anything else.
    fn walk_failure(&self) -> String {
        let header = "Image generation failed: no configured image model served the request.";
        let stop = match &self.stopped {
            Some(stopped) => format!("\n{WALK_STOPPED_PREFIX}{stopped}"),
            None => String::new(),
        };
        let line_bytes = walk_line_bytes(self.trail.len(), header.len() + stop.len());

        let mut out = String::from(header);
        for (model, visited) in &self.trail {
            out.push_str(&crate::util::cut_bytes(
                &self.line(model, visited),
                line_bytes,
            ));
        }
        out.push_str(&stop);
        out
    }

    /// One visited model's line: its own opening — the name, and the label marking the
    /// account's own pick — with what it answered or why it was never sent to.
    fn line(&self, model: &str, visited: &Visited) -> String {
        let own = if model == self.own_model {
            " (the account's image model)"
        } else {
            ""
        };
        let described = match visited {
            Visited::Skipped(reason) => format!("not sent to — {reason}"),
            Visited::Attempted(failures) => describe_failure(failures, self.has_references()),
        };
        format!("\n- `{model}`{own}: {described}")
    }

    /// The note the assistant receives with a picture another model produced: which
    /// model made it, and what happened with the account's own model (the walk's
    /// first entry). `None` when the account's own model produced it — there is
    /// nothing to disclose.
    fn fallback_note(&self, served_by: &str) -> Option<String> {
        if served_by == self.own_model {
            return None;
        }
        let (_, own_visited) = self.trail.first()?;
        let own_fate = match own_visited {
            Visited::Skipped(reason) => format!("was never sent to: {reason}"),
            Visited::Attempted(failures) => format!(
                "did not serve the request: {}",
                describe_failure(failures, self.has_references())
            ),
        };
        // The note carries the provider's own answer, and the annotation it rides on is
        // not scrubbed anywhere downstream — unlike the failure path and the tool text
        // this note also feeds.
        let fate = crate::util::cut_bytes(
            &crate::util::scrub_credentials(&own_fate),
            WALK_ENTRY_MAX_BYTES,
        );
        // A cut fate, or one the provider already punctuated, takes no second stop.
        let full_stop = if fate.ends_with(['.', '…']) {
            ""
        } else {
            "."
        };
        Some(format!(
            "`{served_by}` produced this picture — the account's image model `{}` {fate}{full_stop}",
            self.own_model,
        ))
    }
}

/// Default aspect ratio: supported by every model in the image-models catalog.
const DEFAULT_ASPECT_RATIO: &str = "9:16";

// ── Generation request: dedicated client, bounded retries, true causes ───

/// Total POST attempts (first attempt + 1 retry).
const IMAGE_GEN_MAX_ATTEMPTS: u32 = 2;

/// Retry only failures that surface within this window. Observed generations
/// complete in 92–99 s, so a failure inside this window is a prompt
/// transport/availability failure, not a cut-off generation — per OpenRouter's
/// all-or-nothing billing a generation that never completes is not billed,
/// making the retry safe. Never retry a near-full-timeout attempt: it may
/// have completed server-side (billed) with the response lost in transit.
const IMAGE_GEN_QUICK_FAILURE_MS: u64 = 60_000;

/// Backoff before the retry when the response carries no Retry-After header.
const IMAGE_GEN_BACKOFF_MS: u64 = 5_000;

/// Classification of one image-generation attempt failure, captured at the
/// HTTP boundary so the true cause survives into the surfaced error (a
/// reqwest error's top-level `Display` hides the source chain, e.g.
/// "operation timed out").
enum AttemptFailure {
    /// Client-side timeout. `connect` distinguishes a connection timeout (the
    /// request never reached the provider — safe to retry) from the full
    /// 10-minute request timeout (the generation may have completed server-side).
    Timeout { elapsed_ms: u64, connect: bool },
    /// Transport-level failure (connection refused/reset, no HTTP response).
    Transport { elapsed_ms: u64 },
    /// Non-2xx HTTP response (provider error).
    Http {
        status: u16,
        body: String,
        retry_after_ms: Option<u64>,
        elapsed_ms: u64,
    },
    /// 2xx response whose body could not be read — the generation may have
    /// completed and been billed server-side; never retried.
    BodyRead,
    /// 2xx body that did not parse as JSON.
    Parse { message: String },
}

impl AttemptFailure {
    /// Whether this failure may be retried: prompt, transient failures only.
    /// Body-read and parse failures are never retried (billable-ambiguous or
    /// non-transient).
    fn retryable(&self) -> bool {
        match self {
            Self::Timeout {
                elapsed_ms,
                connect,
            } => {
                // Only connect timeouts are safe to retry — the request never
                // reached the provider. The elapsed guard is defensive: the
                // 10s connect cap keeps genuine connect timeouts well inside
                // the quick-failure window, while a full request timeout is a
                // possible hang and never retried.
                *connect && *elapsed_ms < IMAGE_GEN_QUICK_FAILURE_MS
            }
            Self::Transport { elapsed_ms } => *elapsed_ms < IMAGE_GEN_QUICK_FAILURE_MS,
            Self::Http {
                status, elapsed_ms, ..
            } => {
                matches!(status, 429 | 502 | 503 | 524 | 529)
                    && *elapsed_ms < IMAGE_GEN_QUICK_FAILURE_MS
            }
            Self::BodyRead | Self::Parse { .. } => false,
        }
    }

    fn retry_after_ms(&self) -> Option<u64> {
        match self {
            Self::Http { retry_after_ms, .. } => *retry_after_ms,
            _ => None,
        }
    }

    /// Whether another configured model may be offered the request after this
    /// failure: the cause is on the side of the chosen model or the provider
    /// serving it — the model cannot make pictures, is gone, is unavailable or
    /// overloaded, will not accept the request or the key, has no quota left — or
    /// the request never reached the provider at all. Every HTTP refusal qualifies
    /// whatever it says, since the product does not judge a refusal from its
    /// wording; a transport failure qualifies along the same line the product
    /// already treats such an attempt as safely repeatable (inside the
    /// quick-failure window, i.e. nothing can have been generated). A failure whose
    /// outcome is unknown — a wait that ran out, an answer that could not be read —
    /// never does.
    fn walkable(&self) -> bool {
        match self {
            Self::Http { .. } => true,
            Self::Transport { elapsed_ms }
            | Self::Timeout {
                elapsed_ms,
                connect: true,
            } => *elapsed_ms < IMAGE_GEN_QUICK_FAILURE_MS,
            Self::Timeout { connect: false, .. } | Self::BodyRead | Self::Parse { .. } => false,
        }
    }
}

/// How one model's attempt ended without a picture.
enum AttemptOutcome {
    /// Walk on: another configured model may still serve the request.
    WalkOn(Vec<AttemptFailure>),
    /// The outcome is unknown — the request may have been produced and charged, so
    /// nothing may re-issue it.
    Unknown(Vec<AttemptFailure>),
    /// Global shutdown raced the attempt.
    Aborted,
}

impl AttemptOutcome {
    /// Classify one model's terminal failure trail (its last failure decides).
    fn of(failures: Vec<AttemptFailure>) -> Self {
        if failures.last().is_some_and(AttemptFailure::walkable) {
            Self::WalkOn(failures)
        } else {
            Self::Unknown(failures)
        }
    }
}

/// One model's POST — the walk's only outward step, and the seam through which a
/// test drives the walk with the answers the product receives instead of reaching a
/// provider.
#[async_trait]
trait ImageSender: Send + Sync {
    async fn post(
        &self,
        url: &str,
        body: &serde_json::Value,
        auth: &str,
    ) -> Result<serde_json::Value, AttemptFailure>;
}

/// The production sender: OpenRouter's dedicated Image API over the 10-minute
/// client.
struct HttpSender;

#[async_trait]
impl ImageSender for HttpSender {
    async fn post(
        &self,
        url: &str,
        body: &serde_json::Value,
        auth: &str,
    ) -> Result<serde_json::Value, AttemptFailure> {
        attempt_image_generation(url, body, auth).await
    }
}

/// POST one model's request with bounded retries.
///
/// Retry policy: auto-retry ONLY prompt failures — transport errors and HTTP
/// 429/502/503/524/529 — with backoff (Retry-After when present, else 5 s)
/// and at most one retry per model, the same single retry the attempt always had.
/// Every retryable arm is additionally gated by `IMAGE_GEN_QUICK_FAILURE_MS`: a
/// failure outside the quick-failure window means a long generation was in flight —
/// possibly billed, never re-submitted. Never retry: full-timeout hangs, a 4xx
/// outside the retryable statuses (402 = insufficient credits), body-read
/// failures after a 2xx (may have been billed), or 200-with-error bodies
/// (completed and billed).
async fn generate_image_with_retries(
    url: &str,
    body: &serde_json::Value,
    auth: &str,
    sender: &dyn ImageSender,
) -> Result<serde_json::Value, AttemptOutcome> {
    let mut failures: Vec<AttemptFailure> = Vec::new();
    for attempt in 1..=IMAGE_GEN_MAX_ATTEMPTS {
        match sender.post(url, body, auth).await {
            Ok(body) => return Ok(body),
            Err(failure) => {
                let retryable = failure.retryable();
                failures.push(failure);
                if !retryable || attempt == IMAGE_GEN_MAX_ATTEMPTS {
                    return Err(AttemptOutcome::of(failures));
                }
                let sleep_ms = failures
                    .last()
                    .and_then(AttemptFailure::retry_after_ms)
                    .map_or(IMAGE_GEN_BACKOFF_MS, |ms| {
                        ms.clamp(IMAGE_GEN_BACKOFF_MS, RETRY_AFTER_MAX_MS)
                    });
                if !crate::shutdown::sleep_or_shutdown(Duration::from_millis(sleep_ms)).await {
                    return Err(AttemptOutcome::Aborted);
                }
            }
        }
    }
    unreachable!("loop is bounded by IMAGE_GEN_MAX_ATTEMPTS")
}

/// One image-generation POST attempt with the dedicated 10-minute client.
/// Returns the parsed 2xx body, or a classified failure.
async fn attempt_image_generation(
    url: &str,
    body: &serde_json::Value,
    auth: &str,
) -> Result<serde_json::Value, AttemptFailure> {
    let started = Instant::now();
    let client = crate::util::http::image_gen_http_client();
    let response = match client
        .post(url)
        .header("Authorization", auth)
        .json(body)
        .send()
        .await
    {
        Ok(response) => response,
        Err(e) => {
            let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(0);
            return Err(if e.is_timeout() {
                AttemptFailure::Timeout {
                    elapsed_ms,
                    // A connect timeout never reached the provider (safe to
                    // retry); the full request timeout may have completed
                    // server-side. `is_connect()` covers DNS and TCP-level
                    // connect failures.
                    connect: e.is_connect(),
                }
            } else {
                tracing::debug!(error = %e, "Image generation transport error");
                AttemptFailure::Transport { elapsed_ms }
            });
        }
    };

    let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(0);
    if !response.status().is_success() {
        let status = response.status().as_u16();
        let retry_after_ms = retry_after_header(response.headers());
        let body = read_error_body(response, "Image generation").await;
        return Err(AttemptFailure::Http {
            status,
            body,
            retry_after_ms,
            elapsed_ms,
        });
    }

    let body_text = match response.text().await {
        Ok(text) => text,
        Err(e) => {
            tracing::warn!(error = %e, "Failed to read image generation response body");
            return Err(AttemptFailure::BodyRead);
        }
    };
    let parsed =
        crate::util::http::parse_json_response(&body_text, "Image generation").map_err(|e| {
            AttemptFailure::Parse {
                message: e.to_string(),
            }
        })?;
    Ok(parsed)
}

/// Build the user-visible terminal error from the failure trail, stating the
/// actual cause (timeout vs HTTP status vs transport vs billable-ambiguous).
/// `has_references` says whether the refused request carried input images, so
/// a refusal can carry the reference-count hint only where it could apply.
fn build_terminal_message(failures: &[AttemptFailure], has_references: bool) -> String {
    let last = failures.last().expect("at least one failure");
    let cause = failure_cause(last, has_references);
    let attempts = failures.len();
    let prefix = if attempts > 1 {
        format!("Image generation failed after {attempts} attempt(s): ")
    } else if matches!(last, AttemptFailure::Parse { .. }) {
        // A single-attempt parse failure keeps its fully-formed message
        // without an extra "Image generation failed: " prefix.
        String::new()
    } else {
        "Image generation failed: ".to_string()
    };
    format!("{prefix}{cause}")
}

/// One model's answer as the walk's failure names it: the cause of its terminal
/// attempt, with the attempt count when its own single retry ran.
fn describe_failure(failures: &[AttemptFailure], has_references: bool) -> String {
    let cause = failure_cause(
        failures.last().expect("at least one failure"),
        has_references,
    );
    if failures.len() > 1 {
        format!("{cause} ({} attempts)", failures.len())
    } else {
        cause
    }
}

/// The cause of one attempt failure — the provider's own explanation where it gave
/// one, never an invented one.
fn failure_cause(last: &AttemptFailure, has_references: bool) -> String {
    match last {
        // Parse failures already carry a fully-formed message (error context
        // plus a raw-body preview) — used verbatim.
        AttemptFailure::Parse { message } => message.clone(),
        AttemptFailure::Timeout {
            elapsed_ms,
            connect,
        } => {
            let secs = elapsed_ms / 1000;
            // Only two combinations are reachable given the client's fixed
            // timeouts (10s connect cap, 10-min total): a connect timeout
            // fires inside the connect cap, a request timeout at the full cap.
            if *connect {
                format!(
                    "connection timed out after {secs} s — the request never reached \
                     the provider"
                )
            } else {
                format!(
                    "request timed out after {secs} s (client-side 10-minute limit) — \
                     the provider may still complete the generation server-side; \
                     not auto-retried"
                )
            }
        }
        AttemptFailure::Transport { elapsed_ms } => {
            if *elapsed_ms >= IMAGE_GEN_QUICK_FAILURE_MS {
                format!(
                    "transport error after {} s — the connection was lost mid-flight; the \
                     generation may have completed and been billed server-side; verify \
                     before retrying",
                    elapsed_ms / 1000,
                )
            } else {
                "transport error — the request did not reach the provider or no response \
                 was received"
                    .to_string()
            }
        }
        AttemptFailure::Http { status, body, .. } => {
            describe_http_failure(*status, body, has_references)
        }
        AttemptFailure::BodyRead => {
            "received a 2xx response but the body could not be read — the generation may \
             have completed and been billed server-side; verify before retrying"
                .to_string()
        }
    }
}

/// Map an HTTP status to the typed provider error description, including the
/// provider's embedded error message when the body is parseable JSON.
///
/// The provider detail is extracted with a conservative cascade over the
/// envelope fields providers actually use (standard `error.message`, then the
/// nested-envelope fields — `metadata.raw`, `provider_error_code`, `code`,
/// `type` — see [`crate::util::extract_provider_error_detail`]), so the full
/// provider code/message reaches the model even when the detail is nested.
///
/// The refusal's cause is never invented: an HTTP 400 is reported as a
/// refusal, and the provider's own explanation stays the verdict. When the
/// request carried input images (`has_references`), the message additionally
/// carries a conditional hint that the provider may accept fewer references
/// than the catalog declares — the product does not parse the provider's
/// wording to decide whether the count was the cause.
fn describe_http_failure(status: u16, body: &str, has_references: bool) -> String {
    let typed = match status {
        400 => "provider refused the request (HTTP 400)".to_string(),
        402 => "insufficient credits — add credits to the provider account and retry (HTTP 402)"
            .to_string(),
        429 => "rate limit exceeded (HTTP 429)".to_string(),
        502 => "provider unavailable (HTTP 502)".to_string(),
        503 => "provider overloaded (HTTP 503)".to_string(),
        504 => "provider timeout (HTTP 504)".to_string(),
        524 | 529 => format!("edge timeout or overload (HTTP {status})"),
        _ => format!("HTTP {status}"),
    };
    let provider_msg = serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|v| crate::util::extract_provider_error_detail(&v));
    let mut described = match provider_msg {
        Some(msg) => format!("{typed}: {msg}"),
        None => typed,
    };
    if status == 400 && has_references {
        described.push_str(
            " If the refusal is about the number of input images, the provider may \
             accept fewer than the catalog declares — retry with fewer images.",
        );
    }
    described
}

/// Build the dedicated Image API request body from tool args and (when
/// available) the model's declared capabilities. `info` is `Some` exactly
/// when the catalog is available and the model passed the capability check;
/// `None` means the catalog is unavailable (fail-open) — only explicitly
/// user-provided parameters that need no capability check are sent, and the
/// requested size was already dropped before this point (see [`own_step`]).
///
/// `size` is the pre-validated [`resolve_size`] result: the request-body key
/// to send it under, and the value.
///
/// Reference-count validation is the caller's ([`eligibility`], through
/// [`validate_reference_count`]); this builder trusts the caller's count.
fn build_request_body(
    model: &str,
    prompt: &str,
    aspect_ratio: Option<&str>,
    size: Option<(&'static str, String)>,
    references: &[crate::util::ReferenceImage],
    info: Option<&ImageModelInfo>,
) -> serde_json::Value {
    let mut body = json!({
        "model": model,
        "prompt": prompt,
    });

    if let Some(info) = info {
        if info.declares("aspect_ratio")
            && let Some(ratio) = aspect_ratio
        {
            // "auto" is only valid where the model declares it; otherwise
            // fall back to the safe default.
            let ratio = if ratio == "auto" && !info.enum_contains("aspect_ratio", "auto") {
                DEFAULT_ASPECT_RATIO
            } else {
                ratio
            };
            body["aspect_ratio"] = json!(ratio);
        }

        if !references.is_empty() {
            body[super::INPUT_REFERENCES_KEY] = super::reference_json(references);
        }
    } else {
        // Fail-open: send only what the user explicitly provided. A requested
        // size never gets here — with no catalog data, guessing the model's size
        // parameter is not an option (see [`own_step`], which drops it).
        if let Some(ratio) = aspect_ratio {
            body["aspect_ratio"] = json!(ratio);
        }
        if !references.is_empty() {
            body[super::INPUT_REFERENCES_KEY] = super::reference_json(references);
        }
    }

    // The size is pre-validated by `resolve_size`, so apply it unconditionally.
    if let Some((key, value)) = size {
        body[key] = json!(value);
    }

    body
}

/// Validate the reference count against the catalog's `input_references`
/// range, clamped to the provider's documented ceiling. Runs in [`eligibility`]
/// before any model's body is built — for the account's own model before any
/// reference file is loaded, for a candidate after they are — and is worded for
/// `voice`, exactly as [`unsupported_size`] is.
fn validate_reference_count(
    model: &str,
    info: &ImageModelInfo,
    count: usize,
    voice: RefusalVoice,
) -> Result<(), String> {
    let Some(max) = info
        .range_max("input_references")
        .map(super::effective_reference_max)
    else {
        // The model declares no reference input at all.
        return Err(match voice {
            RefusalVoice::OwnModel => format!(
                "Model `{model}` does not support reference images. \
                 Retry without the `images` parameter."
            ),
            RefusalVoice::Candidate => "does not support reference images".to_string(),
        });
    };
    #[expect(clippy::cast_possible_wrap)]
    if count as i64 <= max {
        return Ok(());
    }
    Err(match voice {
        RefusalVoice::OwnModel => format!(
            "Model `{model}` supports at most {max} reference image(s), got {count}. \
             Retry with fewer images.",
        ),
        RefusalVoice::Candidate => {
            format!("supports at most {max} reference image(s), got {count}")
        }
    })
}

/// Resolve the requested `size` against the model's declared capabilities,
/// returning the request-body key and the normalised value the request must
/// carry. Called for every model the walk builds a body for — the account's own
/// and each candidate alike.
///
/// The model — not MahBot — decides which sizes exist, so a value it cannot
/// express is refused with the values it does support (or an explicit "no
/// adjustable size") instead of being dropped silently or bounced back by the
/// provider as an opaque error. A tier (`2K`) is case-insensitive and goes
/// through `resolution` when the model declares it, through `size` otherwise
/// (OpenRouter's `size` takes a tier or explicit pixels); pixels
/// (`1024x1024`) always go through `size`. A request from a model whose
/// declarations are unknown never reaches this (see [`eligibility`]).
fn resolve_size(
    model: &str,
    size: Option<&str>,
    info: &ImageModelInfo,
    voice: RefusalVoice,
) -> anyhow::Result<Option<(&'static str, String)>> {
    let Some(size) = size.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(None);
    };

    let (key, value) = if size.contains(['x', 'X', '×']) {
        ("size", size.to_string())
    } else if info.declares("resolution") {
        ("resolution", size.to_uppercase())
    } else {
        ("size", size.to_uppercase())
    };

    if !info.declares(key) {
        return Err(unsupported_size(model, size, info, voice));
    }
    // A declared enum is the model's own list of what it takes; another
    // constraint shape declares the parameter without enumerating values.
    if let Some(values) = info.enum_values(key)
        && !values.iter().any(|v| v.eq_ignore_ascii_case(&value))
    {
        return Err(unsupported_size(model, size, info, voice));
    }

    Ok(Some((key, value)))
}

/// The refusal for a size the model cannot express: the values it accepts, or —
/// when the model declares no size parameter at all — a statement that it has no
/// adjustable size (an empty value list would be meaningless), in the voice of
/// `voice`.
fn unsupported_size(
    model: &str,
    size: &str,
    info: &ImageModelInfo,
    voice: RefusalVoice,
) -> anyhow::Error {
    let supported: Vec<&str> = ["resolution", "size"]
        .iter()
        .filter_map(|param| info.enum_values(param))
        .flatten()
        .map(String::as_str)
        .collect();
    // A candidate's line names the model already, so the fact alone is what it adds.
    match (voice, supported.is_empty()) {
        (RefusalVoice::OwnModel, true) => anyhow::anyhow!(
            "usage: model `{model}` has no adjustable image size — \
             hint: omit the `size` argument"
        ),
        (RefusalVoice::OwnModel, false) => anyhow::anyhow!(
            "usage: model `{model}` does not support size `{size}` — \
             hint: pass a size it supports: {}",
            supported.join(", ")
        ),
        (RefusalVoice::Candidate, true) => anyhow::anyhow!("has no adjustable image size"),
        (RefusalVoice::Candidate, false) => anyhow::anyhow!(
            "does not support size `{size}` — \
             it accepts: {}",
            supported.join(", ")
        ),
    }
}

/// All canonical aspect ratios supported by OpenRouter, mapped to their float
/// value (width / height). Used to find the closest match when auto-detecting
/// from a reference image.
static CANONICAL_ASPECT_RATIOS: &[(&str, f64)] = &[
    ("1:1", 1.0),
    ("16:9", 16.0 / 9.0),
    ("9:16", 9.0 / 16.0),
    ("4:3", 4.0 / 3.0),
    ("3:4", 3.0 / 4.0),
    ("3:2", 3.0 / 2.0),
    ("2:3", 2.0 / 3.0),
    ("4:5", 4.0 / 5.0),
    ("5:4", 5.0 / 4.0),
    ("1:2", 1.0 / 2.0),
    ("2:1", 2.0 / 1.0),
    ("1:4", 1.0 / 4.0),
    ("4:1", 4.0 / 1.0),
    ("21:9", 21.0 / 9.0),
    ("9:21", 9.0 / 21.0),
    ("1:8", 1.0 / 8.0),
    ("8:1", 8.0 / 1.0),
    ("9:19.5", 9.0 / 19.5),
    ("19.5:9", 19.5 / 9.0),
    ("9:20", 9.0 / 20.0),
    ("20:9", 20.0 / 9.0),
];

/// Resolve the effective aspect ratio: the user-provided value, the closest
/// canonical ratio detected from the first reference image, or the
/// [`DEFAULT_ASPECT_RATIO`] default.
fn resolve_aspect_ratio(aspect_ratio: Option<&str>, reference_paths: &[PathBuf]) -> String {
    match aspect_ratio {
        Some(ar) => ar.to_string(),
        None if !reference_paths.is_empty() => {
            let first = &reference_paths[0];
            if let Some(ratio) = detect_aspect_ratio_from_image(first) {
                tracing::debug!(
                    "Auto-detected aspect ratio {ratio} from reference image `{}`",
                    first.display(),
                );
                ratio.to_string()
            } else {
                tracing::debug!(
                    "Could not detect aspect ratio from reference image `{}`, falling back to {DEFAULT_ASPECT_RATIO}",
                    first.display(),
                );
                DEFAULT_ASPECT_RATIO.to_string()
            }
        }
        None => DEFAULT_ASPECT_RATIO.to_string(),
    }
}

/// Detect the closest canonical aspect ratio from an image file.
///
/// Reads only the file header (no full decode) via the `image` crate's
/// magic-byte dimension detection. Returns `None` if the file cannot be
/// read, is an unsupported format (only PNG/JPEG/WebP are compiled in), or
/// has zero dimensions.
fn detect_aspect_ratio_from_image(path: &Path) -> Option<&'static str> {
    let (width, height) = image::image_dimensions(path).ok()?;
    find_closest_aspect_ratio(width, height)
}

/// Find the closest canonical aspect ratio string for the given dimensions.
///
/// Returns `None` when either dimension is zero.
fn find_closest_aspect_ratio(width: u32, height: u32) -> Option<&'static str> {
    // Guard against zero dimensions (would produce ∞ or panic at division)
    if width == 0 || height == 0 {
        return None;
    }

    let ratio = f64::from(width) / f64::from(height);

    // Find the closest canonical ratio via `min_by`. When two ratios are
    // equally close, the first in declaration order wins (a practical
    // impossibility with the given spacing, but `unwrap_or(Equal)` gives
    // the correct tie-break).
    CANONICAL_ASPECT_RATIOS
        .iter()
        .min_by(|a, b| {
            let da = (ratio - a.1).abs();
            let db = (ratio - b.1).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(name, _)| *name)
}

/// Why a 2xx image response could not be turned into an image. Both cases
/// mean the attempt completed and was billed server-side — never retried.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ExtractImageError {
    /// The body carries a provider `error` object (a billed 200-with-error).
    Provider { err_type: String, message: String },
    /// The body has no usable `data[].b64_json` payload.
    NoImageData,
}

impl std::fmt::Display for ExtractImageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Provider { err_type, message } => write!(f, "{err_type}: {message}"),
            Self::NoImageData => {
                write!(f, "response did not contain image data (data[].b64_json)")
            }
        }
    }
}

/// Extract the first non-empty `data[].b64_json` payload and its optional
/// `media_type` from a dedicated Image API response. `b64_json` is raw
/// base64 (not a data URI); `media_type` may be absent (PNG default).
///
/// Returns `Err` when the body carries a provider `error` object (surfacing
/// the embedded type/message — a billed 200-with-error response) or lacks
/// image data entirely; the variant drives the telemetry class.
fn extract_response_image(
    body: &serde_json::Value,
) -> Result<(String, Option<String>), ExtractImageError> {
    if let Some(err) = body.get("error") {
        let err_type = err
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("provider error");
        let message = err
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("no details");
        return Err(ExtractImageError::Provider {
            err_type: err_type.to_string(),
            message: message.to_string(),
        });
    }
    let Some(data) = body.get("data").and_then(|v| v.as_array()) else {
        return Err(ExtractImageError::NoImageData);
    };
    for entry in data {
        if let Some(b64) = entry.get("b64_json").and_then(|v| v.as_str())
            && !b64.is_empty()
        {
            let media_type = entry
                .get("media_type")
                .and_then(|v| v.as_str())
                .map(String::from);
            return Ok((b64.to_string(), media_type));
        }
    }
    Err(ExtractImageError::NoImageData)
}

/// Map a response media type to a file extension; PNG when absent or unknown.
#[must_use]
fn extension_for_media_type(media_type: Option<&str>) -> &'static str {
    match media_type {
        Some("image/jpeg") => "jpg",
        Some("image/webp") => "webp",
        Some("image/svg+xml") => "svg",
        _ => "png",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::media_catalog::image::{ImageCatalog, parse_catalog};
    use crate::util::UnwrapPoison;
    use crate::util::test::assert_contains_not_contains;
    use RefusalVoice::OwnModel;
    use std::collections::{HashMap, VecDeque};
    use std::sync::{Arc, Mutex};

    /// Fixture catalog covering a hybrid image model (resolution + reference
    /// caps), a recraft-like model (aspect ratio only, declares "auto"), a
    /// model with explicit-pixel `size`, a model declaring more reference
    /// images than the provider accepts, and a text-only model.
    fn fixture_catalog() -> ImageCatalog {
        parse_catalog(&json!({
            "data": [
                {
                    "id": "qwen/qwen-image-3-pro",
                    "architecture": { "output_modalities": ["image"] },
                    "supported_parameters": {
                        "resolution": { "type": "enum", "values": ["1K", "2K"] },
                        "aspect_ratio": { "type": "enum", "values": ["1:1", "9:16"] },
                        "input_references": { "type": "range", "min": 0, "max": 4 }
                    }
                },
                {
                    "id": "recraft/recraft-v4.1",
                    "architecture": { "output_modalities": ["image"] },
                    "supported_parameters": {
                        "aspect_ratio": { "type": "enum", "values": ["1:1", "9:16", "auto"] },
                        "input_references": { "type": "range", "min": 0, "max": 1 }
                    }
                },
                {
                    "id": "acme/pixels",
                    "architecture": { "output_modalities": ["image"] },
                    "supported_parameters": {
                        "size": { "type": "enum", "values": ["1024x1024", "2048x2048"] }
                    }
                },
                {
                    "id": "huge/refs",
                    "architecture": { "output_modalities": ["image"] },
                    "supported_parameters": {
                        "input_references": { "type": "range", "min": 0, "max": 20 }
                    }
                },
                {
                    "id": "text-only/model",
                    "architecture": { "output_modalities": ["text"] },
                    "supported_parameters": {}
                }
            ]
        }))
        .expect("valid fixture")
    }

    /// One tiny 1×1 reference PNG on disk and the `ReferenceImage`s loaded from it —
    /// what a walk's request carries, with the path the aspect ratio is resolved
    /// from.
    struct Refs {
        _dir: tempfile::TempDir,
        paths: Vec<PathBuf>,
        loaded: Vec<crate::util::ReferenceImage>,
    }

    impl Refs {
        /// No reference at all.
        fn none() -> Self {
            Self {
                _dir: tempfile::TempDir::new().unwrap(),
                paths: Vec::new(),
                loaded: Vec::new(),
            }
        }
    }

    /// Build `n` real validated reference images from one tiny PNG file.
    async fn refs(n: usize) -> Refs {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("ref.png");
        std::fs::write(&path, crate::util::test::noisy_png(1, 1)).unwrap();
        let mut loaded = Vec::with_capacity(n);
        for _ in 0..n {
            loaded.push(
                crate::util::load_reference_image(&path, super::super::MAX_REFERENCE_IMAGE_BYTES)
                    .await
                    .unwrap(),
            );
        }
        Refs {
            _dir: dir,
            paths: vec![path; n],
            loaded,
        }
    }

    /// One [`build_request_body`] scenario: the model, its catalog declarations
    /// (if the catalog is reachable), the arguments and the body fields they must
    /// produce.
    struct RequestBodyCase {
        name: &'static str,
        model: &'static str,
        catalog_model: Option<&'static str>,
        prompt: &'static str,
        aspect: Option<&'static str>,
        size_arg: Option<&'static str>,
        resolved_size: Option<(&'static str, &'static str)>,
        refs: usize,
        expected: &'static [(&'static str, &'static str)],
        absent: &'static [&'static str],
    }

    #[tokio::test]
    async fn build_request_body_cases() {
        for case in [
            RequestBodyCase {
                name: "catalog declarations drive the body",
                model: "qwen/qwen-image-3-pro",
                catalog_model: Some("qwen/qwen-image-3-pro"),
                prompt: "a cat",
                aspect: Some("9:16"),
                size_arg: Some("2k"),
                // size "2k" → resolution "2K" (catalog case).
                resolved_size: Some(("resolution", "2K")),
                refs: 2,
                expected: &[
                    ("model", "qwen/qwen-image-3-pro"),
                    ("prompt", "a cat"),
                    ("resolution", "2K"),
                    ("aspect_ratio", "9:16"),
                ],
                absent: &["size", "n"],
            },
            RequestBodyCase {
                name: "fail-open sends only user-provided parameters",
                model: "any/model",
                catalog_model: None,
                prompt: "p",
                aspect: Some("16:9"),
                size_arg: None,
                // With no catalog data no model capability is known.
                resolved_size: None,
                refs: 1,
                expected: &[
                    ("model", "any/model"),
                    ("prompt", "p"),
                    ("aspect_ratio", "16:9"),
                ],
                absent: &["resolution", "size"],
            },
            RequestBodyCase {
                name: "fail-open without an aspect ratio sends neither it nor references",
                model: "any/model",
                catalog_model: None,
                prompt: "p",
                aspect: None,
                size_arg: None,
                resolved_size: None,
                refs: 0,
                expected: &[("model", "any/model"), ("prompt", "p")],
                absent: &["aspect_ratio", "input_references", "resolution", "size"],
            },
        ] {
            let catalog = fixture_catalog();
            let info = case.catalog_model.map(|model| {
                catalog
                    .find(model)
                    .unwrap_or_else(|| panic!("[{}] missing fixture {model}", case.name))
            });
            let size = info.map(|info| {
                let size = resolve_size(case.model, case.size_arg, info, OwnModel)
                    .expect("a resolvable size");
                assert_eq!(
                    size.as_ref().map(|(key, value)| (*key, value.as_str())),
                    case.resolved_size,
                    "[{}] resolved size",
                    case.name
                );
                size
            });
            let references = refs(case.refs).await.loaded;
            let body = build_request_body(
                case.model,
                case.prompt,
                case.aspect,
                size.flatten(),
                &references,
                info,
            );
            for (field, value) in case.expected {
                assert_eq!(body[field], *value, "[{}] field {field}", case.name);
            }
            for field in case.absent {
                assert!(
                    body.get(field).is_none(),
                    "[{}] field {field} must not be sent: {body}",
                    case.name
                );
            }
            if case.refs > 0 {
                assert_eq!(
                    body["input_references"].as_array().unwrap().len(),
                    case.refs,
                    "[{}] reference count",
                    case.name
                );
                assert!(
                    body["input_references"][0]["image_url"]["url"]
                        .as_str()
                        .unwrap()
                        .starts_with("data:image/png;base64,"),
                    "[{}] reference should be a data URI",
                    case.name
                );
            }
        }
    }

    #[tokio::test]
    async fn test_build_request_body_omits_undeclared_and_validates_caps() {
        let catalog = fixture_catalog();
        let recraft = catalog.find("recraft/recraft-v4.1");
        let qwen = catalog.find("qwen/qwen-image-3-pro");
        let acme = catalog.find("acme/pixels");

        // recraft declares no size parameter at all → refused, not dropped.
        let err = resolve_size(
            "recraft/recraft-v4.1",
            Some("2K"),
            recraft.expect("fixture"),
            OwnModel,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("has no adjustable image size"),
            "unexpected error: {err}"
        );

        // qwen declares resolution ["1K", "2K"] → an unsupported tier names
        // the values the model does support.
        let err = resolve_size(
            "qwen/qwen-image-3-pro",
            Some("512"),
            qwen.expect("fixture"),
            OwnModel,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("1K, 2K"),
            "unexpected error: {err}"
        );

        // '1024X1024' (uppercase separator) is pixels; qwen declares no `size`
        // → refused with its resolution values, never sent as a bogus resolution.
        let err = resolve_size(
            "qwen/qwen-image-3-pro",
            Some("1024X1024"),
            qwen.expect("fixture"),
            OwnModel,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("1K, 2K"),
            "unexpected error: {err}"
        );

        // A tier is case-insensitive: "1k" matches the declared "1K".
        assert_eq!(
            resolve_size(
                "qwen/qwen-image-3-pro",
                Some("1k"),
                qwen.expect("fixture"),
                OwnModel
            )
            .unwrap(),
            Some(("resolution", "1K".to_string()))
        );

        // Explicit-pixel model: a supported pixel value is sent verbatim under
        // `size`; an unsupported one is refused with the supported list.
        let size = resolve_size(
            "acme/pixels",
            Some("1024x1024"),
            acme.expect("fixture"),
            OwnModel,
        )
        .unwrap();
        assert_eq!(size, Some(("size", "1024x1024".to_string())));
        let body = build_request_body("acme/pixels", "p", None, size, &[], acme);
        assert_eq!(body["size"], "1024x1024");

        let err = resolve_size(
            "acme/pixels",
            Some("512x512"),
            acme.expect("fixture"),
            OwnModel,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("1024x1024, 2048x2048"),
            "unexpected error: {err}"
        );

        // "auto" not declared by qwen → falls back to the 9:16 default.
        let body = build_request_body("qwen/qwen-image-3-pro", "p", Some("auto"), None, &[], qwen);
        assert_eq!(body["aspect_ratio"], "9:16");

        // recraft declares "auto" → passed through.
        let body = build_request_body(
            "recraft/recraft-v4.1",
            "p",
            Some("auto"),
            None,
            &[],
            recraft,
        );
        assert_eq!(body["aspect_ratio"], "auto");

        // Reference overflow → error (count validated pre-flight, not
        // truncated by the body builder).
        let err = validate_reference_count("qwen/qwen-image-3-pro", qwen.unwrap(), 5, OwnModel)
            .unwrap_err();
        assert!(err.contains("at most 4 reference image(s)"));

        // Model without input_references → error when images provided.
        let text_only = catalog.find("text-only/model").unwrap();
        let err = validate_reference_count("text-only/model", text_only, 1, OwnModel).unwrap_err();
        assert!(err.contains("does not support reference images"));

        // A candidate the walk passed over gets the fact alone: no model name the line
        // already carries, no advice about the account's own pick.
        let err = validate_reference_count(
            "qwen/qwen-image-3-pro",
            qwen.unwrap(),
            5,
            RefusalVoice::Candidate,
        )
        .unwrap_err();
        assert_eq!(err, "supports at most 4 reference image(s), got 5");

        // A catalog declaring more than the provider accepts is clamped to the
        // documented ceiling: 16 passes, 17 is refused with that same maximum.
        let huge = catalog.find("huge/refs").unwrap();
        validate_reference_count("huge/refs", huge, 16, OwnModel).expect("16 references accepted");
        let err = validate_reference_count("huge/refs", huge, 17, OwnModel).unwrap_err();
        assert!(err.contains("at most 16 reference image(s)"));
    }

    // ── find_closest_aspect_ratio tests ──────────────────────────────

    #[test]
    fn find_closest_aspect_ratio_cases() {
        struct Case {
            name: &'static str,
            width: u32,
            height: u32,
            expected: Option<&'static str>,
        }

        // Every canonical ratio should round-trip exactly — generated from
        // CANONICAL_ASPECT_RATIOS so future ratios are auto-covered (and
        // `ratio_tuple_from_f64` stays exercised).
        let mut cases: Vec<Case> = CANONICAL_ASPECT_RATIOS
            .iter()
            .map(|&(ratio_str, ratio_val)| {
                let (w, h) = ratio_tuple_from_f64(ratio_val);
                Case {
                    name: ratio_str,
                    width: w,
                    height: h,
                    expected: Some(ratio_str),
                }
            })
            .collect();

        cases.extend([
            // Between-candidates: closest canonical, not an exact match.
            Case {
                name: "between_3x2",
                width: 1400,
                height: 900,
                expected: Some("3:2"),
            },
            Case {
                name: "between_16x9",
                width: 1700,
                height: 900,
                expected: Some("16:9"),
            },
            Case {
                name: "exact_5x4",
                width: 5,
                height: 4,
                expected: Some("5:4"),
            },
            Case {
                name: "between_4x5",
                width: 17,
                height: 20,
                expected: Some("4:5"),
            },
            // Zero dimensions → None (no meaningful ratio).
            Case {
                name: "zero_width",
                width: 0,
                height: 100,
                expected: None,
            },
            Case {
                name: "zero_height",
                width: 100,
                height: 0,
                expected: None,
            },
            Case {
                name: "zero_both",
                width: 0,
                height: 0,
                expected: None,
            },
        ]);

        for case in &cases {
            assert_eq!(
                find_closest_aspect_ratio(case.width, case.height),
                case.expected,
                "case: {name} (w={width}, h={height})",
                name = case.name,
                width = case.width,
                height = case.height,
            );
        }
    }

    /// Helper: convert a f64 ratio into integer width/height that produce
    /// the same ratio (within rounding). Used to construct test inputs.
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn ratio_tuple_from_f64(ratio: f64) -> (u32, u32) {
        // Scale to avoid integer division rounding errors:
        // multiply by a large power of 10 then reduce.
        let scale = 10_000_000.0;
        let w = (ratio * scale).round() as u32;
        let h = scale as u32;
        (w, h)
    }

    // ── detect_aspect_ratio_from_image integration tests ──────────────

    /// A minimal valid 2×1 PNG (2:1 aspect ratio), base64-encoded.
    /// Generated with: python3 -c "..."
    const MINI_2X1_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAIAAAB7QOjdAAAAC0lEQVR4nGNgAAMAAAcAAbKGrPQAAAAASUVORK5CYII=";

    /// A minimal valid 16×9 PNG (16:9 aspect ratio), base64-encoded.
    const MINI_16X9_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAJCAIAAAC0SDtlAAAADklEQVR4nGNgGAVDEgAAAbkAAftY4pIAAAAASUVORK5CYII=";

    /// One [`detect_aspect_ratio_from_image`] scenario: the png written to disk
    /// (if any) and the ratio it must yield.
    struct AspectRatioCase {
        name: &'static str,
        png_b64: Option<&'static str>,
        file: &'static str,
        expected: Option<&'static str>,
    }

    #[test]
    fn detect_aspect_ratio_cases() {
        use base64::{Engine as _, engine::general_purpose::STANDARD};

        for case in [
            AspectRatioCase {
                name: "a 2:1 png",
                png_b64: Some(MINI_2X1_PNG_B64),
                file: "test.png",
                expected: Some("2:1"),
            },
            AspectRatioCase {
                name: "a 16:9 png",
                png_b64: Some(MINI_16X9_PNG_B64),
                file: "wide.png",
                expected: Some("16:9"),
            },
            AspectRatioCase {
                name: "a missing file",
                png_b64: None,
                file: "missing.png",
                expected: None,
            },
        ] {
            let dir = tempfile::tempdir().expect("tempdir");
            let path = dir.path().join(case.file);
            if let Some(png) = case.png_b64 {
                let bytes = STANDARD.decode(png).expect("valid base64");
                std::fs::write(&path, &bytes).expect("write");
            }
            assert_eq!(
                detect_aspect_ratio_from_image(&path),
                case.expected,
                "[{}]",
                case.name
            );
        }
    }

    // ── Retry policy and true-cause reporting tests ────────────────────

    #[test]
    fn test_attempt_failure_retryable_decisions() {
        // Quick transport/timeout failures are retried (not billed per
        // OpenRouter all-or-nothing billing).
        assert!(AttemptFailure::Transport { elapsed_ms: 5_000 }.retryable());
        assert!(
            AttemptFailure::Timeout {
                elapsed_ms: 5_000,
                connect: true
            }
            .retryable()
        );
        // A near-full-timeout attempt is a hang — never retried.
        assert!(
            !AttemptFailure::Timeout {
                elapsed_ms: 600_000,
                connect: false
            }
            .retryable()
        );
        // A non-connect timeout is treated as a possible hang even when quick
        // (defensive; request timeouts only fire at the full cap in practice).
        assert!(
            !AttemptFailure::Timeout {
                elapsed_ms: 5_000,
                connect: false
            }
            .retryable()
        );

        // Transient provider statuses are retried; other statuses are not.
        for status in [429, 502, 503, 524, 529] {
            let f = AttemptFailure::Http {
                status,
                body: String::new(),
                retry_after_ms: None,
                elapsed_ms: 10_000,
            };
            assert!(f.retryable(), "status {status} should be retryable");
        }
        for status in [400, 401, 402, 404, 413, 500, 504] {
            let f = AttemptFailure::Http {
                status,
                body: String::new(),
                retry_after_ms: None,
                elapsed_ms: 10_000,
            };
            assert!(!f.retryable(), "status {status} should not be retryable");
        }

        // Billed-ambiguous and non-transient failures are never retried.
        assert!(!AttemptFailure::BodyRead.retryable());
        assert!(
            !AttemptFailure::Parse {
                message: "x".into()
            }
            .retryable()
        );
    }

    #[test]
    fn test_build_terminal_message_states_true_cause() {
        // Full-cap request timeout: explicitly not auto-retried.
        let msg = build_terminal_message(
            &[AttemptFailure::Timeout {
                elapsed_ms: 600_000,
                connect: false,
            }],
            false,
        );
        assert!(msg.contains("timed out after 600 s"));
        assert!(msg.contains("not auto-retried"));

        // Retried quick failures: the attempt-count prefix carries the retry
        // context — the cause itself does not claim a retry.
        let msg = build_terminal_message(
            &[
                AttemptFailure::Timeout {
                    elapsed_ms: 30_000,
                    connect: true,
                },
                AttemptFailure::Timeout {
                    elapsed_ms: 45_000,
                    connect: true,
                },
            ],
            false,
        );
        assert!(msg.contains("failed after 2 attempt(s)"));
        assert!(msg.contains("connection timed out"));

        // A connect timeout is honest about never reaching the provider, and
        // never claims a retry it did not make.
        let msg = build_terminal_message(
            &[AttemptFailure::Timeout {
                elapsed_ms: 10_000,
                connect: true,
            }],
            false,
        );
        assert!(msg.contains("never reached the provider"));
        assert!(!msg.contains("retried"));

        // HTTP cause with provider message.
        let msg = build_terminal_message(
            &[AttemptFailure::Http {
                status: 503,
                body: r#"{"error":{"message":"down"}}"#.into(),
                retry_after_ms: None,
                elapsed_ms: 1_000,
            }],
            false,
        );
        assert!(msg.contains("provider overloaded (HTTP 503)"));

        // A refused request that carried input images carries the hint; the
        // same request without them does not.
        let http_400 = AttemptFailure::Http {
            status: 400,
            body: r#"{"error":{"message":"too many images"}}"#.into(),
            retry_after_ms: None,
            elapsed_ms: 1_000,
        };
        let msg = build_terminal_message(&[http_400], true);
        assert!(msg.contains("retry with fewer images"));
        assert!(msg.contains("too many images"));

        // A long mid-flight transport failure warns about possible billing.
        let msg = build_terminal_message(
            &[AttemptFailure::Transport {
                elapsed_ms: 300_000,
            }],
            false,
        );
        assert!(msg.contains("may have completed and been billed"));

        // A retry followed by a parse failure keeps the attempt context.
        let msg = build_terminal_message(
            &[
                AttemptFailure::Transport { elapsed_ms: 5_000 },
                AttemptFailure::Parse {
                    message: "Image generation response parse error: bad".into(),
                },
            ],
            false,
        );
        assert!(msg.contains("failed after 2 attempt(s)"));
        assert!(msg.contains("parse error"));
    }

    // ── ImageWalk fallback tests ─────────────────────────────────────────

    /// Scripted answers per model: the walk is driven by exactly the answers the
    /// product would receive, never by a provider.
    struct Scripted {
        answers: Mutex<HashMap<String, VecDeque<Result<serde_json::Value, AttemptFailure>>>>,
        sent: Mutex<Vec<(String, serde_json::Value)>>,
    }

    #[async_trait]
    impl ImageSender for Scripted {
        async fn post(
            &self,
            url: &str,
            body: &serde_json::Value,
            _auth: &str,
        ) -> Result<serde_json::Value, AttemptFailure> {
            let model = body["model"]
                .as_str()
                .expect("every request body names its model")
                .to_string();
            self.sent
                .lock()
                .unwrap_poison()
                .push((model.clone(), body.clone()));
            let mut answers = self.answers.lock().unwrap_poison();
            let queue = answers
                .get_mut(&model)
                .unwrap_or_else(|| panic!("unscripted image POST to `{model}` ({url})"));
            queue
                .pop_front()
                .unwrap_or_else(|| panic!("scripted answers for `{model}` exhausted ({url})"))
        }
    }

    impl Scripted {
        fn new() -> Self {
            Self {
                answers: Mutex::new(HashMap::new()),
                sent: Mutex::new(Vec::new()),
            }
        }

        fn answer(&self, model: &str, answer: Result<serde_json::Value, AttemptFailure>) {
            self.answers
                .lock()
                .unwrap_poison()
                .entry(model.to_string())
                .or_default()
                .push_back(answer);
        }

        /// The models POSTed to, in POST order.
        fn models_sent(&self) -> Vec<String> {
            self.sent
                .lock()
                .unwrap_poison()
                .iter()
                .map(|(model, _)| model.clone())
                .collect()
        }

        /// The body of the last POST made to `model`.
        fn body_for(&self, model: &str) -> serde_json::Value {
            self.sent
                .lock()
                .unwrap_poison()
                .iter()
                .rev()
                .find(|(sent_to, _)| sent_to.as_str() == model)
                .map_or_else(
                    || panic!("no POST was made to `{model}`"),
                    |(_, body)| body.clone(),
                )
        }
    }

    /// An HTTP refusal that came back in 1 s.
    fn http(status: u16, body: &str) -> AttemptFailure {
        AttemptFailure::Http {
            status,
            body: body.to_string(),
            retry_after_ms: None,
            elapsed_ms: 1_000,
        }
    }

    /// A 2xx body carrying one base64 picture.
    fn served(b64: &str) -> serde_json::Value {
        json!({"data": [{"b64_json": b64}]})
    }

    /// A picker list from bare model names.
    fn picker(models: &[&str]) -> Vec<String> {
        models.iter().map(|model| (*model).to_string()).collect()
    }

    /// A walk whose trail is already every visited model's refusal, without
    /// sending anything: what the failure text is built from.
    fn refused_walk<'a>(
        sender: &'a Scripted,
        models: usize,
        message: &str,
        stopped: Option<String>,
    ) -> ImageWalk<'a> {
        let refusal = format!(r#"{{"error":{{"message":"{message}"}}}}"#);
        ImageWalk {
            request: RequestMaterial {
                prompt: "a cat",
                aspect_ratio: None,
                size_arg: None,
                references: &mut [],
            },
            own_model: "vendor/model-0",
            url: "https://example.invalid/images",
            auth: "Bearer test",
            sender,
            trail: (0..models)
                .map(|index| {
                    (
                        format!("vendor/model-{index}"),
                        Visited::Attempted(vec![http(400, &refusal)]),
                    )
                })
                .collect(),
            stopped,
        }
    }

    /// What the model actually receives for a failure: the wrapper the caller raises
    /// the fault text with, then the failure envelope — with a long multi-byte request
    /// preview in it — as `Agent::execute_tool`'s error arm composes them.
    fn delivered(failure: &str) -> String {
        crate::tools::format_tool_failure_feedback(
            "image_gen",
            &json!({ "prompt": "生成".repeat(600) }),
            &crate::tools::tool_failure_reason("image_gen", failure),
        )
    }

    fn seed_fixture() {
        crate::tools::media_catalog::image::seed_cache(Some(Arc::new(fixture_catalog())));
    }

    /// Seed the process-global catalog with the fixture, arming the
    /// membership-miss refresh cooldown on the way.
    ///
    /// Resolving a model the snapshot does not list forces one catalog refresh;
    /// with no provider key configured in tests that refresh fails and records a
    /// negative cache entry — wiping the snapshot. Priming the cooldown once (its
    /// side effect undone by the re-seed right after) makes every later
    /// membership miss re-check the seeded snapshot, as a reachable catalog does
    /// in production.
    async fn seed_fixture_catalog() {
        seed_fixture();
        let _ = resolve_image_model(crate::config::DEFAULT_PROVIDER_ENDPOINT, "probe/absent").await;
        seed_fixture();
    }

    /// Restore the shared catalog fixture at the end of a test that seeds a negative
    /// cache into it, however that test ends: a poisoned global cache would fail every
    /// later test sharing the `active_models` key.
    struct CatalogFixture;

    impl Drop for CatalogFixture {
        fn drop(&mut self) {
            seed_fixture();
        }
    }

    /// Restore the global config at the end of a test that changes it, however that test
    /// ends: a panic between the change and the restore would otherwise leave every later
    /// test in this binary running on it.
    struct ConfigFixture(Option<crate::config::ConfigData>);

    impl Drop for ConfigFixture {
        fn drop(&mut self) {
            if let Some(config) = self.0.take() {
                crate::config::CONFIG.swap(config);
            }
        }
    }

    /// The plan `generate` builds for these fixtures: the real [`plan_walk`] over the
    /// fixture catalog and the given picker.
    async fn plan_for(
        own: &str,
        refs: &Refs,
        size_arg: Option<&str>,
        candidates: &[&str],
    ) -> WalkPlan {
        plan_walk(own, refs.loaded.len(), size_arg, &picker(candidates))
            .await
            .expect("the plan walks")
    }

    /// Run the walk `generate` would run for these fixtures: the real plan, then the
    /// walk itself over scripted answers.
    async fn planned_report(
        own: &str,
        size_arg: Option<&str>,
        candidates: &[&str],
        refs: Refs,
        scripted: &Scripted,
    ) -> WalkReport {
        let plan = plan_for(own, &refs, size_arg, candidates).await;
        walk_report(own, plan, size_arg, refs, scripted).await
    }

    /// One walk driven by scripted answers: what it came to, the failure text it
    /// would report, the note it would attach, and the models it POSTed to.
    struct WalkReport {
        /// The serving model, when one served.
        walked_is_served: Option<String>,
        /// The failure text, when none did.
        failure: Option<String>,
        /// The note for the serving model.
        note: Option<String>,
        sent: Vec<String>,
    }

    /// Run one walk from an explicit plan — for the states [`plan_walk`] never returns
    /// (a catalog that goes unreachable mid-walk, and the walk-only skip the plan
    /// refuses outright).
    async fn walk_report(
        own: &str,
        plan: WalkPlan,
        size_arg: Option<&str>,
        refs: Refs,
        scripted: &Scripted,
    ) -> WalkReport {
        let WalkPlan {
            own_step,
            candidates,
            catalog_reached,
        } = plan;
        let Refs {
            _dir,
            paths,
            loaded,
        } = refs;
        let mut references = loaded;
        // What `generate` resolves for these fixtures — no fixture carries an aspect
        // ratio argument of its own, so it is the detection over the same references.
        let aspect_ratio = request_aspect_ratio(None, &paths, catalog_reached);
        let mut walk = ImageWalk {
            request: RequestMaterial {
                prompt: "a cat",
                aspect_ratio,
                size_arg,
                references: &mut references,
            },
            own_model: own,
            url: "https://example.invalid/images",
            auth: "Bearer test",
            sender: scripted,
            trail: Vec::new(),
            stopped: None,
        };
        let walked = walk
            .run(own_step, candidates)
            .await
            .expect("the walk completes");
        let (walked_is_served, failure, note) = match walked {
            Walked::Served { model, .. } => {
                let note = walk.fallback_note(&model);
                (Some(model), None, note)
            }
            Walked::Unserved => (None, Some(walk.failure_message()), None),
        };
        WalkReport {
            walked_is_served,
            failure,
            note,
            sent: scripted.models_sent(),
        }
    }

    /// One walk that served a picture: whether the account's own model refused the
    /// request, the picker models left to try, and the note the picture carries —
    /// none when the account's own model served it, the fallback disclosure naming
    /// both models when another one did.
    struct ServingNoteCase {
        name: &'static str,
        /// The picker models besides the account's own.
        candidates: &'static [&'static str],
        /// Whether the account's own model refuses the request.
        own_refused: bool,
        /// The model that serves, and the models POSTed to in order.
        served: &'static str,
        sent: &'static [&'static str],
        /// Substrings the note must carry; empty means there is no note at all.
        note: &'static [&'static str],
    }

    #[tokio::test]
    #[serial_test::serial(active_models)]
    async fn serving_note_cases() {
        seed_fixture_catalog().await;
        let own = "qwen/qwen-image-3-pro";
        for case in [
            // Nothing to disclose: the account's own model produced the picture.
            ServingNoteCase {
                name: "the account's own model leaves nothing to disclose",
                candidates: &[],
                own_refused: false,
                served: "qwen/qwen-image-3-pro",
                sent: &["qwen/qwen-image-3-pro"],
                note: &[],
            },
            // The result the assistant receives names the model that served it and
            // what happened with the account's own.
            ServingNoteCase {
                name: "another picker model's picture names it and what the own model did",
                candidates: &["recraft/recraft-v4.1"],
                own_refused: true,
                served: "recraft/recraft-v4.1",
                sent: &["qwen/qwen-image-3-pro", "recraft/recraft-v4.1"],
                note: &[
                    "recraft/recraft-v4.1",
                    "qwen/qwen-image-3-pro",
                    "did not serve the request",
                    "unsupported",
                ],
            },
        ] {
            let scripted = Scripted::new();
            if case.own_refused {
                scripted.answer(
                    own,
                    Err(http(400, r#"{"error":{"message":"unsupported"}}"#)),
                );
            } else {
                scripted.answer(own, Ok(served("aGVsbG8=")));
            }
            for candidate in case.candidates {
                scripted.answer(candidate, Ok(served("aGVsbG8=")));
            }

            let report = planned_report(own, None, case.candidates, Refs::none(), &scripted).await;

            assert_eq!(
                report.walked_is_served.as_deref(),
                Some(case.served),
                "[{}] serving model",
                case.name
            );
            // The own model's 400 is its own refusal: one POST to it — never
            // retried — and then one to the model that served.
            assert_eq!(
                report.sent, case.sent,
                "[{}] one POST per model, in order",
                case.name
            );
            assert_eq!(
                scripted.body_for(case.served)["model"],
                case.served,
                "[{}] the body names the model",
                case.name
            );

            if case.note.is_empty() {
                assert_eq!(
                    report.note, None,
                    "[{}] nothing to disclose without a fallback",
                    case.name
                );
                continue;
            }
            let note = report
                .note
                .unwrap_or_else(|| panic!("[{}] a fallback carries the note", case.name));
            for needle in case.note {
                assert!(
                    note.contains(needle),
                    "[{}] the note must carry {needle:?}: {note}",
                    case.name
                );
            }
            // The note rides with the picture — the result (and so the payload
            // annotation the assistant reads) carries it beside the image.
            let generated = Generated {
                path: std::path::PathBuf::from("x.png"),
                note: Some(note),
            };
            assert!(
                ImageGenTool
                    .result_text(&generated)
                    .contains("did not serve the request"),
                "[{}] the result carries the note",
                case.name
            );
        }
    }

    /// The note carries the provider's own words, and the payload annotation it rides on
    /// is scrubbed nowhere downstream — so the note scrubs them where it is built.
    #[tokio::test]
    #[serial_test::serial(active_models)]
    async fn the_fallback_note_scrubs_the_providers_own_words() {
        seed_fixture_catalog().await;
        let own = "qwen/qwen-image-3-pro";
        let fallback = "recraft/recraft-v4.1";
        let scripted = Scripted::new();
        scripted.answer(
            own,
            Err(http(
                400,
                r#"{"error":{"message":"rejected: api_key: sk-live-0123456789"}}"#,
            )),
        );
        scripted.answer(fallback, Ok(served("aGVsbG8=")));

        let report = planned_report(own, None, &[fallback], Refs::none(), &scripted).await;

        let note = report.note.expect("a fallback carries the note");
        assert!(
            !note.contains("sk-live-0123456789"),
            "a credential in the provider's answer is redacted: {note}"
        );
        assert!(note.contains("*[REDACTED]"), "{note}");
    }

    /// A refusal of the account's own model before anything was sent — here a size
    /// its declarations cannot express — starts the walk instead of ending the
    /// attempt.
    #[tokio::test]
    #[serial_test::serial(active_models)]
    async fn walk_starts_when_the_own_model_is_refused_before_sending() {
        seed_fixture_catalog().await;
        let own = "recraft/recraft-v4.1";
        let fallback = "acme/pixels";
        // A size `acme/pixels` declares; `recraft` has no adjustable size at all.
        let size_arg = Some("2048x2048");
        let scripted = Scripted::new();
        scripted.answer(fallback, Ok(served("aGVsbG8=")));

        let refs = Refs::none();
        let plan = plan_for(own, &refs, size_arg, &[fallback]).await;
        assert!(
            matches!(plan.own_step, Step::Skip(_)),
            "the own model must be refused before sending"
        );

        let report = walk_report(own, plan, size_arg, refs, &scripted).await;

        assert_eq!(report.walked_is_served.as_deref(), Some(fallback));
        assert_eq!(
            report.sent,
            [fallback],
            "the refused own model is never POSTed to"
        );
        assert_eq!(scripted.body_for(fallback)["size"], "2048x2048");

        let note = report.note.expect("a fallback carries the note");
        assert!(
            note.contains("was never sent to"),
            "the note says the own model was never sent to: {note}"
        );
        assert!(
            note.contains("no adjustable image size"),
            "the note carries the refusal: {note}"
        );
        assert!(
            note.contains("hint:"),
            "the note carries the refusal's size hint: {note}"
        );
    }

    /// One walk whose candidates are skipped before sending: the request, the
    /// candidates and the per-model refusal lines the failure must carry.
    struct CandidateSkipCase {
        name: &'static str,
        own: &'static str,
        size_arg: Option<&'static str>,
        refs: usize,
        candidates: &'static [&'static str],
        /// `(model, reason)` — the model's line must carry the reason.
        expected_lines: &'static [(&'static str, &'static str)],
        contains: &'static [&'static str],
        not_contains: &'static [&'static str],
    }

    #[tokio::test]
    #[serial_test::serial(active_models)]
    async fn walk_skips_candidates_that_cannot_serve_the_request() {
        let cases = [
            CandidateSkipCase {
                name: "missing model, text-only model and a capped reference count",
                own: "qwen/qwen-image-3-pro",
                size_arg: None,
                // Two references: `recraft` declares at most one.
                refs: 2,
                candidates: &["gone/model", "text-only/model", "recraft/recraft-v4.1"],
                expected_lines: &[
                    ("gone/model", NOT_AN_IMAGE_MODEL),
                    ("text-only/model", NOT_AN_IMAGE_MODEL),
                    (
                        "recraft/recraft-v4.1",
                        "supports at most 1 reference image(s), got 2",
                    ),
                ],
                contains: &[
                    "Image generation failed: no configured image model served the request.",
                    // A candidate the walk passed over is not the account's pick to
                    // change: the fact alone, no repeated model name, no remediation.
                    "`recraft/recraft-v4.1`: not sent to — \
                     supports at most 1 reference image(s), got 2",
                ],
                not_contains: &["Set an image-capable model"],
            },
            // A candidate whose declarations cannot express the requested size is
            // skipped with the declaration fact, in the walk's own voice: the
            // `usage:`/`hint:` framing speaks to the user's own pick, which a
            // passed-over model never was.
            CandidateSkipCase {
                name: "two candidates whose declarations cannot express the tier",
                own: "qwen/qwen-image-3-pro",
                // qwen expresses the 1K tier; recraft declares no size parameter at
                // all, and acme declares only explicit pixel values.
                size_arg: Some("1K"),
                refs: 0,
                candidates: &["recraft/recraft-v4.1", "acme/pixels"],
                expected_lines: &[
                    (
                        "recraft/recraft-v4.1",
                        "not sent to — has no adjustable image size",
                    ),
                    ("acme/pixels", "not sent to — does not support size `1K`"),
                ],
                contains: &["it accepts: 1024x1024, 2048x2048"],
                not_contains: &["usage:", "hint:"],
            },
        ];

        seed_fixture_catalog().await;
        for case in &cases {
            let scripted = Scripted::new();
            scripted.answer(
                case.own,
                Err(http(400, r#"{"error":{"message":"unsupported"}}"#)),
            );
            let report = planned_report(
                case.own,
                case.size_arg,
                case.candidates,
                refs(case.refs).await,
                &scripted,
            )
            .await;

            assert_eq!(report.walked_is_served, None, "[{}]", case.name);
            assert_eq!(
                report.sent,
                [case.own],
                "[{}] no candidate may be POSTed to",
                case.name
            );

            let failure = report.failure.expect("nothing served");
            for (model, reason) in case.expected_lines {
                let line = failure
                    .lines()
                    .find(|line| line.contains(model))
                    .unwrap_or_else(|| {
                        panic!("[{}] `{model}` is missing from: {failure}", case.name)
                    });
                assert!(
                    line.contains(reason),
                    "[{}] `{model}` says why it was skipped: {line}",
                    case.name
                );
            }
            assert_contains_not_contains(case.name, &failure, case.contains, case.not_contains);
            // The failure reaches the assistant whole, through the truncation a
            // tool result goes through.
            assert_eq!(
                crate::util::truncate_tool_output(&failure),
                failure,
                "[{}] failure must survive truncation whole",
                case.name
            );
        }
    }

    /// Every model the picker offers is offered the request once, in picker order,
    /// the account's own model first.
    #[tokio::test]
    #[serial_test::serial(active_models)]
    async fn walk_visits_each_model_once_in_picker_order() {
        seed_fixture_catalog().await;
        let own = "qwen/qwen-image-3-pro";
        let other = "recraft/recraft-v4.1";
        let scripted = Scripted::new();
        // One scripted answer each: a second POST to either model panics.
        scripted.answer(own, Err(http(402, r#"{"error":{"message":"no credits"}}"#)));
        scripted.answer(
            other,
            Err(http(402, r#"{"error":{"message":"no credits"}}"#)),
        );

        // The raw picker list, de-duplicated through the real plan exactly as
        // `generate` hands it to the walk.
        let report = planned_report(
            own,
            None,
            &[own, other, own, other, other],
            Refs::none(),
            &scripted,
        )
        .await;

        assert_eq!(report.walked_is_served, None);
        assert_eq!(
            report.sent,
            [own, other],
            "one POST per distinct model, in picker order, the own model first"
        );
    }

    /// A failure whose outcome is unknown — a wait that ran out, an answer that could
    /// not be read — is never retried and never handed to another model, since the
    /// request may have been produced and billed.
    #[tokio::test]
    #[serial_test::serial(active_models)]
    async fn walk_never_reissues_an_unknown_outcome() {
        seed_fixture_catalog().await;
        let own = "qwen/qwen-image-3-pro";
        let fallback = "recraft/recraft-v4.1";

        for (failure, cause) in [
            (
                AttemptFailure::Timeout {
                    elapsed_ms: 600_000,
                    connect: false,
                },
                "timed out after 600 s",
            ),
            (AttemptFailure::BodyRead, "body could not be read"),
        ] {
            let scripted = Scripted::new();
            scripted.answer(own, Err(failure));
            // Would have served — the walk must never reach it.
            scripted.answer(fallback, Ok(served("aGVsbG8=")));

            let report = planned_report(own, None, &[fallback], Refs::none(), &scripted).await;

            assert_eq!(report.walked_is_served, None);
            assert_eq!(report.sent, [own], "one POST: no retry, no walk");

            let failure = report.failure.expect("nothing served");
            assert!(
                failure.starts_with("Image generation failed:"),
                "the single-model formatter reports it: {failure}"
            );
            assert!(failure.contains(cause), "{failure}");
            assert!(
                failure.contains(own),
                "the model that ran out is named: {failure}"
            );
            assert!(
                failure.contains("outcome is unknown"),
                "the stop reason is stated: {failure}"
            );
            assert!(failure.contains("The walk stopped there:"), "{failure}");
        }
    }

    /// The reason the walk stopped is reported only when a model was left untried —
    /// with the list exhausted, and with nothing configured to fall back to, the
    /// lone attempt's own failure stands unembellished.
    #[tokio::test]
    #[serial_test::serial(active_models)]
    async fn walk_stop_reason_appears_only_when_a_model_was_left_untried() {
        seed_fixture_catalog().await;
        let own = "qwen/qwen-image-3-pro";
        let fallback = "recraft/recraft-v4.1";
        let timed_out = || AttemptFailure::Timeout {
            elapsed_ms: 600_000,
            connect: false,
        };

        // The account's own model left its outcome unknown with nothing else
        // configured: today's failure exactly, with no line about a walk that has
        // no models in it.
        let scripted = Scripted::new();
        scripted.answer(own, Err(timed_out()));
        let report = planned_report(own, None, &[], Refs::none(), &scripted).await;
        let failure = report.failure.expect("nothing served");
        assert_eq!(
            failure,
            build_terminal_message(std::slice::from_ref(&timed_out()), false),
            "with nothing to walk to the failure is exactly as before the walk"
        );

        // The same outcome with one model left untried says why it was left.
        let scripted = Scripted::new();
        scripted.answer(own, Err(timed_out()));
        let report = planned_report(own, None, &[fallback], Refs::none(), &scripted).await;
        let failure = report.failure.expect("nothing served");
        assert!(failure.contains("The walk stopped there:"), "{failure}");
        assert!(!failure.contains(fallback), "{failure}");

        // A candidate the walk reached last leaves nothing untried either.
        let scripted = Scripted::new();
        scripted.answer(own, Err(http(400, r#"{"error":{"message":"nope"}}"#)));
        scripted.answer(fallback, Err(timed_out()));
        let report = planned_report(own, None, &[fallback], Refs::none(), &scripted).await;
        let failure = report.failure.expect("nothing served");
        assert!(
            failure.contains(fallback),
            "the last model is named: {failure}"
        );
        assert!(
            !failure.contains("The walk stopped there:"),
            "nothing was left untried, so there is nothing to explain: {failure}"
        );
    }

    /// The walk's failure text stays inside the budget the model receives even when
    /// every answer is a long multi-byte refusal: the per-model line budget is
    /// positive, every visited model keeps its own line, and the composed result
    /// survives the tool-output truncation untouched.
    #[tokio::test]
    #[serial_test::serial(active_models)]
    async fn walk_failure_fits_the_budget_with_every_model_on_its_own_line() {
        seed_fixture_catalog().await;
        let own = "qwen/qwen-image-3-pro";
        let candidates = ["recraft/recraft-v4.1", "acme/pixels", "huge/refs"];
        let refusal = format!(
            r#"{{"error":{{"message":"{}"}}}}"#,
            "запрос отклонён провайдером. ".repeat(200)
        );
        let scripted = Scripted::new();
        scripted.answer(own, Err(http(400, &refusal)));
        for model in candidates {
            scripted.answer(model, Err(http(400, &refusal)));
        }
        let report = planned_report(own, None, &candidates, Refs::none(), &scripted).await;
        let failure = report.failure.expect("nothing served");
        let case = "long multi byte answers do not crowd out other models";
        for model in std::iter::once(own).chain(candidates) {
            assert!(
                failure.contains(model),
                "{case}: `{model}` keeps its line: {failure}"
            );
        }
        assert!(
            failure.contains('…'),
            "{case}: a cut answer is marked: {failure}"
        );
        let envelope = delivered(&failure);
        assert_eq!(
            crate::util::truncate_tool_output(&envelope),
            envelope,
            "{case}: the whole result the model receives survives the truncation a tool result goes through"
        );
        for model in std::iter::once(own).chain(candidates) {
            assert!(
                envelope.contains(model),
                "{case}: `{model}` keeps its line in the envelope: {envelope}"
            );
        }

        let case = "the walk failure fits the tool output budget it is delivered in";
        let sender = Scripted::new();
        let walk = refused_walk(
            &sender,
            20,
            &"запрос отклонён провайдером. ".repeat(200),
            Some("`vendor/model-18` was not sent to: request body too large".to_string()),
        );
        let failure = walk.failure_message();
        assert!(
            failure.len() <= WALK_MESSAGE_BYTES,
            "{case}: {} bytes over the failure's own budget",
            failure.len()
        );
        assert!(
            walk_line_bytes(20, 0) > 0 && walk_line_bytes(20, 0) <= WALK_ENTRY_MAX_BYTES,
            "{case}: every model's line has room"
        );
        let envelope = delivered(&failure);
        assert_eq!(
            crate::util::truncate_tool_output(&envelope),
            envelope,
            "{case}: the composed result is not truncated: {} bytes",
            envelope.len()
        );
        for index in 0..20 {
            assert!(
                envelope.contains(&format!("vendor/model-{index}")),
                "{case}: every model keeps its line: {envelope}"
            );
        }
        assert!(
            envelope.contains('…'),
            "{case}: a cut answer is marked: {envelope}"
        );

        let case = "a picker too long to name stays inside the message room";
        let sender = Scripted::new();
        let walk = refused_walk(&sender, 200, "no", None);
        let failure = walk.failure_message();
        assert!(
            failure.len() <= WALK_MESSAGE_BYTES,
            "{case}: {} bytes over the failure's own budget",
            failure.len()
        );
        let envelope = delivered(&failure);
        assert_eq!(
            crate::util::truncate_tool_output(&envelope),
            envelope,
            "{case}: the composed result is not truncated: {} bytes",
            envelope.len()
        );
    }

    /// With nothing else configured to walk to, the failure is byte for byte what a
    /// lone attempt reported before the walk existed.
    #[tokio::test]
    #[serial_test::serial(active_models)]
    async fn single_model_failure_is_exactly_as_before_the_walk() {
        seed_fixture_catalog().await;
        let own = "qwen/qwen-image-3-pro";

        // A single refused attempt.
        let refusal = r#"{"error":{"message":"unsupported"}}"#;
        let scripted = Scripted::new();
        scripted.answer(own, Err(http(400, refusal)));
        let report = planned_report(own, None, &[], Refs::none(), &scripted).await;
        assert_eq!(
            report.failure.as_deref(),
            Some(build_terminal_message(&[http(400, refusal)], false).as_str())
        );
        assert_eq!(report.sent, [own]);

        // The in-attempt retry: the same model, refused twice.
        let overload = r#"{"error":{"message":"down"}}"#;
        let scripted = Scripted::new();
        scripted.answer(own, Err(http(503, overload)));
        scripted.answer(own, Err(http(503, overload)));
        let report = planned_report(own, None, &[], Refs::none(), &scripted).await;
        assert_eq!(
            report.failure.as_deref(),
            Some(
                build_terminal_message(&[http(503, overload), http(503, overload)], false).as_str()
            )
        );
        assert_eq!(report.sent, [own, own], "the retry belongs to the attempt");

        // A step refused before sending: the refusal itself is the failure. This is
        // the walk-only state `plan_walk` refuses outright (nothing to walk to), so
        // the plan is built by hand.
        let reason = resolve_size(
            "recraft/recraft-v4.1",
            Some("2K"),
            fixture_catalog()
                .find("recraft/recraft-v4.1")
                .expect("fixture"),
            OwnModel,
        )
        .expect_err("a size recraft cannot express")
        .to_string();
        let scripted = Scripted::new();
        let report = walk_report(
            own,
            WalkPlan {
                own_step: Step::Skip(reason.clone()),
                candidates: Vec::new(),
                catalog_reached: true,
            },
            None,
            Refs::none(),
            &scripted,
        )
        .await;
        assert_eq!(report.failure.as_deref(), Some(reason.as_str()));
        assert!(report.sent.is_empty());
    }

    /// Only a cause on the chosen model's or the provider's side is walked on — every
    /// HTTP refusal, and a transport or connect failure that never reached the
    /// provider. Unknown outcomes never are.
    #[test]
    fn walkable_failures_are_the_model_or_provider_side_ones() {
        // Every HTTP refusal qualifies, whatever its status and however long the
        // attempt took: the product never judges a refusal from its wording.
        for status in [400, 401, 402, 404, 429, 500, 503] {
            for elapsed_ms in [1_000, 90_000] {
                let failure = AttemptFailure::Http {
                    status,
                    body: String::new(),
                    retry_after_ms: None,
                    elapsed_ms,
                };
                assert!(failure.walkable(), "HTTP {status} after {elapsed_ms} ms");
            }
        }
        // A transport failure is safely repeatable only inside the quick-failure
        // window, where nothing can have been generated.
        assert!(AttemptFailure::Transport { elapsed_ms: 1_000 }.walkable());
        assert!(!AttemptFailure::Transport { elapsed_ms: 90_000 }.walkable());
        // A connect timeout never reached the provider; a request timeout may
        // have completed server-side.
        assert!(
            AttemptFailure::Timeout {
                elapsed_ms: 10_000,
                connect: true,
            }
            .walkable()
        );
        assert!(
            !AttemptFailure::Timeout {
                elapsed_ms: 1_000,
                connect: false,
            }
            .walkable()
        );
        assert!(
            !AttemptFailure::Timeout {
                elapsed_ms: 600_000,
                connect: false,
            }
            .walkable()
        );
        // A 2xx whose body was lost is billable-ambiguous, and a parse failure is
        // not a model/provider-side cause.
        assert!(!AttemptFailure::BodyRead.walkable());
        assert!(
            !AttemptFailure::Parse {
                message: "bad".into()
            }
            .walkable()
        );
    }

    /// Each model is sent a body built from its own catalog declarations — the size
    /// under the key it declares, an aspect ratio only where it declares one, and a
    /// reference-carrying request never turned into a plain generation nor handed to
    /// a model that declares fewer.
    #[tokio::test]
    #[serial_test::serial(active_models)]
    async fn candidate_body_is_built_from_its_own_declarations() {
        seed_fixture_catalog().await;

        // A request carrying one reference and no size: the auto-detected aspect
        // ratio goes only to a model that declares the parameter (qwen does,
        // `huge/refs` does not), and the same reference reaches both.
        let own = "qwen/qwen-image-3-pro";
        let candidate = "huge/refs";
        let scripted = Scripted::new();
        scripted.answer(
            own,
            Err(http(400, r#"{"error":{"message":"unsupported"}}"#)),
        );
        scripted.answer(
            candidate,
            Err(http(400, r#"{"error":{"message":"unsupported"}}"#)),
        );
        let report = planned_report(own, None, &[candidate], refs(1).await, &scripted).await;

        assert_eq!(report.walked_is_served, None);
        assert_eq!(report.sent, [own, candidate]);
        let own_body = scripted.body_for(own);
        // The 1×1 fixture reference auto-detects as 1:1, which qwen declares.
        assert_eq!(own_body["aspect_ratio"], "1:1");
        let body = scripted.body_for(candidate);
        assert!(
            body.get("aspect_ratio").is_none(),
            "huge/refs declares no `aspect_ratio`: {body}"
        );
        assert!(
            body.get("resolution").is_none(),
            "huge/refs declares no `resolution`: {body}"
        );
        assert!(
            body.get("size").is_none(),
            "huge/refs declares no `size`: {body}"
        );
        assert_eq!(body["input_references"].as_array().unwrap().len(), 1);
        assert_eq!(
            body["input_references"], own_body["input_references"],
            "the request keeps carrying the same references"
        );

        // No references: a candidate that declares no aspect ratio gets none.
        let own = "qwen/qwen-image-3-pro";
        let candidate = "acme/pixels";
        let size_arg = Some("1024x1024");
        let scripted = Scripted::new();
        scripted.answer(
            candidate,
            Err(http(400, r#"{"error":{"message":"unsupported"}}"#)),
        );
        let report = planned_report(own, size_arg, &[candidate], Refs::none(), &scripted).await;

        assert_eq!(report.sent, [candidate]);
        let body = scripted.body_for(candidate);
        assert!(
            body.get("aspect_ratio").is_none(),
            "acme/pixels declares no aspect_ratio: {body}"
        );
        assert_eq!(body["size"], "1024x1024");
    }

    /// An unreachable catalog leaves every remaining model's capability unknown, so
    /// the walk skips them and says so instead of POSTing blindly.
    #[tokio::test]
    #[serial_test::serial(active_models)]
    async fn walk_stops_when_the_catalog_cannot_be_reached() {
        let own = "qwen/qwen-image-3-pro";
        let candidate = "recraft/recraft-v4.1";
        let after = "acme/pixels";
        let _restore = CatalogFixture;
        crate::tools::media_catalog::image::seed_cache(None);

        let scripted = Scripted::new();
        scripted.answer(
            own,
            Err(http(400, r#"{"error":{"message":"unsupported"}}"#)),
        );
        // The catalog is unreachable, so the own model's capability is unknown
        // too: the attempt keeps its fail-open behaviour. A plan the walk could
        // never have built: the catalog went unreachable after `plan_walk`.
        let report = walk_report(
            own,
            WalkPlan {
                own_step: Step::Attempt {
                    info: None,
                    size: None,
                },
                candidates: picker(&[candidate, after]),
                catalog_reached: false,
            },
            None,
            Refs::none(),
            &scripted,
        )
        .await;

        assert_eq!(report.walked_is_served, None);
        assert_eq!(report.sent, [own], "no candidate may be POSTed to blindly");

        let failure = report.failure.expect("nothing served");
        assert!(failure.contains(candidate), "{failure}");
        assert!(
            failure.contains("catalog could not be reached"),
            "{failure}"
        );
        assert!(
            failure.contains("remaining models were not checked"),
            "the walk says why it stopped there: {failure}"
        );
        assert!(
            !failure.contains(after),
            "a model the walk never reached is not reported on: {failure}"
        );
        assert_eq!(crate::util::truncate_tool_output(&failure), failure);
    }

    enum PlanGatingScenario {
        OverReferenceCeiling,
        NoPicker,
        PickerOffersAnother,
        CatalogUnreachable,
    }

    /// One [`plan_walk`] gating scenario.
    struct PlanGatingCase {
        name: &'static str,
        scenario: PlanGatingScenario,
    }

    #[tokio::test]
    #[serial_test::serial(active_models)]
    async fn plan_gating_cases() {
        seed_fixture_catalog().await;
        let own = "qwen/qwen-image-3-pro";
        // qwen declares the 1K/2K tiers only, so explicit pixels are refused
        // before anything is sent.
        let size_arg = Some("1024x1024");
        let picker_list = picker(&["acme/pixels"]);

        for case in [
            PlanGatingCase {
                name: "the provider reference ceiling is refused before any model lookup",
                scenario: PlanGatingScenario::OverReferenceCeiling,
            },
            PlanGatingCase {
                name: "with nothing to walk to the pre-send refusal is the plan's error",
                scenario: PlanGatingScenario::NoPicker,
            },
            PlanGatingCase {
                name: "a picker model makes the plan walk",
                scenario: PlanGatingScenario::PickerOffersAnother,
            },
            PlanGatingCase {
                name: "an unreachable catalog walks nothing and stays fail-open",
                scenario: PlanGatingScenario::CatalogUnreachable,
            },
        ] {
            match case.scenario {
                PlanGatingScenario::OverReferenceCeiling => {
                    let refused = plan_walk(
                        own,
                        super::super::MAX_REFERENCE_IMAGES_PER_REQUEST + 1,
                        None,
                        &picker(&["recraft/recraft-v4.1"]),
                    )
                    .await;
                    let Err(err) = refused else {
                        panic!(
                            "[{}] a request over the provider's ceiling must be refused",
                            case.name
                        );
                    };
                    assert!(
                        err.to_string()
                            .contains("at most 16 reference image(s), got 17"),
                        "[{}] {err}",
                        case.name
                    );
                }
                PlanGatingScenario::NoPicker => {
                    let refused = plan_walk(own, 0, size_arg, &[]).await;
                    let Err(err) = refused else {
                        panic!(
                            "[{}] with nothing to walk to the refusal must be the plan's error",
                            case.name
                        );
                    };
                    assert!(
                        err.to_string().contains("does not support size"),
                        "[{}] {err}",
                        case.name
                    );
                }
                PlanGatingScenario::PickerOffersAnother => {
                    let plan = plan_walk(own, 0, size_arg, &picker_list)
                        .await
                        .expect("a picker model to walk to");
                    assert!(matches!(plan.own_step, Step::Skip(_)), "[{}]", case.name);
                    assert_eq!(plan.candidates, picker_list, "[{}]", case.name);
                    assert!(plan.catalog_reached, "[{}]", case.name);
                }
                PlanGatingScenario::CatalogUnreachable => {
                    let _restore = CatalogFixture;
                    crate::tools::media_catalog::image::seed_cache(None);
                    let plan = plan_walk(own, 0, size_arg, &picker_list)
                        .await
                        .expect("an unreachable catalog is fail-open");
                    assert!(plan.candidates.is_empty(), "[{}]", case.name);
                    assert!(!plan.catalog_reached, "[{}]", case.name);
                    assert!(
                        matches!(
                            plan.own_step,
                            Step::Attempt {
                                info: None,
                                size: None
                            }
                        ),
                        "[{}]",
                        case.name
                    );
                }
            }
        }
    }

    /// Each refusal that used to end the attempt — the model gone from the catalog, a
    /// reference count or a size its own declarations cannot take — becomes the
    /// reason the walk starts, while an unreachable catalog keeps the attempt
    /// fail-open.
    #[test]
    fn own_step_turns_each_pre_send_refusal_into_a_walk_reason() {
        let model = "qwen/qwen-image-3-pro";
        let info = fixture_catalog().find(model).cloned().expect("fixture");
        let recraft = fixture_catalog()
            .find("recraft/recraft-v4.1")
            .cloned()
            .expect("fixture");

        assert!(
            matches!(
                own_step(model, Err(anyhow::anyhow!("not in the catalog")), 0, None),
                Step::Skip(reason) if reason.contains("not in the catalog")
            ),
            "a model the catalog confirms absent is refused before sending"
        );
        assert!(
            matches!(
                own_step("recraft/recraft-v4.1", Ok(Some(recraft)), 2, None),
                Step::Skip(reason) if reason.contains("at most 1 reference image")
            ),
            "its own declared reference cap is a pre-send refusal"
        );
        assert!(
            matches!(
                own_step(model, Ok(Some(info)), 0, Some("1024x1024")),
                Step::Skip(reason) if reason.contains("does not support size")
            ),
            "a size its declarations cannot express is a pre-send refusal"
        );
        assert!(
            matches!(
                own_step(model, Ok(None), 0, Some("1K")),
                Step::Attempt {
                    info: None,
                    size: None
                }
            ),
            "an unreachable catalog stays fail-open: nothing is checked, no size sent"
        );
    }

    /// The whole path, not only its parts: the account's own model is refused, another
    /// model the picker offers serves the picture, and the note arrives in the image
    /// annotation that stands in place of the tool's own result — while the account
    /// keeps its own model choice.
    ///
    /// `generate` is the real path both trait entry points run (they hardwire the
    /// production sender, so a scripted one can only enter here); the payload is then
    /// assembled exactly as `execute_with_payloads` assembles it, so what the assistant
    /// reads in place of the tool's own text is what this asserts on.
    #[tokio::test]
    #[serial_test::serial(active_models, config_persist)]
    async fn fallback_reaches_the_assistant_through_the_real_generate_path() {
        crate::util::test::init_test_stores().await;
        seed_fixture_catalog().await;

        let own = "qwen/qwen-image-3-pro";
        let fallback = "recraft/recraft-v4.1";
        let user = "image_gen_fallback_user";
        crate::users::store()
            .set_image_gen_model(user, own)
            .await
            .expect("the account's own model is set");

        // The picker list and a provider key, as the settings page would leave them:
        // restored however this test ends, so a panic here cannot leave the rest of the
        // binary running on this config.
        let _config = ConfigFixture(Some(crate::config::CONFIG.snapshot()));
        let _ = crate::config::CONFIG.set_string_field("provider_key", "test-key");
        let _ = crate::config::CONFIG.set_string_field("image_gen_models", fallback);

        let scripted = Scripted::new();
        scripted.answer(
            own,
            Err(http(400, r#"{"error":{"message":"unsupported"}}"#)),
        );
        scripted.answer(
            fallback,
            Ok(served(&STANDARD.encode(crate::util::test::noisy_png(4, 4)))),
        );

        let dir = tempfile::TempDir::new().expect("tempdir");
        let ws =
            crate::workspace::test_ws_named(&dir.path().to_string_lossy(), "image-gen-fallback");
        let output = crate::agent::CURRENT_TOOL_USER_NAME
            .scope(user.to_string(), async {
                let generated = ImageGenTool
                    .generate(&ws, &json!({ "prompt": "a cat" }), &scripted)
                    .await?;
                anyhow::Ok(ImageGenTool.output(&generated).await)
            })
            .await;
        let output = output.expect("a fallback-covered request is a success");
        assert_eq!(
            scripted.models_sent(),
            [own, fallback],
            "the account's own model first, then the model that served"
        );
        assert!(
            output.text.contains("[IMAGE:"),
            "the marker still reaches the caller: {}",
            output.text
        );
        assert!(
            !output.text_is_content,
            "the picture is offered as a payload, not as text"
        );
        let annotation = output
            .image_payloads
            .first()
            .expect("the picture is attached")
            .attached_annotation();
        assert!(
            annotation.contains(fallback),
            "the assistant is told which model produced the picture: {annotation}"
        );
        assert!(
            annotation.contains(own) && annotation.contains("did not serve the request"),
            "and what happened with the account's own model: {annotation}"
        );
        assert_eq!(
            crate::users::resolve_image_gen_model(user).await,
            own,
            "the account keeps its own stored model choice"
        );
    }
}
