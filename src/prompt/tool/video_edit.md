Edit an existing video clip. Two modes:

1. **Reference edit** — pass `video_url` (a public HTTPS URL, or a local path from the workspace `uploads/` received-attachments dir — shown as "[Saved video: /path]" in the chat — or from the workspace `generated/` dir — shown as "[VIDEO:path]", the output of a previous video_gen/video_edit) plus a text `instruction` (max 5000 chars) describing the edit. Optionally add up to 9 reference images via `images` (local paths from `uploads/` or `generated/`, or public HTTPS URLs) to guide style/subject/identity.
2. **Image-to-video** — pass `first_frame` and/or `last_frame` (local paths from `uploads/` or `generated/`, or public HTTPS URLs) to anchor the exact first/last frame. Frame anchors are mutually exclusive with ALL reference inputs: a request combining them with `video_url` or `images` is rejected.

Editing technique (style changes):
- For a style change, pair the source video with a style-reference image — generate one via image_gen if none exists — and give each reference an explicit role in the instruction (e.g. "apply this image's color grade to the clip").
- Name the target style concretely and state the transformation explicitly; never vague phrasing like "make it artistic".
- Keep composition, camera, movement, and timing unchanged; change ONE visual category per call, in order: background → wardrobe → lighting → grade.
- Full restyle while preserving the plot is at the edge of every current model: iterate single-category passes and verify each pass before continuing.

Local paths are accepted ONLY from the workspace `uploads/` directory (received media attachments) or the workspace `generated/` directory (previously generated media); any other local path is rejected. Image inputs must be JPG/JPEG/PNG/WEBP/HEIC/HEIF and at most 30 MB; source clips are limited to 50 MB. Per-model parameter support (duration, image inputs, frame anchors) is listed in the active model's capability block when present; otherwise keep parameters conservative and model-agnostic.

Each invocation bills a paid job and can take up to 1 hour to complete (provider queue stalls are common — do not abandon a job that is still pending). Never retry a timed-out edit — that bills a second job; instead report the timeout to the user. Returns the path to the edited video file (the result also includes a "Video content:" description of what the edited clip shows) — use [VIDEO:path] in your reply to send it to the user. Never attempt more than 1 generation before sending the result to the user - avoid long waiting and excessive billing.