Edit an existing video clip. Two modes:

1. **Reference edit** — pass `video_url` (a local path shown as "[Saved video: /path]" in the chat, or a public HTTPS URL) plus a text `instruction` (max 1000 chars) describing the edit. Optionally add up to 9 reference images via `images` (local paths or public HTTPS URLs) to guide style/subject/identity; each reference image bills a $0.04 surcharge.
2. **Image-to-video** — pass `first_frame` and/or `last_frame` (local paths or public HTTPS URLs) to anchor the exact first/last frame. Frame anchors are mutually exclusive with ALL reference inputs: a request combining them with `video_url` or `images` is rejected.

Image inputs must be JPG/JPEG/PNG/WEBP/HEIC/HEIF and at most 30 MB; source clips are limited to 50 MB. Parameter support depends on the active model — hailuo-3 supports both modes with 5–15 s output; aleph-2 does not support image inputs and no `duration`.

Each invocation bills a paid job and takes ~3–4 minutes to complete. Never retry a timed-out edit — that bills a second job; instead report the timeout to the user. Returns the path to the edited video file — use [VIDEO:path] in your reply to send it to the user. Never attempt more than 1 generation before sending the result to the user - avoid long waiting and excessive billing.