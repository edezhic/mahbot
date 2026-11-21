# video_gen — reference

One call, three mutually exclusive shapes, decided by the inputs you pass:

1. **A new clip** — `prompt` alone, optionally with ONE reference picture in `images` to start from (image-to-video; a local path inside your workspace, never a URL); omit the picture for text-to-video. Takes `duration`, `resolution`, `aspect_ratio`, `size`, `generate_audio` and `seed`.
2. **A new clip anchored on frames** — `first_frame` and/or `last_frame` (a local path inside your workspace or a public HTTPS URL; JPG/JPEG/PNG/WEBP/HEIC/HEIF, at most 30 MB) to pin the exact first and last frame.
3. **An edit of an existing clip** — `video_url` (a public HTTPS URL, or a local path inside your workspace — an attachment is shown as "[Saved video: /path]" in the chat, a previous call's output as "[VIDEO:path]") plus the `prompt` describing the edit. Up to 9 reference images via `images` — local paths inside your workspace or public HTTPS URLs — guide style/subject/identity.

## Arguments

- `prompt` — one text input for all three shapes, capped at 5000 characters.
- `video_url` — the clip to edit; present, the call edits that clip, omitted, a new clip is generated.
- `images` — a new clip's ONE start picture, or an edit's up to 9 guiding pictures.
- `first_frame`, `last_frame` — the exact first and/or last frame of a new clip.
- `duration` — output duration in seconds for a new clip; the active model's capability block gives the valid values.
- `resolution` — desired resolution for a new clip, from the active model's capability block.
- `aspect_ratio` — aspect ratio for a new clip, from the active model's capability block.
- `size` — exact size of a new clip in `WxH` form (e.g. `1280x720`).
- `generate_audio` — whether a new clip carries an audio track.
- `seed` — seed for a repeatable generation of a new clip.

Frame anchors are mutually exclusive with ALL reference inputs: a request combining them with `video_url` or `images` is rejected. A parameter belongs to the shape that takes it: `resolution`, `aspect_ratio`, `size`, `generate_audio` and `seed` describe a new clip only, and passing one together with a video reference or with frame anchors is refused rather than silently dropped. An input that is present but empty (`video_url`, an anchor, a picture in `images`, `resolution`, `aspect_ratio`, `size`) names nothing and is refused too, and so is a value of the wrong type — omit the argument instead.

## Restyling a clip

For a style change, pair the source with a style-reference image (generate one via `image_gen` if none exists) and give each reference an explicit role in the prompt; name the target style concretely; keep composition, camera, movement and timing unchanged and change ONE visual category per call, in order: background → wardrobe → lighting → grade. Full restyle while preserving the plot is at the edge of every model: iterate single-category passes and verify each pass before continuing.

## References and limits

A new clip's reference must be PNG, JPEG or WebP. References over 1.5 MB are compressed (downscaled and re-encoded as JPEG), and one near the cap may be compressed further to keep the request within the ~2 MB body budget. Missing, unreadable or undecodable references are hard errors — the tool fails visibly rather than falling back to text-to-video, so fix or remove the reference and retry.

An edit's guiding pictures and the frame anchors must be JPG/JPEG/PNG/WEBP/HEIC/HEIF and at most 30 MB; source clips are limited to 50 MB, and local MP4/MOV clips whose duration falls outside the active model's input range are rejected before upload.

`resolution`, `aspect_ratio`, `size` and a new clip's `duration` are checked against the active model's capability block before submission — a value the model does not declare is refused, never sent — while an edit is not measured against that range. When the block is absent nothing is pre-checked, so keep those parameters conservative and model-agnostic.

## The result

Provider queue stalls are common — do not abandon a pending job. Never make more than one call before sending the result to the user.

The timeline groups frames four to a picture, each frame with its mark and each picture naming the frames it holds, with the speech in marks and text and a sound label where there is no speech.
