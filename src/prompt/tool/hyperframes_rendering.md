What a render does, what each format produces, what the options mean, where the file lands and what the tool reports afterwards.

## What happens during a render

The composition is compiled in a headless browser: its scenes are assembled, its timeline is paused, and then the renderer seeks that timeline frame by frame — frame *n* is the page at `n / fps` seconds — capturing each frame and handing it to FFmpeg. That is why a render is reproducible and why an animation that depends on wall-clock time or randomness cannot render correctly: nothing advances except the seek.

A render is CPU-bound and grows with frames, resolution and quality. A 6 s 1080p composition at 30 fps is 180 frames and takes seconds to a couple of minutes depending on the machine; a minute of 4K is a different order of magnitude. There is no timeout, no frame cap and no duration cap, so a long render is expected to take long — wait for it instead of issuing it again.

The renderer compiles the page first and prints its own progress; the tool does not forward that noise. Only the outcome reaches you.

## Formats

| `format` | Container and codec | Transparency | Typical use |
| --- | --- | --- | --- |
| `mp4` (default) | MP4, H.264 with AAC audio | No | Delivery: the file plays everywhere. |
| `webm` | WebM, VP9 with Opus audio | Yes | Overlays and web delivery where alpha matters. |
| `mov` | MOV, ProRes 4444 with alpha | Yes | Editing: ingest into another editor, big files. |
| `gif` | Animated GIF | — | A short silent loop for a chat or a README. |

GIF is best at 15 fps and a few seconds; it has no audio and its palette is limited. ProRes files are large — a minute of 1080p is on the order of a gigabyte.

Image sequences and HLS are not offered: both produce a folder, and this product delivers files.

## Options

- `quality`: `draft` encodes fast with visible banding — right for checking timing and layout. `looks` is the default and is what most clips want. `delivery` is the slow, clean master.
- `fps`: an integer 1–240 or an FFmpeg rational (`30000/1001`, `24000/1001`, `60000/1001`). Omitted, the root's `data-fps` decides, else 30. A higher rate samples the same timeline more often — smoother motion, the same duration, more frames to encode.
- `resolution`: `landscape` (1920×1080), `portrait` (1080×1920), `square` (1080×1080), or their `-4k` variants. The composition is not re-laid-out: the same canvas is captured at a higher device pixel ratio, so the composition's aspect ratio must match the preset and the scale must be a whole multiple, otherwise the render is refused.

## Output

- The file goes exactly where you name it. Without `output`, it lands in your workspace's `generated/` folder.
- An existing file is never overwritten: the product writes the next free name instead (`clip_2.mp4`, `clip_3.mp4`) and reports the path it used, so read the path in the result rather than the one you asked for.
- If the output's extension disagrees with `format`, the call is refused rather than writing a mislabelled container.
- A render that produced no file a player can open is a failure, and the file it was writing is removed. A render that produced one is a success and leaves exactly one new file — if the renderer complained about a step after writing it, the answer adds that step in one sentence.
- The renderer keeps a `.hyperframes/` state folder beside the composition (recent-video state and a history id). Leave it in place; the product puts a `.gitignore` inside it, so a repository never commits it.

## The result

The result starts with a `[VIDEO:path]` marker — put it in your reply to send the video — followed by what was produced: duration, pixel size, container and file size, and the frame rate when you gave `fps` as a whole number. When the product or the renderer had something to say after the file was written — a stop, a run that ended badly, a failed step after the write, a probe that could not check it — a further sentence says so; the video is still the result.

## The machine side

The renderer runs on the product's managed bun runtime, with the product's own headless Chrome for Testing and the product's own FFmpeg. None of it comes from the user's installed software: a machine with no Chrome and no FFmpeg still renders, and a Chrome the user happens to have is never used.

The first call on a machine installs the renderer package from the npm registry, and the first render then downloads the browser into the product's storage — hundreds of megabytes, minutes rather than seconds. That is a first-use cost only: later calls reuse both, and a browser build this machine has already proved is used as it is, so the newest published build is fetched only when a download is needed. The product keeps the renderer at its newest release by itself.

Rendered pages reach the network by design: a CDN script, a web font or a remote image is fetched while the composition loads. Nothing blocks that, and a composition is free to depend on a CDN — but what your own JavaScript fetches at runtime makes frames irreproducible and is banned.
