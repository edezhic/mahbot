HyperFrames renders an HTML page to video by seeking a paused timeline frame by frame and encoding the frames with FFmpeg. Every render of the same file is identical, and nothing is uploaded anywhere: the renderer, its browser and its FFmpeg are the product's own.

You do not run the renderer yourself — there is no `hyperframes` command to call. This tool is the whole interface: author the composition as a file in your workspace, then `render` it.

The instruction text below is mahbot's own, written against the exact renderer this tool drives; it follows the guidance HyperFrames publishes (HyperFrames, heygen-com/hyperframes, Apache-2.0) and is not the vendor's text.

## The composition

A composition is an ordinary HTML document whose root element carries `data-composition-id`, `data-width`, `data-height` and `data-duration`. Everything visible lives in that root; the video is exactly `data-duration` seconds long — overrun is cut, an early end holds the last frame.

Each timed element is a clip: `class="clip"` with `data-start` and `data-duration` in seconds, local to the file that declares it. A clip shows over `[start, start + duration)` — an end state placed exactly on the end is never rendered, so land the final state a frame before it.

Every composition builds one paused timeline and registers it under its own id, after its tweens:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <script src="https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"></script>
    <style>
      html, body { margin: 0; width: 1280px; height: 720px; overflow: hidden; background: #0b1020; }
      #root { position: relative; width: 100%; height: 100%; }
      #title { position: absolute; left: 80px; top: 260px; color: #fff; font: 700 96px sans-serif; }
    </style>
  </head>
  <body>
    <div id="root" data-composition-id="main" data-start="0" data-duration="6"
         data-width="1280" data-height="720">
      <h1 id="title" class="clip" data-start="0.3" data-duration="5.2">Hello</h1>
    </div>
    <script>
      const tl = gsap.timeline({ paused: true });
      tl.fromTo("#title", { opacity: 0, y: 40 },
        { opacity: 1, y: 0, duration: 0.5, ease: "power3.out" }, 0.3);
      window.__timelines["main"] = tl;
      tl.seek(0);
    </script>
  </body>
</html>
```

A longer video is a project folder: `index.html` as the root composition, scene files under `compositions/`, media and fonts under `assets/`. A scene file wraps its style, markup and script in `<template>` and is mounted from the root by `data-composition-src`, with the host, the scene root and `window.__timelines[...]` all sharing the same id.

## The three rules that break renders

1. **Everything visible is a function of timeline time.** No `Date.now()`, no unseeded `Math.random()`, no `setTimeout`, no `repeat: -1`, no network fetches at runtime, never `play()`, `pause()` or seek media. A frame must be reproducible from its timestamp alone.
2. **Register the timeline.** A composition (and every scene) that does not set `window.__timelines[<data-composition-id>]` makes the render poll for 45 s and then show static frames. A genuinely still composition declares `data-no-timeline` on its root instead.
3. **Animate what the renderer owns.** Never tween `display`, `visibility` or `autoAlpha` on a `.clip` — animate a child. Never put a CSS `transform` on an element GSAP transforms. Give every `<audio>` an `id`; a timed `<video>` needs `data-has-audio="true"` or `muted`; time a `<video>` or its wrapper, never both.

`data-track-index` is a Studio lane only: the render never reads it, and paint order is CSS `z-index`. Tracks do not prevent overlap.

Fonts: a family installed on the machine resolves by name, and so do the families the renderer bundles (`reference` → `authoring` lists them, and says which are only fetched from Google at render time). Any other family needs an `@font-face` pointing at a file in the composition's own `assets/fonts/`: a name alone is fetched from Google when the render reaches it, which needs a working connection. Asset paths are relative to the composition file (`assets/logo.png`), never absolute.

Rendered pages do reach the network — a CDN script, a web font, a remote image is fetched while the page loads. That is normal and allowed; only runtime *fetches driven by your JavaScript* are banned, because they make frames irreproducible.

## Rendering

`render` takes the composition — a path to an `.html` file, or a project folder holding `index.html` — and an output file. Options:

- `format`: `mp4` (H.264, the default), `webm` (VP9/Opus, supports transparency), `mov` (ProRes 4444, supports transparency), `gif`.
- `quality`: `draft` (fast iteration), `looks` (default), `delivery` (slower, best).
- `fps`: an integer 1–240 or an FFmpeg-style rational such as `30000/1001`; defaults to the root's `data-fps`, else 30.
- `resolution`: `landscape`, `portrait`, `square`, or their `-4k` variants; the composition is captured at a higher device pixel ratio, so the aspect ratio must match and the scale must be a whole multiple.

A render produces exactly one file. A frame count follows from duration and fps (6 s at 30 fps is 180 frames); a minute of 1080p is a normal size, and the work grows with resolution, fps and quality.

There is no timeout on a render, and a long one is not a stalled one. Do not re-issue a render because it is slow — wait for it. The one thing that can end one early is the product stopping a run that overruns the machine's memory, which is reported as a machine-side failure rather than as a broken composition.

## When a render fails

Run `inspect` first: it checks the composition without rendering and reports structural mistakes with a code, a message and a fix. Most failures are one of: a missing `data-width`/`data-height` on the root, an unregistered timeline, a clip whose slot ends past the root duration, or a scene whose host id differs from the scene's.

Then read what the tool says. A failed render carries the renderer's own code and message and keeps its `hint:` line — read the hint, which names the cause and the fix. It also names the piece that is missing — the renderer package, its browser, the product's FFmpeg, the network on a first use — rather than a command to run. A failure that names the composition itself is about your HTML; a failure that names the file the render wrote is about the machine.

What decides a render is the file: one you can play is a success whatever the renderer's exit status or printed failure said, and no such file is a failure with the renderer's own diagnosis. Anything the product or the renderer had to say after the write — a stopped run, a run that ended badly, a failed step after the file, a probe that could not check it — is named beside the video, and the video stands.

Nothing half-written is left behind: a failed render removes the file it was writing.

## What this tool does with files

- It reads the composition under your own access level, and writes only where you ask — inside your workspace, or anywhere for the administrator's assistant. What the composition itself refers to — a script, an image, a font — is fetched by the renderer's browser while the page loads: a relative path resolves inside the composition's own folder, and a remote URL is fetched from the network. The page is served to that browser from a loopback address rather than from the file system, so a reference to a local path outside the folder being served does not resolve.
- An output path is never overwritten: if the name is taken, the product picks the next free one (`clip_2.mp4`) and reports the path it actually wrote. A folder that does not exist yet is created for it.
- A render leaves a `.hyperframes/` state folder beside the composition. It is the renderer's own bookkeeping, and the product puts a `.gitignore` inside it so a repository never commits it — leave it in place, and do not delete it while a render is running.
- Only single-file formats are produced. Image sequences and HLS playlists are not offered: their output is a directory, which this product does not deliver.

Deeper detail: `reference` with `topic: "authoring"` (composition format, attributes, scenes, media, variables), `"rendering"` (pipeline, formats, quality, frames, output rules) or `"troubleshooting"` (lint codes and failure classes).
