The composition format in full: project shape, the attributes that carry timing, scene files, variables, media, fonts, and the lint codes the checker reports. Read this before a composition that is more than a title card.

## Project shape

A single self-contained `.html` file renders fine and is the right size for a title card, a lower third, a chart reveal. Anything longer is a folder, because a folder keeps the scenes editable:

```
project/
  index.html                 the root composition: variables, scene hosts, media, audio
  compositions/intro.html    one scene per file
  assets/logo.png            media and fonts, referenced as assets/...
```

`index.html` is rendered by default when the composition is given as a folder. Any other file is rendered by naming it; it must live inside the project folder, and the nearest ancestor folder holding an `index.html` is treated as the project root.

## The root

| Attribute | Meaning |
| --- | --- |
| `data-composition-id` | The composition's id; also the key its timeline registers under. Required. |
| `data-width`, `data-height` | The composition canvas in pixels. Required — `1920`×`1080` is the usual landscape frame, `1080`×`1920` portrait. |
| `data-duration` | The video's length in seconds. Required. Overrun is cut; an early end holds the last frame. |
| `data-fps` | Optional source frame rate; a render without an explicit `fps` uses it, else 30. |
| `data-no-timeline` | Declares a still composition, so the renderer does not wait 45 s for a timeline that never arrives. |
| `data-motion-blur="on"` | Opt-in render-time motion blur along a moving element's real path. |

A palette fill on the root is dropped when the video has HDR media; put the background on a full-bleed child (`position: absolute; inset: 0`) instead.

## Clips and timing

Every timed element is `class="clip"` with `data-start` and `data-duration`, in seconds, local to the file that declares it. It shows over `[start, start + duration)`: an end state landing exactly on the end is never rendered, so land it a frame early.

| Attribute | Meaning |
| --- | --- |
| `data-start`, `data-duration` | When the element shows, and for how long. |
| `data-track-index` | Studio lane only. The render never reads it; paint order is `z-index`; it prevents no overlap. |
| `data-track-kind` | `graphics` or `captions` on scene hosts. Video and audio take their kind from the tag. |
| `data-composition-src` | Mounts a scene file at this host. |
| `data-variable-values` | A JSON object of per-mount variable overrides for one scene host. |
| `data-media-start` | Offset into the source media — a trim point. |
| `data-playback-rate` | Constant speed, 0.1–10. |
| `data-has-audio="true"` | Keeps a `<video>`'s own sound. A timed `<video>` with neither this nor `muted` fails lint. |
| `data-volume`, `data-fade-in`, `data-fade-out` | Gain (`1` is 0 dB, up to `3.98` = +12 dB) and linear edge fades, multiplied on top of the gain. |
| The vendor's layout family — `data-layout-name`, `data-layout-check`, `data-layout-ignore`, the `data-layout-allow-*` waivers (`-overflow`, `-overlap`, `-occlusion`, `-caption-zone`, `-orbit`), `data-layout-boundary`, `data-layout-bleed` | Read only by the vendor's `inspect`, `check`, `layout` and `snapshot` commands. This tool runs `render` and `lint`, so none of them changes anything it produces. |

- A `<video>` or `<audio>` without `data-duration` plays its full length; an `<img>` without one lasts 3 s.
- A host's `data-duration` is the scene's slot: a shorter scene timeline holds its last frame, and a slot shorter than the scene goes blank at its end.
- Root children carrying `data-start` are laid out full-frame by the runtime.
- `data-start="intro + 0.5"` starts after the `intro` clip ends — spaces around `+` are required, and a typo, a missing id or a cycle silently resolves to 0.
- A clip nested inside another timed element counts its `data-start` from the composition, not from its parent, and shows only while the parent shows.
- Never `tl.play()`. Only the root composition's timeline is paused; nested timelines are not.

## Scenes

A scene file wraps everything in `<template>` — styles, markup and script outside it are dropped:

```html
<template>
  <style>
    #root { position: absolute; inset: 0; }
    #headline { color: #fff; font: 600 72px sans-serif; }
  </style>
  <div id="root" data-composition-id="intro" data-width="1920" data-height="1080">
    <h1 id="headline" class="clip" data-start="0.4" data-duration="3">…</h1>
  </div>
  <script>
    const tl = gsap.timeline({ paused: true });
    tl.fromTo("#headline", { opacity: 0, filter: "blur(8px)" },
      { opacity: 1, filter: "blur(0px)", duration: 0.18, ease: "power4.out" }, 0.4);
    window.__timelines["intro"] = tl;
  </script>
</template>
```

- The host, the scene root and the timeline key share one id. A mismatch makes the render wait 45 s per scene and then show static frames.
- Size the scene root with `inset: 0`, never with hard-coded pixels; the canvas is the root's `data-width`×`data-height`.
- Keep ids unique across the assembled page (`#intro-title`); a scene's timeline cannot reach elements outside its own file.
- Use `fromTo`, never `from`, in a scene: the host re-seeks the scene every time it shows, and `from` records its start state once, so it drifts on seek-back.
- Never `add()` a scene's timeline to the root's timeline; the runtime drives each scene itself and would seek it twice.

## Variables

The `<html>` element of a composition may declare a schema:

```html
<html data-composition-variables='[
  {"id":"title","type":"string","label":"Title","default":"Hello"},
  {"id":"accent","type":"color","label":"Accent","default":"#ff4d4f"}
]'>
```

`window.__hyperframes.getVariables()` inside the composition returns the merged values: declared defaults first, then a host's `data-variable-values` for that mount. A top-level composition gets only its defaults — this tool does not pass render-time variable overrides.

## Media, sound and fonts

- Every `<audio>` needs an `id` or it is left out of the mix. No `crossorigin` on media.
- Time a `<video>` or its wrapper, never both; a nested video inside a timed element fails lint.
- A `<video>` keeps its sound with `data-has-audio="true"`, otherwise mark it `muted`. Music, voice and replacement sound are a separate `<audio>`.
- A family installed on the machine resolves by name. These the renderer bundles, so they resolve with no network at all: Inter, Roboto, Montserrat, Poppins, Outfit, Nunito, Oswald, League Gothic, Archivo Black, Space Mono, IBM Plex Mono, JetBrains Mono, EB Garamond, Playfair Display, Source Code Pro, Noto Sans JP, Open Sans, Lato. Bebas Neue is aliased to the bundled League Gothic: it resolves offline, but it renders as League Gothic, never as Bebas Neue. Any other family — Anton, Lora, Pacifico, Permanent Marker and Fira Code among them — is fetched from Google at render time, so it needs a working connection, and `inspect` reports `font_family_without_font_face` for it unless the composition declares it with an `@font-face` or a Google Fonts `<link>`. Embedding the file in `assets/fonts/` and declaring it is what makes a family work offline.
- Asset paths are `assets/...`, relative to the composition file, never absolute.

## Motion that reads as intended

- One move is one tween: a fast move takes 0.3–0.6 s with a long-tailed ease-out, and the landing takes longer than the leaving.
- A move of a whole frame is 15–20 px, of a text-sized element 6–12 px. Anything larger in the same time reads as a jump cut.
- Animate `opacity` and transforms. Animating layout properties (`left`, `top`, `width`, `height`) re-lays-out the page every frame.
- Exit in one motion — never slide, stop, then leave.
- A still composition (a card, a lower third that only fades in) needs no more than the entrance; mark it `data-no-timeline` only if there is no timeline at all.

## Lint codes you will meet first

`inspect` reports these by code, with a message and a fix. The common ones:

| Code | Fix |
| --- | --- |
| `root_missing_dimensions` | Set numeric `data-width` and `data-height` on the entry composition root. |
| `missing_timeline_registry` | Register the composition's timeline on `window.__timelines[<id>]`. |
| `missing_data_no_timeline` | A root or scene host with no timeline: add `data-no-timeline`, or every render polls 45 s for one. |
| `video_missing_muted` | A timed `<video>` gets `data-has-audio="true"` or `muted`. |
| `media_missing_id`, `media_crossorigin_breaks_preview` | Give every `<audio>` an `id`; drop `crossorigin`. |
| `gsap_css_transform_conflict` | Put the start state inside the tween (`fromTo`) instead of a CSS `transform`. |
| `gsap_animates_clip_element` | Animate a child of the `.clip`, or only its `opacity`. |
| `video_nested_in_timed_element` | Time the `<video>` or its wrapper, not both. |
| `standalone_composition_wrapped_in_template` | Only scene files use `<template>`. |
| `subcomposition_root_styled_by_class` | Style a scene's root as `#root`. |
| `subcomposition_blanks_before_host` | Make the scene's slot span the time it should show. |
| `clip_ends_past_root_duration` | Raise the root's `data-duration`. |
| `linked_clips_out_of_sync`, `linked_clip_orphan` | Retime every linked member together. |

A lint warning is not a failed render — the render proceeds and the flaw shows in the video. Fix errors before rendering; read warnings as the thing you will see in the output.
