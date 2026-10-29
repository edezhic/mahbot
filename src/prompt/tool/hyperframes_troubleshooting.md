How a failure reads, what each class means, and what to fix instead of retrying blindly. Run `inspect` before every render: a lint pass reports many of the composition problems below without waiting for a render.

## The tool says a piece is missing

These are facts about the machine, not about your composition. Say them to the user in your own words, plainly, and do not try to install anything yourself — the tool installs and refreshes everything it needs.

| What it says | What it means |
| --- | --- |
| `the product's managed bun runtime is not installed` | The product's JavaScript runtime is absent; nothing can run until it is restored. |
| `the renderer could not be installed: …` | The renderer package could not be fetched from the npm registry — a machine with no network on a first use, a registry that refused, or a download the product stopped (`timeout: installing the hyperframes package did not finish within 1800s and was killed`). The sentence continues with the failing step (`the release check request failed`, `installing the hyperframes package failed`). The next call tries again; a first use needs a working connection. |
| `the renderer's browser is unavailable: …` | The headless browser is fetched once, on the first render. Same network condition; the download is large and slow links fail it. The same sentence also carries `this platform is not supported` — the product ships six targets and this is not one of them. A build that does not run on this machine is named as such and is not downloaded again; the newest build published is fetched when a download is needed, and a build this machine already proved is reused. |
| `the renderer's browser could not be checked: …` | The product could not finish proving its own browser build: the check itself was stopped by the machine (the build never started, the guard's memory ceiling or time limit ended it). Nothing is wrong with the composition and nothing is recorded against the build: an older build that proves is used instead, the build is checked again on the next call, and it is never written down as one that cannot run here. |
| `the product's own FFmpeg payload is unavailable: …` | The product's own FFmpeg could not be installed, so nothing can encode. |
| a render failure containing `… has no encoder for this format on this machine` | The payload is there but cannot encode what was asked for (an option this build does not carry). Not a composition problem: pick another format, or say plainly that this machine cannot produce it. |

## The render failed

A failed render carries the renderer's own diagnosis: `<code>: <message>`, then a `hint:` line naming the cause and the fix when it knows one. Read the hint before guessing — the message alone is often just `Render failed`.

The code is almost always `runtime_error`; the specific cause is named in the hint, as `sub_timeline_script_failure`, `audio_processing_failed` or a message of the renderer's own.

| What the hint names | What it means |
| --- | --- |
| `sub_timeline_script_failure` | A script in the composition threw while the timeline was being sought. The hint gives the location — the composition file and line when the renderer can name them, an in-page `body:1:N` position otherwise; the page is fine, its JavaScript is not. |
| `audio_processing_failed` | An audio asset could not be read or mixed. The hint names the clip; check the `src` and that the file is really there and decodable. |
| a `resolution` or the composition's aspect ratio | The preset does not fit the page; the hint names the preset that would work. |

A render the product itself ended is not one of the renderer's failures: `terminated: exceeded memory limit (used ~… of …)` means the machine ran out of memory, and the render is gone rather than broken. Say so plainly; lowering `resolution`, `quality` or duration is the user's call, not a fix to apply silently. A renderer process that exited on its own is judged by the status it gave, whatever a process it left behind then did to the output channel.

What came out of the call reads as one of these. A failed `inspect` reports the same shapes with `the check` in place of `the render`, and its own stop is described under the next heading.

| What it says | What it means |
| --- | --- |
| `the render failed: exit status N`, then the renderer's last lines | It exited non-zero and left no file to play. The lines below are its own words — paste them rather than paraphrasing. |
| `the render failed: failed to start …`, or `terminated: …`, with a tail | The process never started, or the product ended it — the status says which, and the tail is what it printed. |
| `the render failed: the renderer exited clean but wrote no playable file at …` | It reported success and produced nothing a player can open. The lines below say what it printed; treat it as a renderer or machine problem before rewriting the composition. |
| `<code>: <message>` with a `hint:` line | The renderer's own failure entry — the shape the section above explains, reported whatever the exit status was. |
| `the file was written; a step after the render failed: …` | The video is there and playable — something the renderer does after writing it failed. Deliver the video; mention the step only if the user needs it. |
| `the file was written; the renderer's run ended: exit status N` (or `… ended: terminated by signal`) | The video is there and playable, and the run itself exited badly — its own teardown usually. Deliver the video; mention it only if the user needs it. |
| `the file was written; the renderer reported a failure after the write: <code>: <message>` | The video is there and playable, and the renderer printed its own failure entry after the write. Deliver the video; the entry is what it says, not a reason to re-render. |
| `the file was written; the renderer reported a failure after the write: <code>` | Same, with an entry whose message was nothing but the vendor's own advice, which the tool strips. The code is all the renderer said that is worth passing on. |
| `the product stopped the render after the file was written: …` | The video is there and playable, and the product ended the run after the write (usually the memory guard). The video stands; say what stopped it. |
| `this machine could not check the file: …` | The video was written and the product's own probe could not answer, so it was not verified. The file is kept — deliver it, and say it went unchecked. |
| `the check exited clean but printed no report this tool could read` | The static check answered nothing usable, so nothing was verified — never read it as a clean composition. |
| `the render reported a failure without a diagnosis`, or `the file was written; the renderer reported a failure after the write: no diagnosis` | Everything the renderer said was its own advice, which the tool strips. Report that it failed and that it gave no reason. |

## The check was stopped

`timeout: the check did not finish within 600s and was killed` — the inspection was ended at the tool's own ceiling, which a composition this tool can check never reaches. A stop is not a finding: retry once, and report it plainly if it happens again.

## The composition is refused before rendering

| Message | Cause |
| --- | --- |
| that folder has no index.html | The folder named as the composition holds no `index.html`; name the composition file instead, or add one. |
| `usage: \`fps\` must be …`, `\`resolution\` must be one of …`, `\`format\` must be one of …` | The value is not one the renderer accepts. fps is 1–240 or a rational; resolution must match the composition's aspect ratio at a whole multiple. |
| `format … writes a .… file, but the output is named .…` | `format` decides the container; name a file with that extension or change the format. |

## The render ran and the video is wrong

**Black or blank frames.** The most common cause is a missing timeline registration: the composition never sets `window.__timelines[<data-composition-id>]`, so the renderer shows a page that was never sought. Next: a scene host whose id differs from the scene's own id — the scene renders empty, 45 s late. Next: a `data-start` expression that silently resolved to 0, so the clip is gone before the first captured frame. Next: the element is there but invisible — `opacity: 0` left as the final state, a `display: none` toggled by the timeline, or `overflow: hidden` on an ancestor clipping it.

**The video never moves.** A still composition without `data-no-timeline` renders static by definition; a composition whose animation lives in `setTimeout`, `requestAnimationFrame` driven by wall time or `repeat: -1` never advances under a frame-by-frame seek.

**The video is the wrong length.** The root's `data-duration` *is* the length: a longer timeline is cut, a shorter one holds its last frame. A clip whose slot ends past the root duration is cut mid-motion — either raise the root duration or retime the clip.

**A scene is blank where it should play.** The host's `data-duration` is the scene's slot. A slot that starts after the scene does, or ends before the scene's own animation completes, shows a still or nothing; a host id that differs from the scene's makes the whole scene wait 45 s.

**There is no sound.** Every `<audio>` needs an `id` or it is left out of the mix. A `<video>` is silent unless it carries `data-has-audio="true"`. `data-volume` at `0`, or a fade-in longer than the clip, silences a clip that is otherwise timed correctly. `webm` and `gif` outputs differ from `mp4` only in codec and container; a missing track is authored, not encoded.

**The motion stutters or jumps.** A CSS `transform` on an element GSAP also transforms double-applies the start state; use `fromTo` inside the tween. Animating layout properties re-lays-out the page every frame. A move whose distance is too large for its duration reads as a cut.

**The picture is soft or grey.** `quality: draft` bands gradients and crushes detail; re-render at `looks` or `delivery`. A `resolution` preset that does not match the composition's aspect ratio is refused rather than upscaled, so a 1920×1080 composition in a portrait frame is a composition problem, not an option.

## The render is slow

That is normal, not a failure. Work grows with frames × pixels × quality, and a minute of 4K at delivery quality can take many minutes. There is no timeout to hit and no partial result to salvage: wait, then read the message if it fails. Do not re-issue a render because it is slow, and do not reduce quality purely to make it finish unless the user accepts the loss.

## Reporting a failure

If the message names a missing piece (the table at the top), report the missing piece and stop — the user may need to fix a network or a disk. If it names the composition, fix the composition and render again. If it is none of the above, paste the message as it arrived; it is written to be read, and a vague paraphrase loses the only diagnostic there is.
