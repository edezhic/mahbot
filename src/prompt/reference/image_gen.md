# image_gen — reference

## Arguments

- `prompt` — one text description shared by every generation in the call.
- `generations` — one entry per picture, at most 10. Each entry takes its own `images`, `aspect_ratio` and `size`; a setting left unset behaves exactly as a single picture with it unset and never borrows a sibling's value. A generation's position in the list, counted from 1, is how the result names it.

## Reference images

- `images` — paths to this generation's reference images for image-to-image; none makes it text-to-image.
- A relative path resolves against the workspace root.
- References over 1.5 MB are compressed (downscaled and re-encoded as JPEG), and several may be compressed further to keep the request within the ~2 MB body budget.
- At most 16 references per generation, and the total reference material must fit the input cap.

## Aspect ratio and size

- `aspect_ratio` — e.g. `16:9`, `1:1`, `4:3`. Unset keeps the model's default, or the first reference's own ratio.
- `size` — a tier (e.g. `1K`, `2K`) or explicit pixels (e.g. `1024x1024`). A model that cannot express the value is not sent the request; unset keeps the model's own default.

## Models and fallback

The account's chosen model is tried first; when it cannot produce the picture or take the request at all — gone from the catalog, cannot make pictures, cannot take that many references, or cannot express the requested size — the account's other models are tried in turn, and the request succeeds once one serves it. When a fallback served it, the result names the model that produced the picture and what happened with the account's own model (what it answered, or that it was never sent and why); tell the user both when you send the picture.

## The result

The attached copy may be smaller than the picture itself; when it is, the attachment note states that copy's size as a separate fact, and the result's size is the picture's own — that is the resolution to tell the user about.

The result reports one line per generation, in position order: a produced picture's marker, the picture's own pixel size in parentheses after it, and — when a fallback model served it — the note about that model, or one line saying why that generation produced nothing. The marker is `[IMAGE:path]` exactly; the size follows it as a separate parenthesised fact.

A call that produced nothing at all names every position in its message. Tell the user what the result says for each picture, and for a generation that produced nothing, which position it was and why.
