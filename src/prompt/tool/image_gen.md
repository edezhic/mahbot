Generate pictures with an AI image model: text-to-image, or image-to-image when a generation carries reference `images`. One call carries one `prompt` shared by up to 10 `generations`; each generation has its own `images`, `aspect_ratio` and `size`, so one call can retouch a set or produce several shapes. An unset setting behaves as if the generation were alone; it never borrows a sibling's value.

Generations are requested together, so a batch takes about as long as its slowest picture. Every picture is attached as a native image and named by its position in the result, with its pixel size; the result carries one `[IMAGE:path]` marker per picture, and a picture reaches the user only when your reply names it. A generation that produced nothing costs its siblings nothing; only a call that produced nothing at all fails.

A reference must be PNG, JPEG or WebP and a local path must be inside your own workspace (an attachment, a previous output, or a file you filed).
