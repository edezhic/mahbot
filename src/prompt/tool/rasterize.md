Turn a file or a page into an ordinary picture. The input is rendered and saved as a PNG in your workspace's generated/ folder, and the picture is attached to your conversation as a native image in the same step; the output carries an [IMAGE:path] marker, which sends the picture to the user like any generated image.

Accepted inputs:
- SVG markup, plain or gzip-compressed.
- One page of a PDF, named with `page` (1-based).
- A local HTML file inside your personal workspace, rendered as a page.
- An http(s) address, rendered as a page.

The picture keeps the source's aspect ratio, its long side capped at 2048 px.

