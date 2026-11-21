Create and edit documents: write a new .docx/.xlsx/.pptx/.pdf, fill a user's own sample, edit an existing Office file, and change a PDF's pages. Pass `action` and that action's own arguments — every action is a fixed operation over the values you pass. Every `path` is {{input_location}}.

**The input is never modified.** Each call writes a new file into the workspace's `generated/` directory, whose name the tool owns — `file_name` is only a base name. Name every produced file in your reply with `[FILE:path]`; the media sweep removes generated files that no session mentions. An input may be up to {{input_mb}} MB.

The answer reports what it wrote and names every loss it found.
