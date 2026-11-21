# web_search — reference

## Arguments

- `query` — a web search query. Exactly one of `query` and `expand` per call; passing both or neither is refused.
- `expand` — the expand ID of a result from an earlier search in this same session. It retrieves that result's full cached text instead of running a new search.
- `pictures` — declared only when a picture backend is configured. It asks for pictures of the results' subjects in addition to the text results.

## The two modes

A `query` call searches the web and answers with short highlights: title, address and an extract per result. Each highlight carries an expand ID; expanding it returns the cached page text whole, which is what you read when a highlight is not enough.

An `expand` call runs no search. Expand IDs are session-scoped — they work only for a search that ran earlier in this same conversation — and they are integers, not addresses.

## Pictures

With `pictures: true` the answer also lists pictures of the results' subjects, each with its address and hosting page — or `No usable pictures for this query.`, or `Pictures unavailable for this query.` when the backend could not look. Pictures never take expand IDs: only text results are cached.

## Result

`Search results for: <query>` and one entry per result — `[title](url)`, an extract when there is one, and its `expand` id. A search with no hits answers `No results found for: <query>`. With `pictures`, a picture section follows. The whole answer is delivered untruncated.
