# search_archived_tickets — reference

## Arguments

- `query` — words matched against ticket titles by the board's full-text index.

## Scope

Searches only archived tickets — the `is_archived` subset of completed and cancelled tickets — across the whole installation, every workspace. Live tickets never consume the result limit.

## Result

At most 10 matches, highest relevance first, under a header naming the count. Each line is `[workspace] [phase] id: title`, so `get_ticket` can open the hit wherever it was filed. A miss answers `No archived tickets found matching the query.`

Several parallel calls with differently worded queries often find more than one query does.
