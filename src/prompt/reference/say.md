# say — reference

## Arguments

- `text` — the phrase to speak, in the language the user speaks. At most 100 characters, counted after trimming the surrounding whitespace — characters, not bytes.

## Limits

A longer phrase is not spoken at all: say it in one short phrase instead, because long text is not read aloud. An empty phrase is refused too.

## Playback

The call returns once the phrase has been rendered; the speech then plays in the background, and a newer phrase stops the one being spoken. Nothing is spoken while the user has speech output turned off, or while nobody is at the computer.

## Result

`Spoken: the phrase plays now.`, or `Nothing was spoken: <reason>` — a phrase the engine refused, a phrase too long, speech off, or nobody at the computer each say which.
