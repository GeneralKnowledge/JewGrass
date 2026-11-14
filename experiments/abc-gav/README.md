# A/B/C test: does a composed reference make Gav's Song better?

Question: if ACE-Step gets our arrangement as audio (cover mode), does the take follow the designed chords, and does it sound better than the text-only take?

| Clip | What |
|---|---|
| A | take 7202 (design-v1 winner, text only), first 47.4 s |
| B | ACE-Step `cover` from `reference-composed.mp3`, light (`audio_cover_strength` 0.3) |
| C | same, strong (`audio_cover_strength` 0.6) |

Same caption (`caption.txt`), lyrics (`lyrics-intro-verse1.txt`: intro + verse 1), key D major, 6/8, 76 bpm, seed 7202, 48 s.

`reference-composed.mp3`: intro 4 bars on D, verse 1 = Bm Bm G G D D A A Bm Bm G G D D G A (2 bars per lyric line), fingerpicked guitar from the start, upright bass from halfway through verse 1. Made by tools/daw/from_design.py and rendered with the GM SoundFont. Chord check on it: D 38%, Bm 16%, G 16%, A 16%.

Judged blind (shuffled), plus `tools/daw/check_take.py` on each clip.
