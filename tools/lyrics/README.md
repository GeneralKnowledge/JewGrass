# Timed lyrics

`time_take.py` works out when each written word of a take is sung and writes it next to the take as
`<take>.lrc`. The take server sends that file to the phone (`tools/player`, the `lrc` field), and Soundscape
lights up the line and the word being sung. Takes without a `.lrc` show their plain lyrics.

How it works:

1. faster-whisper hears the words with a time for each. It runs on the GPU when there is one and on the CPU
   otherwise. A vocal stem works better than the full mix.
2. Whisper's words are matched to the lyrics in the take's sidecar, so the phone shows the written words and
   sections, not Whisper's spelling. Words it missed get times spread across their line.
3. The file is enhanced LRC, `[00:12.30] <00:12.30>One <00:12.61>more <00:12.90>night`. A line Whisper heard
   poorly gets only a line time, so the phone lights the whole line instead of guessing words. A section tag
   goes in at the time of its first line.

| Where | How |
|---|---|
| Kaggle, new takes | Set `TIME_LYRICS = True` in the notebook. After the renders, each take with lyrics is timed on card 0, about half a minute per take |
| Laptop, one take | `flock ~/band/.gpu.lock ~/band/heavy.sh ~/band/tools/asr/.venv/bin/python tools/lyrics/time_take.py TAKE.mp3 [--vocals STEM.wav]` (CPU, about 3 minutes per 4-minute song, see `laptop/tools/asr.md`) |
| Laptop, the picked takes | The same with `--all ~/band --picks ~/band/picks.json` |
| Already have Whisper's words | `--words whisper-words.json` (the output of `laptop/tools/whisper_words.py`) skips hearing them again |

`--all` skips takes that already have a `.lrc`, so a stopped run can just be started again. To redo a take,
delete its `.lrc`. Generated vocals sometimes slur or skip words, so some lines will show only line timing.

Tests: `python3 -m unittest tools/lyrics/test_time_take.py` (from `tools/lyrics`). They need no model.
