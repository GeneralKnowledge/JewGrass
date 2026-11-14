"""Lyric timing from speech-recognition word timestamps (faster-whisper).

  python tools/whisper_timing.py WHISPER_WORDS.json LYRICS.txt BEATS.json OUT_timing.yaml --beats-out grid.json

The recogniser hears most words but mishears some ("glen" -> "glenn"), drops
some (doubled chorus lines) and adds ad-libs. We know the real words, so we
align the two word lists (Needleman-Wunsch with fuzzy word similarity), take
the recogniser's time for every confident match, and interpolate the rest
between their matched neighbours. Section starts go on the bar line at or
before their first word, as in align_lyrics.py.

WHISPER_WORDS.json: a list of segments {start, end, text, words: [{start, end,
word, probability}]} (or {"segments": [...]}).
"""
import argparse
import json
import os
import re
import sys
from difflib import SequenceMatcher

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from align_lyrics import parse_lyrics, bars_from_beats  # noqa: E402


def norm(w):
    w = w.lower().replace("’", "'")
    w = re.sub(r"[^a-z0-9']", "", w)
    return w.strip("'")


def sim(a, b):
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def load_whisper(path):
    d = json.load(open(path))
    segs = d["segments"] if isinstance(d, dict) else d
    out = []
    for s in segs:
        for w in s.get("words") or []:
            n = norm(w["word"])
            if n:
                out.append((float(w["start"]), float(w["end"]), n, float(w.get("probability", 1.0))))
    out.sort(key=lambda x: x[0])
    return out


def align_words(lyr, heard, gap=-0.6, min_sim=0.6):
    """Global alignment; returns {lyric index: heard index} for confident matches."""
    n, m = len(lyr), len(heard)
    S = np.zeros((n + 1, m + 1), np.float32)
    B = np.zeros((n + 1, m + 1), np.int8)  # 0 diag, 1 up (skip lyric), 2 left (skip heard)
    S[1:, 0] = gap * np.arange(1, n + 1)
    S[0, 1:] = gap * 0.5 * np.arange(1, m + 1)  # ad-libs and repeats are cheaper to skip
    B[1:, 0], B[0, 1:] = 1, 2
    for i in range(1, n + 1):
        a = lyr[i - 1]
        for j in range(1, m + 1):
            s = sim(a, heard[j - 1][2])
            diag = S[i - 1, j - 1] + (2.0 * s if s >= min_sim else -1.0)
            up = S[i - 1, j] + gap
            left = S[i, j - 1] + gap * 0.5
            if diag >= up and diag >= left:
                S[i, j], B[i, j] = diag, 0
            elif up >= left:
                S[i, j], B[i, j] = up, 1
            else:
                S[i, j], B[i, j] = left, 2
    match = {}
    i, j = n, m
    while i > 0 and j > 0:
        if B[i, j] == 0:
            if sim(lyr[i - 1], heard[j - 1][2]) >= min_sim:
                match[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif B[i, j] == 1:
            i -= 1
        else:
            j -= 1
    return match


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("whisper")
    ap.add_argument("lyrics")
    ap.add_argument("beats")
    ap.add_argument("out")
    ap.add_argument("--duration", type=float, help="song length in seconds")
    ap.add_argument("--audio", help="read the song length from this file instead")
    ap.add_argument("--beats-out")
    a = ap.parse_args()

    secs = parse_lyrics(a.lyrics)
    heard = load_whisper(a.whisper)
    owner, lyr = [], []
    for si, s in enumerate(secs):
        for li, line in enumerate(s["lines"]):
            for wi, w in enumerate(line.split()):
                owner.append((si, li, wi))
                lyr.append(norm(w))
    match = align_words(lyr, heard)

    # times: matched words take the heard start; others interpolate by word index
    t = np.full(len(lyr), np.nan)
    for k, j in match.items():
        t[k] = heard[j][0]
    # enforce order (a stray match can't jump backwards)
    last = -1.0
    for k in range(len(t)):
        if not np.isnan(t[k]):
            if t[k] < last:
                t[k] = np.nan
            else:
                last = t[k]
    known = np.where(~np.isnan(t))[0]
    if len(known) < 2:
        raise SystemExit("too few words matched; is this the right vocal / lyrics?")
    lineof = [(o[0], o[1]) for o in owner]
    full = np.interp(np.arange(len(t)), known, t[known])
    for k0, k1 in zip(known[:-1], known[1:]):
        if k1 - k0 < 2:
            continue
        run = range(k0 + 1, k1)
        # a missed word belongs with the sung words of its own line, not
        # halfway across the pause between two lines
        for k in run:
            if lineof[k] == lineof[k1]:
                full[k] = max(t[k1] - 0.28 * (k1 - k), t[k0] + 0.05 * (k - k0))
            elif lineof[k] == lineof[k0]:
                full[k] = min(t[k0] + 0.28 * (k - k0), t[k1] - 0.05 * (k1 - k))
    t = full
    # before the first / after the last match: extend at 0.35 s per word
    t[: known[0]] = t[known[0]] - 0.35 * (known[0] - np.arange(known[0]))
    t[known[-1] + 1:] = t[known[-1]] + 0.35 * (np.arange(known[-1] + 1, len(t)) - known[-1])
    t = np.maximum(t, 0)

    b = json.load(open(a.beats))
    if a.audio:
        import soundfile
        a.duration = soundfile.info(a.audio).duration
    duration = a.duration or max(float(t[-1]) + 10, float(b["beats"][-1]) + 1)
    beats, downs, tempo = bars_from_beats(a.beats, duration)

    times = {o: float(x) for o, x in zip(owner, t)}
    lines, first_word = [], []
    for si, s in enumerate(secs):
        first_word.append(times[(si, 0, 0)] if s["lines"] else None)
        for li, line in enumerate(s["lines"]):
            words = line.split()
            lines.append({"section": s["name"], "text": line,
                          "words": [[round(times[(si, li, wi)], 3), w] for wi, w in enumerate(words)]})
    starts = []
    for si, t0 in enumerate(first_word):
        if si == 0:
            starts.append(0)
            continue
        if t0 is None:
            starts.append(starts[-1] + 1)
            continue
        bb = int(np.searchsorted(downs, t0 + 0.25) - 1)
        starts.append(max(bb, starts[-1] + 1))
    nbars = int(np.searchsorted(downs, duration))
    sec_out = []
    for si, s in enumerate(secs):
        end = starts[si + 1] if si + 1 < len(secs) else nbars
        sec_out.append({"name": s["name"], "start_bar": starts[si], "bars": end - starts[si]})

    with open(a.out, "w") as fh:
        yaml.safe_dump({"note": "word times from faster-whisper, matched to our lyrics by tools/whisper_timing.py; fix any word by hand",
                        "tempo": round(float(tempo), 2), "sections": sec_out, "lines": lines},
                       fh, sort_keys=False, allow_unicode=True, width=200)
    if a.beats_out:
        json.dump({"tempo": float(tempo), "beats": [round(float(x), 4) for x in beats],
                   "downbeats": [round(float(x), 4) for x in downs if x >= -0.01]}, open(a.beats_out, "w"))
    print(f"{len(heard)} heard words, {len(lyr)} lyric words, {len(match)} matched "
          f"({100 * len(match) / max(len(lyr), 1):.0f}%)")
    # per section: how much of it was actually heard (low = interpolated, check by eye)
    for si, s in enumerate(secs):
        ks = [k for k, o in enumerate(owner) if o[0] == si]
        got = sum(1 for k in ks if k in match)
        print(f"  {s['name']:<12} bar {sec_out[si]['start_bar']:>3} ({sec_out[si]['bars']} bars)  "
              f"{got}/{len(ks)} words heard")


if __name__ == "__main__":
    main()
