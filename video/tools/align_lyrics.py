"""Lyric timing without a speech model: align our own lyrics to the vocal stem.

  python tools/align_lyrics.py VOCALS.flac LYRICS.txt BEATS.json OUT_timing.yaml [--duration S]

We already know every word (we wrote them), so we only need *when*. Steps:
  1. find syllable-like onsets in the vocal stem (spectral flux, sensitive);
  2. split the lyrics into syllables (vowel groups), remembering where each
     word and line starts;
  3. dynamic programming: give each syllable its own onset, in order, so that
     syllables inside a line sit close together, lines may start after any
     gap, and long silences inside a line are expensive;
  4. word time = the onset of its first syllable; sections start on the bar
     line at or before their first word.

It's not a forced aligner; a line off by a beat can be fixed by hand in the
timing file (`words: [[time, word], ...]`). faster-whisper word timestamps can
replace step 1-3 later where a model is available.

LYRICS.txt uses our section tags ([Intro], [Verse 1], [Chorus], ...);
performance notes in a second bracket line ([Spoken word, ...]) are ignored.
"""
import argparse
import json
import re

import numpy as np
import librosa
import yaml

SR = 22050
HOP = 220  # 10 ms


def syllables(word):
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 1
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and n > 1 and not w.endswith(("le", "ee", "ie")):
        n -= 1
    return max(1, n)


def parse_lyrics(path):
    sections, cur = [], None
    for raw in open(path, encoding="utf-8"):
        line = raw.strip()
        if not line or line.startswith("```"):
            continue
        m = re.fullmatch(r"\[([^\]]+)\]", line)
        if m:
            tag = m.group(1)
            # a section tag is short; a performance note has commas or many words
            if "," in tag or len(tag.split()) > 3:
                continue
            name = tag.lower().replace(" ", "-")
            cur = {"name": name, "lines": []}
            sections.append(cur)
            continue
        if cur is None:
            cur = {"name": "intro", "lines": []}
            sections.append(cur)
        cur["lines"].append(line)
    # unique names: chorus, chorus-2, chorus-3 ...
    seen = {}
    for s in sections:
        base = s["name"]
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            s["name"] = f"{base}-{seen[base]}"
    return sections


def vocal_onsets(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP)[0]
    voiced = rms > max(np.percentile(rms, 60) * 0.25, 1e-4)
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP, lag=2, max_size=3)
    on = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=HOP, units="frames",
                                    delta=0.02, wait=6, pre_max=3, post_max=3)
    on = [f for f in on if f < len(voiced) and voiced[f]]
    times = librosa.frames_to_time(np.array(on), sr=SR, hop_length=HOP)
    return np.array(times), len(y) / SR


def align(syl_line_start, onsets, gap=0.16):
    """syl_line_start[k] is True when syllable k begins a lyric line.
    Returns the onset index for every syllable (monotonic)."""
    n, m = len(syl_line_start), len(onsets)
    if m < n:
        raise SystemExit(f"only {m} vocal onsets for {n} syllables; lower the onset threshold")
    INF = 1e18
    W = 8  # how many onsets one step may skip
    cost = np.full((n, m), INF)
    back = np.zeros((n, m), np.int32)
    cost[0, : m - n + 1] = onsets[: m - n + 1] * 0.001  # prefer starting early, barely
    for i in range(1, n):
        lo, hi = i, m - (n - i) + 1
        for j in range(lo, hi):
            j0 = max(i - 1, j - W)
            prev = cost[i - 1, j0:j]
            dt = onsets[j] - onsets[j0:j]
            skips = (j - np.arange(j0, j)) - 1
            if syl_line_start[i]:
                pen = 0.15 * skips  # a new line may start after any pause
                # ... but a long gap is likelier between lines than inside one
                pen -= np.clip(dt - 0.4, 0, 1.0) * 0.3
            else:
                pen = 0.5 * skips + 4.0 * np.clip(dt - gap * 2.2, 0, None) ** 2
            tot = prev + pen
            k = int(np.argmin(tot))
            cost[i, j] = tot[k]
            back[i, j] = j0 + k
        # free long gaps across lines when the previous line ended: handled by W on purpose
    j = int(np.argmin(cost[n - 1]))
    path = [j]
    for i in range(n - 1, 0, -1):
        j = back[i, j]
        path.append(j)
    return path[::-1]


def bars_from_beats(beats_json, duration):
    b = json.load(open(beats_json))
    beats = np.array(b["beats"], float)
    downs = np.array(b.get("downbeats", beats[::4]), float)
    bar = np.median(np.diff(downs)) if len(downs) > 1 else 4 * np.median(np.diff(beats))
    beat = np.median(np.diff(beats))
    if 60 / beat < 100:  # half-time tracking: split every beat in two
        beats = np.sort(np.concatenate([beats, beats[:-1] + np.diff(beats) / 2]))
        beat /= 2
    if bar > 3.2 * beat * 1.5:  # bars tracked as two bars: split them
        downs = np.sort(np.concatenate([downs, downs[:-1] + np.diff(downs) / 2]))
    # fill holes (e.g. a break where the tracker lost the downbeats)
    period = np.median(np.diff(downs))
    filled = [downs[0]]
    for d in downs[1:]:
        while d - filled[-1] > period * 1.5:
            filled.append(filled[-1] + period)
        filled.append(d)
    downs = np.array(filled)
    # extend the bar grid back to 0 and forward to the end
    while downs[0] > period * 0.5:
        downs = np.insert(downs, 0, downs[0] - period)
    while downs[-1] + period < duration + period:
        downs = np.append(downs, downs[-1] + period)
    tempo = 60 / beat
    return beats, downs, tempo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vocals")
    ap.add_argument("lyrics")
    ap.add_argument("beats")
    ap.add_argument("out")
    ap.add_argument("--beats-out", help="write the corrected beat grid here (for engine.analysis --beats)")
    a = ap.parse_args()

    secs = parse_lyrics(a.lyrics)
    onsets, duration = vocal_onsets(a.vocals)
    beats, downs, tempo = bars_from_beats(a.beats, duration)

    syl_start, owner = [], []  # owner: (section idx, line idx, word idx)
    for si, s in enumerate(secs):
        for li, line in enumerate(s["lines"]):
            words = line.split()
            for wi, w in enumerate(words):
                for k in range(syllables(w)):
                    syl_start.append(wi == 0 and k == 0)
                    owner.append((si, li, wi, k))
    if len(onsets) < len(syl_start):
        # too few onsets for syllables (e.g. a smooth sung line): align words instead
        print(f"note: {len(onsets)} onsets < {len(syl_start)} syllables, aligning by word")
        keep = [k for k, o in enumerate(owner) if o[3] == 0]
        owner = [owner[k] for k in keep]
        syl_start = [syl_start[k] for k in keep]
    path = align(syl_start, onsets)

    times = {}
    for (si, li, wi, k), j in zip(owner, path):
        if k == 0:
            times[(si, li, wi)] = float(onsets[j])

    lines, sec_out = [], []
    first_word = []
    for si, s in enumerate(secs):
        t0 = min((times[(si, li, 0)] for li in range(len(s["lines"]))), default=None)
        first_word.append(t0)
        for li, line in enumerate(s["lines"]):
            words = line.split()
            lines.append({"section": s["name"], "text": line,
                          "words": [[round(times[(si, li, wi)], 3), w] for wi, w in enumerate(words)]})
    # sections start on the bar at or before their first word; the first starts at bar 0
    starts = []
    for si, t0 in enumerate(first_word):
        if si == 0:
            starts.append(0)
            continue
        b = int(np.searchsorted(downs, t0 + 0.25) - 1)  # a pickup just before the bar counts as that bar
        starts.append(max(b, starts[-1] + 1))
    nbars = int(np.searchsorted(downs, duration))
    for si, s in enumerate(secs):
        end = starts[si + 1] if si + 1 < len(secs) else nbars
        sec_out.append({"name": s["name"], "start_bar": starts[si], "bars": end - starts[si]})

    with open(a.out, "w") as fh:
        yaml.safe_dump({"note": "aligned by tools/align_lyrics.py; fix any word by hand",
                        "tempo": round(float(tempo), 2), "sections": sec_out, "lines": lines},
                       fh, sort_keys=False, allow_unicode=True, width=200)
    if a.beats_out:
        json.dump({"tempo": float(tempo), "beats": [round(float(x), 4) for x in beats],
                   "downbeats": [round(float(x), 4) for x in downs if x >= -0.01]}, open(a.beats_out, "w"))
    print(f"{len(onsets)} onsets, {len(syl_start)} syllables, {len(lines)} lines, {len(secs)} sections")
    for s in sec_out:
        print(f"  {s['name']:<12} bar {s['start_bar']:>3} ({s['bars']} bars)")


if __name__ == "__main__":
    main()
