#!/usr/bin/env python3
"""Did a generated take follow its Song Design's key and chords?

    python3 tools/daw/check_take.py instrumental.mp3 docs/album/design/gavs-song.yaml

Estimates the key (Krumhansl profiles on the chroma of the whole track) and the
chord heard in every 1.5 s window (major/minor triad templates), then reports
how much of the take uses the chords the design asked for. CPU only, seconds.
Rough by nature: it can't see sevenths or slash chords, and a window that
straddles a change picks whichever chord is louder.
"""
import collections
import os
import sys

import librosa
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(__file__))
import from_design  # noqa: E402

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
FLAT = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}


def norm(sym):
    for f, s in FLAT.items():
        if sym.startswith(f):
            return s + sym[2:]
    return sym


def main(audio, design_path, window=1.5):
    d = yaml.safe_load(open(design_path))
    want = set()
    for s in d["sections"]:
        loop, tail = from_design.section_chords(d, s["name"], d["render"]["key"])
        want |= {norm(c.rstrip("7").replace("maj", "")) for c in loop + tail}
    y, sr = librosa.load(audio, sr=22050)
    C = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=2048)
    prof = C.mean(1)
    keys = [(np.corrcoef(np.roll(MAJ, k), prof)[0, 1], f"{NAMES[k]} major") for k in range(12)] + \
           [(np.corrcoef(np.roll(MIN, k), prof)[0, 1], f"{NAMES[k]} minor") for k in range(12)]
    key = max(keys)
    temps = []
    for r in range(12):
        for q, iv in (("", [0, 4, 7]), ("m", [0, 3, 7])):
            v = np.zeros(12)
            v[[(r + i) % 12 for i in iv]] = 1
            temps.append((NAMES[r] + q, v / np.linalg.norm(v)))
    win = max(1, int(window * sr / 2048))
    cnt = collections.Counter()
    for i in range(0, C.shape[1] - win, win):
        c = C[:, i:i + win].mean(1)
        c = c / (np.linalg.norm(c) + 1e-9)
        cnt[max(temps, key=lambda t: t[1] @ c)[0]] += 1
    tot = sum(cnt.values())
    print(f"design key: {d['render']['key']}   heard: {key[1]} (fit {key[0]:.2f})")
    print(f"design chords: {' '.join(sorted(want))}")
    print("heard chords (% of time): " + ", ".join(f"{k} {100 * v / tot:.0f}" for k, v in cnt.most_common(8)))
    print(f"time on design chords: {100 * sum(v for k, v in cnt.items() if k in want) / tot:.0f}%")
    missing = [c for c in sorted(want) if 100 * cnt.get(c, 0) / tot < 2]
    if missing:
        print(f"design chords barely heard (<2%): {' '.join(missing)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
