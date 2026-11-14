"""Make a 30-second synthetic test song (140 BPM) plus a timing.yaml, so the
engine can be developed without the real songs. Not music; just drums, an 808,
a drone and a "voice" tone where the words are.

  python tools/make_test_song.py songs/test-glen
"""
import os
import sys

import numpy as np
import soundfile as sf
import yaml

SR = 44100
BPM = 140
BEAT = 60 / BPM
BAR = 4 * BEAT
BARS = 18  # 4 intro + 8 chorus + 6 verse


def main(out):
    os.makedirs(out, exist_ok=True)
    dur = BARS * BAR + 0.5
    t = np.arange(int(dur * SR)) / SR
    y = 0.08 * np.sin(2 * np.pi * 110 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.1 * t))  # drone
    g = np.random.default_rng(7)

    def add(sig, at):
        i = int(at * SR)
        j = min(len(y), i + len(sig))
        y[i:j] += sig[: j - i]

    def kick():
        n = int(0.35 * SR)
        tt = np.arange(n) / SR
        f = 45 + 90 * np.exp(-tt * 30)
        return 0.9 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 6)

    def snare():
        n = int(0.2 * SR)
        return 0.35 * g.normal(0, 1, n) * np.exp(-np.arange(n) / SR * 18)

    def hat():
        n = int(0.05 * SR)
        return 0.08 * g.normal(0, 1, n) * np.exp(-np.arange(n) / SR * 80)

    for b in range(BARS):
        t0 = b * BAR
        drums_on = b >= 4  # intro is drone + spoken word only
        for k in range(4):
            if drums_on and k in (0, 2):
                add(kick(), t0 + k * BEAT)
            if drums_on and k in (1, 3):
                add(snare(), t0 + k * BEAT)
            if drums_on:
                add(hat(), t0 + k * BEAT)
                add(hat(), t0 + (k + 0.5) * BEAT)

    words_out = []

    vox = np.zeros_like(y)

    def voice(text_words, start_bar, beats_each=1.0, f0=196):
        tt0 = start_bar * BAR
        ws = []
        for k, w in enumerate(text_words):
            at = tt0 + k * beats_each * BEAT
            n = int(beats_each * BEAT * 0.8 * SR)
            tt = np.arange(n) / SR
            sig = 0.12 * np.sign(np.sin(2 * np.pi * f0 * tt)) * np.exp(-tt * 3)
            add(sig, at)
            i = int(at * SR)
            vox[i:i + len(sig)] += sig[: len(vox) - i]
            ws.append([round(at, 3), w])
        return ws

    lines = []
    intro = "Raise a dram, gentlemen. To the glen my great-great-grandfather emptied.".split()
    lines.append({"section": "intro", "text": " ".join(intro[:4]), "words": voice(intro[:4], 0.25, 1.5, 150)})
    lines.append({"section": "intro", "text": " ".join(intro[4:]), "words": voice(intro[4:], 2.0, 1.0, 150)})
    ch = [("I own the glen, own the loch", 4), ("Own the pier and the parish clock", 6),
          ("You pay for parking at your granny's hill", 8), ("Sell you back the dram from your granddad's still", 10)]
    for text, bar in ch:
        ws = text.split()
        lines.append({"section": "chorus-1", "text": text, "words": voice(ws, bar, 7 / len(ws) * 1.0, 220)})
    v = [("Great-great-granddad cleared the glen for sheep", 12), ("Burned the roofs so the tenants couldnae sleep", 14),
         ("Put 'em on a boat with a Bible and a cough", 16)]
    for text, bar in v:
        ws = text.split()
        lines.append({"section": "verse-1", "text": text, "words": voice(ws, bar, 7 / len(ws), 180)})

    sf.write(os.path.join(out, "master.wav"), np.clip(y, -1, 1).astype(np.float32), SR)
    sf.write(os.path.join(out, "vocals.wav"), np.clip(vox, -1, 1).astype(np.float32), SR)
    with open(os.path.join(out, "lyrics.txt"), "w") as fh:
        for sec in ("intro", "chorus-1", "verse-1"):
            fh.write(f"[{sec}]\n")
            fh.writelines(ln["text"] + "\n" for ln in lines if ln["section"] == sec)
    timing = {
        "note": "synthetic test song; real songs get this file from the alignment step",
        "sections": [{"name": "intro", "start_bar": 0, "bars": 4},
                     {"name": "chorus-1", "start_bar": 4, "bars": 8},
                     {"name": "verse-1", "start_bar": 12, "bars": 6}],
        "lines": lines,
    }
    with open(os.path.join(out, "timing.yaml"), "w") as fh:
        yaml.safe_dump(timing, fh, sort_keys=False, allow_unicode=True, width=200)
    # exact beat grid for the synthetic song (real songs: beat_this)
    beats = [round(k * BEAT, 4) for k in range(int(dur / BEAT))]
    import json
    json.dump({"tempo": BPM, "beats": beats, "downbeats": beats[::4]}, open(os.path.join(out, "beats.json"), "w"))
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
