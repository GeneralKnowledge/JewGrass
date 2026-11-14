"""megamix.py CONFIG_JSON: beat-matched album megamix from the masters in ~/band/final.

Runs in ~/band/tools/fx/.venv (CPU) as `flock ~/band/.gpu.lock ~/band/heavy.sh python megamix.py megamix-config.json`.
Loads one track at a time.

Config (JSON):
  {"tracks": [{"title": "...", "file": "/abs/final/NN-slug.wav", "analysis": "/abs/analysis.json",
               "bpm": 140, "skit": false, "avoid": [[s, e], ...]}, ...],      # avoid = flagged silences
   "first": "parcel-of-rogues", "last": ["gavs-song", "we-bought-the-hill"],
   "out_dir": "/home/red/band/final/megamix", "target_min_s": 540, "target_max_s": 720}

Steps:
  1. Excerpts: for each track, the best window of N real 4/4 bars at the song's own tempo (16 by default, 8 for
     skits), starting and ending on detected downbeats. A window scores on loudness plus "vocal activity" (the share of 1-4 kHz energy, a stand-in for vocals
     when no stem exists). Windows overlapping an `avoid` interval are skipped. N grows (16, 20, 24, 28, 32) until
     the mix reaches target_min_s.
  2. Order: first and last fixed; the rest sorted by tempo, folded into 100-200 BPM so half/double time match; then
     neighbours with the same key are swapped apart where possible.
  3. Transitions: 4-bar crossfades on downbeats. If the tempos are within 8% (half/double time counting as equal), the incoming excerpt is
     time-stretched (Rubber Band via pedalboard) to the outgoing tempo, and the outgoing track is high-passed during
     the fade (bass swap). Otherwise: a hard cut on the downbeat, with a 1-bar low-pass sweep on the outgoing bar and
     a reverse-reverb swell into the incoming downbeat.
  4. Master: produce.py's gentle master (no more than 3 dB of limiting, -14 LUFS / -1 dBTP; stays quieter if needed).
Outputs: natural-capital-megamix.wav (24-bit 48 kHz), megamix.json (order, reasons, windows, transitions, levels),
megamix-cues.txt.
"""
import json
import os
import sys

import numpy as np
import soundfile as sf
from pedalboard import HighpassFilter, LowpassFilter, Pedalboard, Reverb, time_stretch
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import produce  # noqa: E402  (load, db, lin, master, checks)

SR = produce.SR


def fold(bpm):
    t = float(bpm)
    while t < 100:
        t *= 2
    while t >= 200:
        t /= 2
    return t


def bar_seconds(a, bpm_real):
    """Real 4/4 bar length from the song's own tempo, and the detected downbeat spacing.
    beat_this sometimes counts half or double time, so windows are measured in real bars, not detected ones."""
    d = np.diff(a["downbeats"])
    real = 240.0 / float(bpm_real)
    det = float(np.median(d)) if len(d) else real
    return real, det


def octave_ratio(t_out, t_in):
    """Tempo ratio out/in, with half/double time treated as the same tempo (result within ~0.71..1.41)."""
    r = float(t_out) / float(t_in)
    while r > 1.414:
        r /= 2
    while r < 0.707:
        r *= 2
    return r


def best_window(x, a, n_bars, bpm, avoid):
    downs = np.array(a["downbeats"])
    bar_s, det = bar_seconds(a, bpm)
    k = max(1, int(round(n_bars * bar_s / det)))  # detected-downbeat steps covering n_bars real bars
    mono = x.mean(axis=1)
    sos = butter(4, [1000, 4000], "bandpass", fs=SR, output="sos")
    best = None
    for i in range(0, len(downs) - k):
        s, e = downs[i], downs[i + k]
        if e * SR > len(mono) or any(not (e <= a0 or s >= b0) for a0, b0 in avoid):
            continue
        seg = mono[int(s * SR):int(e * SR)]
        if len(seg) < SR:
            continue
        loud = produce.db(seg)
        voc = produce.db(sosfiltfilt(sos, seg[:: 2]))  # decimated: cheap band-energy estimate
        score = loud + 0.5 * (voc - loud)
        if best is None or score > best[0]:
            best = (score, float(s), float(e))
    if best is None:  # every window hit a flagged silence: take the loudest regardless
        return best_window(x, a, n_bars, bpm, [])
    return {"start": round(best[1], 3), "end": round(best[2], 3), "score": round(best[0], 2), "bar_s": round(bar_s, 4)}


def order_tracks(tracks, first, last):
    by = {t["slug"]: t for t in tracks}
    head = [by[first]] if first in by else []
    tail_slug = next((s for s in last if s in by and s != first), None)
    tail = [by[tail_slug]] if tail_slug else []
    mid = sorted([t for t in tracks if t not in head + tail], key=lambda t: fold(t["bpm"]))
    # keep same-key neighbours apart where a swap with the next-but-one fixes it
    seq = head + mid + tail
    for _ in range(3):
        for i in range(1, len(seq) - 2):
            if seq[i]["key"] == seq[i + 1]["key"] and seq[i + 2]["key"] != seq[i]["key"]:
                seq[i + 1], seq[i + 2] = seq[i + 2], seq[i + 1]
    reasons = []
    for i, t in enumerate(seq):
        r = f"{fold(t['bpm']):.0f} BPM (folded), {t['key']}"
        if i == 0:
            r = "cold open (fixed); " + r
        elif i == len(seq) - 1 and tail:
            r = "closer (fixed); " + r
        reasons.append(r)
    return seq, reasons


def fade_curves(n):
    t = np.linspace(0, np.pi / 2, n, dtype=np.float32)
    return np.cos(t), np.sin(t)  # equal-power


def build(cfg):
    tracks = cfg["tracks"]
    for t in tracks:
        a = json.load(open(t["analysis"]))
        t["key"] = a.get("key_estimate", "?")
        t["_a"] = a
    seq, reasons = order_tracks(tracks, cfg.get("first", ""), cfg.get("last", []))
    for n_bars in (16, 20, 24, 28, 32):
        est, wins = 0.0, []
        for t in seq:
            x = produce.load(t["file"])
            nb = 8 if t.get("skit") else n_bars
            w = best_window(x, t["_a"], nb, t["bpm"], t.get("avoid", []))
            w["bars"] = nb
            wins.append(w)
            del x
        xf = [4 * w["bar_s"] for w in wins[:-1]]
        est = sum(w["end"] - w["start"] for w in wins) - sum(xf)
        if est >= cfg.get("target_min_s", 540) or n_bars == 32:
            break
    out = np.zeros((0, 2), np.float32)
    cues, trans = [], []
    prev_tempo = None
    for i, (t, w) in enumerate(zip(seq, wins)):
        x = produce.load(t["file"])
        seg = x[int(w["start"] * SR):int(w["end"] * SR)].copy()
        del x
        tempo = float(t["bpm"])
        if i == 0:
            cues.append((0.0, t["title"]))
            out = seg
            prev_tempo, prev_bar = tempo, w["bar_s"]
            continue
        ratio = octave_ratio(prev_tempo, tempo)
        xf_n = int(4 * prev_bar * SR)
        if abs(ratio - 1) <= 0.08 and len(out) > xf_n and len(seg) > xf_n:
            # stretch the incoming crossfade region (4 of its bars) to the outgoing tempo
            in_bars = int(4 * w["bar_s"] * SR)
            head = time_stretch(seg[:in_bars].T.copy(), SR, ratio).T
            head = head[:xf_n] if len(head) >= xf_n else np.pad(head, ((0, xf_n - len(head)), (0, 0)))
            rest = seg[in_bars:]
            fo, fi = fade_curves(xf_n)
            tail_out = out[-xf_n:]
            hp = Pedalboard([HighpassFilter(cutoff_frequency_hz=180.0)])(tail_out.T.copy(), SR).T
            # bass swap: outgoing loses its lows over the first half of the fade
            mix_hp = np.clip(np.linspace(0, 2, xf_n, dtype=np.float32), 0, 1)[:, None]
            tail_out = tail_out * (1 - mix_hp) + hp * mix_hp
            inc_hp = Pedalboard([HighpassFilter(cutoff_frequency_hz=180.0)])(head.T.copy(), SR).T
            inc_mix = np.clip(np.linspace(-1, 1, xf_n, dtype=np.float32), 0, 1)[:, None]  # lows come in at half way
            head = inc_hp * (1 - inc_mix) + head * inc_mix
            blend = tail_out * fo[:, None] + head * fi[:, None]
            start = len(out) - xf_n
            out = np.concatenate([out[:-xf_n], blend, rest])
            trans.append({"from": seq[i - 1]["slug"], "to": t["slug"], "type": "beatmatched crossfade",
                          "stretch": round(ratio, 4), "bars": 4})
        else:
            # hard cut on the downbeat: 1-bar low-pass sweep out, reverse-reverb swell in
            bar_n = int(prev_bar * SR)
            last = out[-bar_n:]
            steps = 8
            swept = np.concatenate([Pedalboard([LowpassFilter(cutoff_frequency_hz=float(c))])(
                chunk.T.copy(), SR).T for c, chunk in zip(np.geomspace(16000, 400, steps), np.array_split(last, steps))])
            out = np.concatenate([out[:-bar_n], swept])
            beat = seg[: int(w["bar_s"] / 4 * SR)]
            rev = Pedalboard([Reverb(room_size=0.9, wet_level=1.0, dry_level=0.0, width=1.0)])(
                np.concatenate([beat[::-1], np.zeros((SR * 2, 2), np.float32)]).T.copy(), SR).T[::-1]
            swell = rev[-int(prev_bar * SR):] * 0.35
            out[-len(swell):] += swell
            start = len(out)
            out = np.concatenate([out, seg])
            trans.append({"from": seq[i - 1]["slug"], "to": t["slug"], "type": "hard cut + filter sweep + reverse reverb",
                          "tempo_gap": round(ratio, 3)})
        cues.append((start / SR, t["title"]))
        prev_tempo, prev_bar = tempo, w["bar_s"]
    # short fade in/out of the whole mix
    fi_n, fo_n = int(0.02 * SR), int(3 * SR)
    out[:fi_n] *= np.linspace(0, 1, fi_n, dtype=np.float32)[:, None]
    out[-fo_n:] *= np.linspace(1, 0, fo_n, dtype=np.float32)[:, None]
    return seq, reasons, wins, trans, cues, out


def main():
    cfg = json.load(open(sys.argv[1]))
    os.makedirs(cfg["out_dir"], exist_ok=True)
    seq, reasons, wins, trans, cues, mix = build(cfg)
    rep = {}
    mastered = produce.master(mix, dict(produce.DEFAULTS), rep)
    wav = os.path.join(cfg["out_dir"], "natural-capital-megamix.wav")
    sf.write(wav, mastered, SR, subtype="PCM_24")
    checks = produce.checks(mastered)
    with open(os.path.join(cfg["out_dir"], "megamix-cues.txt"), "w") as f:
        for s, title in cues:
            f.write(f"{int(s // 60):02d}:{s % 60:05.2f}  {title}\n")
    json.dump({"order": [{"slug": t["slug"], "title": t["title"], "why": r, "window": w}
                         for t, r, w in zip(seq, reasons, wins)],
               "transitions": trans, "cues": [[round(s, 2), t] for s, t in cues],
               "length_s": round(len(mastered) / SR, 1), "master": rep["master"], "checks": checks},
              open(os.path.join(cfg["out_dir"], "megamix.json"), "w"), indent=1)
    print(json.dumps({"length_s": round(len(mastered) / SR, 1), "master": rep["master"],
                      "hard_cuts": [f"{t['from']}->{t['to']}" for t in trans if t["type"].startswith("hard")]}))


if __name__ == "__main__":
    main()
