"""Stage 1: musical analysis -> analysis.json (CPU, deterministic).

Reads the master and, when present, four stems (drums, bass, vocals, other)
from Demucs. Without stems it falls back to rough frequency bands of the mix,
which is good enough for tests but not for a real video.

Output, all curves sampled at the video frame rate and normalised to 0..1:
  duration, fps, tempo, beats[], downbeats[], bars[]
  env.{bass,drums,vocals,other}   smoothed energy per element
  hits.{bass,drums}               onset strength (sharp, for impacts)
  macro.{energy,tension,stillness}  slow whole-song curves

Usage: python -m engine.analysis master.wav out/analysis.json [--stems DIR] [--fps 30]
"""
import argparse
import json
import os

import numpy as np
import librosa
import scipy.signal as sps

SR = 22050
HOP = 512


def _smooth(x, fps, attack_s, release_s):
    """One-pole attack/release follower (deterministic)."""
    a = np.exp(-1.0 / max(attack_s * fps, 1e-6))
    r = np.exp(-1.0 / max(release_s * fps, 1e-6))
    y = np.zeros_like(x)
    prev = 0.0
    for i, v in enumerate(x):
        k = a if v > prev else r
        prev = k * prev + (1 - k) * v
        y[i] = prev
    return y


def _norm(x, pct=98):
    hi = np.percentile(x, pct) if len(x) else 1.0
    return np.clip(x / max(hi, 1e-9), 0, 1)


def _resample_to_frames(curve, curve_fps, n_frames, fps):
    t_src = np.arange(len(curve)) / curve_fps
    t_dst = np.arange(n_frames) / fps
    return np.interp(t_dst, t_src, curve)


def _rms_frames(y, n_frames, fps):
    hop = HOP
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    return _resample_to_frames(rms, SR / hop, n_frames, fps)


def _onset_frames(y, n_frames, fps):
    o = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    return _resample_to_frames(o, SR / HOP, n_frames, fps)


def _band(y, lo, hi):
    sos = sps.butter(4, [max(lo, 1), hi], btype="bandpass", fs=SR, output="sos") if lo > 0 \
        else sps.butter(4, hi, btype="lowpass", fs=SR, output="sos")
    return sps.sosfiltfilt(sos, y)


def analyse(master, stems_dir=None, fps=30, beats_override=None):
    y, _ = librosa.load(master, sr=SR, mono=True)
    duration = len(y) / SR
    n = int(round(duration * fps))

    # Elements: real stems if we have them, else rough bands of the mix.
    parts = {}
    if stems_dir and os.path.isdir(stems_dir):
        for name in ("drums", "bass", "vocals", "other"):
            for ext in (".wav", ".flac"):
                p = os.path.join(stems_dir, name + ext)
                if os.path.exists(p):
                    parts[name] = librosa.load(p, sr=SR, mono=True)[0]
    if len(parts) < 4:
        harm, perc = librosa.effects.hpss(y)
        parts.setdefault("bass", _band(y, 0, 150))
        parts.setdefault("drums", perc)
        parts.setdefault("vocals", _band(harm, 300, 3400))
        parts.setdefault("other", _band(harm, 3400, 9000))

    # Beats and bars (4/4). On the laptop beat_this output can be passed in.
    if beats_override:
        beats = np.array(beats_override["beats"])
        downbeats = np.array(beats_override["downbeats"])
        tempo = float(beats_override.get("tempo", 60.0 / np.median(np.diff(beats))))
    else:
        tempo, bf = librosa.beat.beat_track(y=y, sr=SR, hop_length=HOP, units="frames")
        tempo = float(np.atleast_1d(tempo)[0])
        beats = librosa.frames_to_time(bf, sr=SR, hop_length=HOP)
        # downbeat phase: the beat phase with the most low-end energy
        bass_env = librosa.feature.rms(y=parts["bass"], hop_length=HOP)[0]
        scores = []
        for ph in range(4):
            idx = bf[ph::4]
            idx = idx[idx < len(bass_env)]
            scores.append(bass_env[idx].mean() if len(idx) else 0)
        downbeats = beats[int(np.argmax(scores))::4]

    env, hits = {}, {}
    for name, sig in parts.items():
        raw = _norm(_rms_frames(sig, n, fps))
        env[name] = _smooth(raw, fps, 0.03, 0.35)
    for name in ("bass", "drums"):
        hits[name] = _norm(_onset_frames(parts[name], n, fps), 99.5)

    total = _norm(_rms_frames(y, n, fps))
    energy = _smooth(total, fps, 1.5, 3.0)
    # tension: energy rising over the last ~8 s, plus onset density
    dens = _smooth(_norm(_onset_frames(y, n, fps)), fps, 2.0, 2.0)
    lag = int(8 * fps)
    rise = np.clip(energy - np.concatenate([np.full(lag, energy[0]), energy[:-lag]]), 0, 1) if n > lag else np.zeros(n)
    tension = _norm(_smooth(0.6 * _norm(rise) + 0.4 * dens, fps, 1.0, 2.0))
    stillness = _smooth((total < 0.08).astype(float), fps, 0.5, 1.0)

    r = lambda a: [round(float(v), 4) for v in a]
    return {
        "version": 1,
        "duration": round(duration, 4),
        "fps": fps,
        "frames": n,
        "tempo": round(tempo, 3),
        "beats": r(beats),
        "downbeats": r(downbeats),
        "env": {k: r(v) for k, v in env.items()},
        "hits": {k: r(v) for k, v in hits.items()},
        "macro": {"energy": r(energy), "tension": r(tension), "stillness": r(stillness)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("master")
    ap.add_argument("out")
    ap.add_argument("--stems")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--beats", help="JSON with beats/downbeats/tempo (e.g. from beat_this)")
    a = ap.parse_args()
    bo = json.load(open(a.beats)) if a.beats else None
    data = analyse(a.master, a.stems, a.fps, bo)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(data, fh, separators=(",", ":"))
    print(f"{a.out}: {data['duration']:.1f}s, {data['tempo']:.1f} BPM, "
          f"{len(data['beats'])} beats, {len(data['downbeats'])} bars")


if __name__ == "__main__":
    main()
