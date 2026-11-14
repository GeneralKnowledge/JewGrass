"""Tiny numpy synth: renders an arrangement's MIDI so it can be heard without a DAW.

Not a production sound. It exists for two jobs: a quick listen to the MIDI
sketch, and the synthetic transcription test (selftest.py), where we need audio
whose notes we know exactly.
"""
import numpy as np

import arrangement as A

SR = 44100


def _pluck(f, n):
    """Plucked string: harmonics that die away faster the higher they are."""
    t = np.arange(n) / SR
    y = np.zeros(n)
    for k in range(1, 9):
        if f * k > SR / 2:
            break
        y += np.sin(2 * np.pi * f * k * t) / k * np.exp(-t * (1.5 + 0.9 * k))
    return y * np.minimum(1, t / 0.003)


def _tone(f, n, kind):
    t = np.arange(n) / SR
    if kind == "bass":
        y = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)
        env = np.exp(-t * 2.5)
    else:                         # soft pad
        y = sum(np.sin(2 * np.pi * f * k * t) / k for k in (1, 2, 3))
        env = np.minimum(1, t / 0.08)
    rel = np.minimum(1, (n - np.arange(n)) / (0.05 * SR))
    return y * env * rel


def render_track(arr, track, total_sec=None):
    total_sec = total_sec or A.beat_to_sec(arr, A.total_beats(arr)) + 2
    out = np.zeros(int(total_sec * SR))
    prog = track["instrument"]["program"]
    kind = "pluck" if 24 <= prog <= 31 else "bass" if 32 <= prog <= 39 else "pad"
    for c in track.get("clips", []):
        for s, d, p, v in c["notes"]:
            t0 = A.beat_to_sec(arr, c["start"] + s)
            n = int(A.beat_to_sec(arr, d) * SR) + (int(0.4 * SR) if kind == "pluck" else 0)
            f = 440.0 * 2 ** ((p - 69) / 12)
            y = _pluck(f, n) if kind == "pluck" else _tone(f, n, kind)
            i0 = int(t0 * SR)
            y = y[: max(0, len(out) - i0)]
            out[i0:i0 + len(y)] += y * (v / 127.0) * (0.5 if kind == "pad" else 0.8)
    return out


def write_wav(path, y):
    import soundfile as sf
    peak = np.max(np.abs(y)) or 1.0
    sf.write(path, (y / peak * 0.8).astype(np.float32), SR, subtype="PCM_16")
