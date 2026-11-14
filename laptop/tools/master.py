"""master.py IN_WAV OUT_DIR REFERENCE_WAV: pedalboard chain to -14 LUFS / <= -1 dBTP, plus Matchering against a reference.

Writes OUT_DIR/pedalboard.wav, OUT_DIR/matchering.wav and OUT_DIR/master.json (before/after LUFS and true peak).
Runs in ~/band/tools/fx/.venv (CPU).
"""
import json
import os
import sys

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from pedalboard import Compressor, Gain, HighpassFilter, Limiter, Pedalboard  # noqa: F401
from scipy.signal import resample_poly

src, out_dir, ref = sys.argv[1:4]
os.makedirs(out_dir, exist_ok=True)


def measure(x, sr):
    lufs = pyln.Meter(sr).integrated_loudness(x)
    tp = 20 * np.log10(np.abs(resample_poly(x, 4, 1, axis=0)).max() + 1e-12)
    return round(float(lufs), 2), round(float(tp), 2)


x, sr = sf.read(src, always_2d=True, dtype="float32")
before = measure(x, sr)

# HPF 30 Hz, gentle 2:1 compression with a slow attack, then limit.
pre = Pedalboard([HighpassFilter(cutoff_frequency_hz=30.0),
                  Compressor(threshold_db=-18.0, ratio=2.0, attack_ms=30.0, release_ms=250.0)])
y = pre(x.T, sr).T
target, ceiling = -14.0, -1.0
# pedalboard's Limiter adds make-up gain of -threshold_db (output peaks at 0 dBFS), so loudness is set by the
# threshold and the ceiling by a trim gain after it. Binary-search the threshold for -14 LUFS; lower the trim
# until the 4x-oversampled true peak is <= -1 dBTP.
trim = -1.3
for _ in range(6):
    lo, hi = -40.0, 0.0
    for _ in range(14):
        thr = (lo + hi) / 2
        z = Pedalboard([Limiter(threshold_db=thr, release_ms=100.0), Gain(gain_db=trim)])(y.T, sr).T
        lufs, tp = measure(z, sr)
        if lufs < target:
            hi = thr
        else:
            lo = thr
        if abs(lufs - target) <= 0.05:
            break
    if tp <= ceiling:
        break
    trim -= (tp - ceiling) + 0.05
gain = trim
sf.write(os.path.join(out_dir, "pedalboard.wav"), z, sr, subtype="PCM_24")
pb = measure(z, sr)

res = {"source": src, "before": {"lufs": before[0], "true_peak_dbtp": before[1]},
       "pedalboard": {"lufs": pb[0], "true_peak_dbtp": pb[1], "limiter_threshold_db": round(thr, 2),
                      "output_trim_db": round(gain, 2)}}
try:
    import matchering as mg

    mg.log(print)
    mout = os.path.join(out_dir, "matchering.wav")
    mg.process(target=src, reference=ref, results=[mg.pcm24(mout)])
    m, msr = sf.read(mout, always_2d=True)
    ml = measure(m, msr)
    res["matchering"] = {"reference": ref, "lufs": ml[0], "true_peak_dbtp": ml[1]}
except Exception as exc:  # noqa: BLE001
    res["matchering"] = {"error": repr(exc)}
json.dump(res, open(os.path.join(out_dir, "master.json"), "w"), indent=2)
print(json.dumps(res))
