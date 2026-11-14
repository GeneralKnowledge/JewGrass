"""stemcheck.py ORIGINAL VOCALS INSTRUMENTAL: null test (orig - (voc+inst)) and vocal presence per 5 s.

Stems may be at a different sample rate (audio-separator writes 44.1 kHz); they are resampled to the original's rate.
"""
import json
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd


def load(p, sr_target=None):
    d, sr = sf.read(p, always_2d=True)
    if sr_target and sr != sr_target:
        g = gcd(sr, sr_target)
        d = resample_poly(d, sr_target // g, sr // g, axis=0)
        sr = sr_target
    return d, sr


def db(x):
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12))


orig, sr = load(sys.argv[1])
voc, _ = load(sys.argv[2], sr)
ins, _ = load(sys.argv[3], sr)
n = min(len(orig), len(voc), len(ins))
orig, voc, ins = orig[:n], voc[:n], ins[:n]
resid = orig - (voc + ins)
w = 5 * sr
pres = [round(db(voc[i:i + w]) - db(orig[i:i + w]), 1) for i in range(0, n - w + 1, w)]
res = {"original_rms_db": round(db(orig), 2), "vocals_rms_db": round(db(voc), 2),
       "instrumental_rms_db": round(db(ins), 2), "null_residual_rms_db": round(db(resid), 2),
       "null_residual_rel_db": round(db(resid) - db(orig), 2),
       "vocal_minus_mix_db_per_5s": pres,
       "vocal_active_fraction": round(float(np.mean(np.array(pres) > -15)), 2)}
print(json.dumps(res))
