"""repaint_check.py ORIGINAL REPAINTED START END: null test outside the repaint window, change inside it, seam levels."""
import json
import sys

import numpy as np
import soundfile as sf

o, sr = sf.read(sys.argv[1], always_2d=True)
r, sr2 = sf.read(sys.argv[2], always_2d=True)
assert sr == sr2, (sr, sr2)
s, e = float(sys.argv[3]), float(sys.argv[4])
n = min(len(o), len(r))
o, r = o[:n], r[:n]


def db(x):
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12))


margin = 1.0  # skip 1 s either side of the window (latent crossfade is ~0.4 s)
S, E, M = int(s * sr), int(e * sr), int(margin * sr)
outside = np.concatenate([np.arange(0, max(S - M, 0)), np.arange(min(E + M, n), n)])
# ACE-Step peak-normalises its output to -1 dBFS, which rescales the whole file when the repainted part is louder.
# Fit that single gain outside the window, then null-test.
g = float((o[outside] * r[outside]).sum() / ((o[outside] ** 2).sum() + 1e-12))
d = r - g * o
res = {
    "output_gain_vs_original_db": round(20 * np.log10(g), 2),
    "length_s": round(n / sr, 2),
    "outside_residual_rel_db": round(db(d[outside]) - db(o[outside]), 1),
    "inside_change_rel_db": round(db(d[S:E]) - db(o[S:E]), 1),
    "inside_level_db": {"original": round(db(o[S:E]), 1), "repaint": round(db(r[S:E]), 1)},
    "seam_levels_db": {
        "before_start": round(db(r[S - sr:S]), 1), "after_start": round(db(r[S:S + sr]), 1),
        "before_end": round(db(r[E - sr:E]), 1), "after_end": round(db(r[E:E + sr]), 1)},
}
w = 5 * sr
res["outside_residual_per_5s_worst_rel_db"] = round(max(
    db(d[i:i + w]) - db(o[i:i + w]) for i in range(0, n - w, w) if (i + w <= S - M or i >= E + M)), 1)
print(json.dumps(res))
