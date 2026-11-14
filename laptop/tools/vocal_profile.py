"""vocal_profile.py VOCAL_STEM [MIX]: vocal level per 1 s, seconds with vocal (>-40 dBFS), and longest vocal run."""
import json
import sys

import numpy as np
import soundfile as sf

v, sr = sf.read(sys.argv[1], always_2d=True)
v = v.mean(axis=1)
lv = [20 * np.log10(np.sqrt(np.mean(v[i:i + sr] ** 2)) + 1e-12) for i in range(0, len(v) - sr + 1, sr)]
act = [x > -40 for x in lv]
runs, cur = [], 0
for a in act:
    cur = cur + 1 if a else 0
    runs.append(cur)
print(json.dumps({"vocal_db_per_s": [int(x) for x in lv], "seconds_with_vocal": int(sum(act)),
                  "longest_vocal_run_s": max(runs) if runs else 0}))
