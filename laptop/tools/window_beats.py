"""window_beats.py ANALYSIS_JSON START END: odd beats/bars and bar levels inside [START-5, END+5]."""
import json
import sys

import numpy as np

a = json.load(open(sys.argv[1]))
s, e = float(sys.argv[2]) - 5, float(sys.argv[3]) + 5
b, d = np.array(a["beats"]), np.array(a["downbeats"])
ibi, bl = np.diff(b), np.diff(d)
mb, mbl = np.median(ibi), np.median(bl)
odd_beats = [(round(float(b[i]), 2), round(float(ibi[i]), 3)) for i in range(len(ibi)) if s <= b[i] <= e and abs(ibi[i] / mb - 1) > .15]
odd_bars = [(round(float(d[i]), 2), round(float(bl[i]), 2)) for i in range(len(bl)) if s <= d[i] <= e and abs(bl[i] / mbl - 1) > .1]
bars = [(x["start"], x["rms_db"]) for x in a["bars"] if s <= x["start"] <= e]
print(json.dumps({"bpm": a["bpm_beat_this"], "odd_beats": odd_beats, "odd_bars": odd_bars, "bar_rms": bars}))
