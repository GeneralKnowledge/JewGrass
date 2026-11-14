"""analyze.py WAV OUT_JSON: beat_this beats/downbeats + BPM, key estimate, LUFS, true peak, LRA, dropouts.

Runs in ~/band/tools/fx/.venv (CPU).
"""
import json
import sys

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import resample_poly

path, out = sys.argv[1:3]
data, sr = sf.read(path, always_2d=True)
mono = data.mean(axis=1)
dur = len(mono) / sr

# Beats and downbeats.
from beat_this.inference import File2Beats

f2b = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)
beats, downbeats = f2b(path)
beats, downbeats = np.asarray(beats), np.asarray(downbeats)
ibi = np.diff(beats)
bpm = float(60 / np.median(ibi)) if len(ibi) else float("nan")

# Key: Krumhansl-Schmuckler on mean CQT chroma.
y22 = librosa.resample(mono.astype(np.float32), orig_sr=sr, target_sr=22050)
chroma = librosa.feature.chroma_cqt(y=y22, sr=22050).mean(axis=1)
maj = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
minr = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
scores = []
for i in range(12):
    scores.append((np.corrcoef(chroma, np.roll(maj, i))[0, 1], f"{names[i]} major"))
    scores.append((np.corrcoef(chroma, np.roll(minr, i))[0, 1], f"{names[i]} minor"))
scores.sort(reverse=True)

# Loudness.
meter = pyln.Meter(sr)
lufs = float(meter.integrated_loudness(data))
tp = float(20 * np.log10(np.abs(resample_poly(data, 4, 1, axis=0)).max() + 1e-12))
# Loudness range (EBU 3342): 3 s short-term windows, 1 s hop, abs gate -70, rel gate -20, 10th..95th pct.
st = []
for s in range(0, len(data) - 3 * sr, sr):
    blk = data[s:s + 3 * sr]
    # K-weighted mean square via pyloudnorm's filters
    filt = blk.copy()
    for f in meter._filters.values():
        filt = f.apply_filter(filt)
    ms = (filt ** 2).mean(axis=0).sum()
    st.append(-0.691 + 10 * np.log10(ms + 1e-12))
st = np.array(st)
st = st[st > -70]
st = st[st > (10 * np.log10(np.mean(10 ** (st / 10))) - 20)]
lra = float(np.percentile(st, 95) - np.percentile(st, 10)) if len(st) else float("nan")

# Per-bar loudness (bars = downbeat to downbeat) and dropouts (>0.5 s below -50 dBFS inside the song).
bars = []
for a, b in zip(downbeats[:-1], downbeats[1:]):
    seg = mono[int(a * sr):int(b * sr)]
    bars.append({"start": round(float(a), 3), "end": round(float(b), 3),
                 "rms_db": round(float(20 * np.log10(np.sqrt((seg ** 2).mean()) + 1e-9)), 1)})
hop = int(0.05 * sr)
env = np.array([np.sqrt((mono[i:i + hop] ** 2).mean()) for i in range(0, len(mono) - hop, hop)])
quiet = 20 * np.log10(env + 1e-9) < -50
drop, run, start = [], 0, 0
for i, q in enumerate(quiet):
    if q:
        if run == 0:
            start = i
        run += 1
    else:
        if run * 0.05 >= 0.5 and start * 0.05 > 5:
            drop.append([round(start * 0.05, 2), round(i * 0.05, 2)])
        run = 0
song_end = None
if run * 0.05 >= 0.5:
    song_end = round(start * 0.05, 2)

res = {
    "file": path, "duration_s": round(dur, 2), "sample_rate": sr,
    "bpm_beat_this": round(bpm, 2), "n_beats": int(len(beats)), "n_downbeats": int(len(downbeats)),
    "beats": [round(float(x), 3) for x in beats], "downbeats": [round(float(x), 3) for x in downbeats],
    "key_estimate": scores[0][1], "key_runner_up": scores[1][1], "key_confidence": round(float(scores[0][0]), 3),
    "integrated_lufs": round(lufs, 2), "true_peak_dbtp": round(tp, 2), "loudness_range_lu": round(lra, 2),
    "dropouts_s": drop, "audio_ends_at_s": song_end, "bars": bars,
}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("beats", "downbeats", "bars")}))
