"""stem_master.py ORIGINAL VOCALS INSTRUMENTAL OUT_DIR [VOCAL_GAIN_DB=1.0]

Processes the two stems separately, remixes them and writes OUT_DIR/stem-remix.wav (unmastered, 48 kHz) and
OUT_DIR/stem-remix.json. Master it afterwards with master.py.
  vocals: gentle gate/expander -> high-pass 90 Hz -> de-esser (5-8 kHz band compressed) -> 3:1 compressor
          -> light presence EQ (+2 dB at 3 kHz) -> short reverb (8% wet)
  instrumental: high-pass 30 Hz -> 1.5:1 glue compression
  remix: each stem's RMS matched to the original separated stem, vocals then +VOCAL_GAIN_DB.
Runs in ~/band/tools/fx/.venv (CPU). Stems from audio-separator are 44.1 kHz; they are resampled to the original rate.
"""
import json
import os
import sys
from math import gcd

import numpy as np
import soundfile as sf
from pedalboard import (Compressor, HighpassFilter, LowpassFilter, NoiseGate, Pedalboard, PeakFilter, Reverb)
from scipy.signal import resample_poly

orig_p, voc_p, ins_p, out_dir = sys.argv[1:5]
vgain = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
os.makedirs(out_dir, exist_ok=True)


def load(p, sr_target=None):
    d, sr = sf.read(p, always_2d=True, dtype="float32")
    if sr_target and sr != sr_target:
        g = gcd(sr, sr_target)
        d = resample_poly(d, sr_target // g, sr // g, axis=0).astype(np.float32)
        sr = sr_target
    return d, sr


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)) + 1e-12)


def db(x):
    return 20 * np.log10(rms(x))


orig, sr = load(orig_p)
voc, _ = load(voc_p, sr)
ins, _ = load(ins_p, sr)
n = min(len(orig), len(voc), len(ins))
orig, voc, ins = orig[:n], voc[:n], ins[:n]

# Vocals.
v = Pedalboard([NoiseGate(threshold_db=-50.0, ratio=2.0, attack_ms=2.0, release_ms=150.0),
                HighpassFilter(cutoff_frequency_hz=90.0)])(voc.T, sr)
band = Pedalboard([HighpassFilter(cutoff_frequency_hz=5000.0), LowpassFilter(cutoff_frequency_hz=8000.0)])(v, sr)
band_c = Compressor(threshold_db=-32.0, ratio=4.0, attack_ms=1.0, release_ms=60.0)(band, sr)
v = v - band + band_c  # de-essed: only the 5-8 kHz band is compressed
v = Pedalboard([Compressor(threshold_db=-24.0, ratio=3.0, attack_ms=10.0, release_ms=120.0),
                PeakFilter(cutoff_frequency_hz=3000.0, gain_db=2.0, q=0.8),
                Reverb(room_size=0.2, damping=0.6, wet_level=0.08, dry_level=1.0, width=0.8)])(v, sr).T
# Instrumental.
i = Pedalboard([HighpassFilter(cutoff_frequency_hz=30.0),
                Compressor(threshold_db=-20.0, ratio=1.5, attack_ms=30.0, release_ms=300.0)])(ins.T, sr).T

v_gain = 20 * np.log10(rms(voc) / rms(v)) + vgain
i_gain = 20 * np.log10(rms(ins) / rms(i))
mix = v * 10 ** (v_gain / 20) + i * 10 ** (i_gain / 20)
peak = float(np.abs(mix).max())
if peak > 0.98:  # keep headroom before the master; master.py sets the final level anyway
    mix *= 0.98 / peak
out = os.path.join(out_dir, "stem-remix.wav")
sf.write(out, mix, sr, subtype="FLOAT")
res = {"original": orig_p, "vocals": voc_p, "instrumental": ins_p, "sample_rate": sr,
       "vocal_gain_db": round(v_gain, 2), "instrumental_gain_db": round(i_gain, 2), "vocal_boost_db": vgain,
       "vocal_to_inst_db": {"original_stems": round(db(voc) - db(ins), 2),
                            "remix": round(db(v * 10 ** (v_gain / 20)) - db(i * 10 ** (i_gain / 20)), 2)},
       "peak_scaled": peak > 0.98, "output": out}
json.dump(res, open(os.path.join(out_dir, "stem-remix.json"), "w"), indent=2)
print(json.dumps(res))
