"""check_audio.py WAV...: duration, loudness per 15 s, tempo per 60 s window (numpy onset autocorrelation)."""
import sys
import numpy as np
import soundfile as sf

def tempo(x, sr):
    hop = 512
    n = (len(x) - 2048) // hop
    if n < 10:
        return float("nan")
    frames = np.lib.stride_tricks.sliding_window_view(x, 2048)[::hop][:n] * np.hanning(2048)
    mag = np.log1p(np.abs(np.fft.rfft(frames, axis=1)))
    flux = np.maximum(np.diff(mag, axis=0), 0).sum(axis=1)
    flux -= flux.mean()
    ac = np.correlate(flux, flux, "full")[len(flux) - 1:]
    fps = sr / hop
    lags = np.arange(len(ac))
    ok = (lags >= fps * 60 / 180) & (lags <= fps * 60 / 70)
    best = lags[ok][np.argmax(ac[ok])]
    return 60 * fps / best

for p in sys.argv[1:]:
    d, sr = sf.read(p)
    m = d.mean(axis=1) if d.ndim > 1 else d
    dur = len(m) / sr
    w = 15 * sr
    db = [20 * np.log10(np.sqrt((m[i:i + w] ** 2).mean()) + 1e-9) for i in range(0, len(m) - sr, w)]
    jumps = np.abs(np.diff(db))
    bpms = [tempo(m[i:i + 60 * sr], sr) for i in range(0, len(m) - 30 * sr, 60 * sr)]
    print(f"{p.split('/')[-1]}: {dur:.1f}s nan={bool(np.isnan(d).any())} peak={np.abs(d).max():.2f}")
    print("  dB/15s:", " ".join(f"{v:.0f}" for v in db), f"| max step {jumps.max():.1f} dB")
    sec = [20 * np.log10(np.sqrt((m[i:i + sr] ** 2).mean()) + 1e-9) for i in range(len(m) - 8 * sr, len(m) - sr + 1, sr)]
    body = np.median(db)
    ending = "fades/ends" if sec[-1] < body - 12 else ("tails off" if sec[-1] < body - 6 else "still loud (likely cut off)")
    print("  last 8 s dB/1s:", " ".join(f"{v:.0f}" for v in sec), f"| body median {body:.0f} dB -> {ending}")
    print("  BPM/60s:", " ".join(f"{b:.0f}" for b in bpms))
