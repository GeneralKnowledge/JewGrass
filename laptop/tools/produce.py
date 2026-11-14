"""produce.py CONFIG_JSON: stem-based production and mastering for one song.

Runs in ~/band/tools/fx/.venv (CPU), through ~/band/heavy.sh. Every processing amount is derived from a
measurement of the stem (noise floor, RMS, crest factor, band balance), and the config can override any of it.

Config (JSON), paths relative to the config file's folder or absolute:
  {
    "original": "40-edits/outro/repaint_219.64-240_8003.wav",      # the full mix the stems came from
    "analysis": "10-analysis/analysis-v2-8003.json",               # analyze.py output (downbeats/bars)
    "stems": {"vocals": "...(Vocals)...wav",                        # BS-RoFormer lead vocal
              "drums": "...", "bass": "...", "guitar": "...", "piano": "...", "other": "..."},  # Demucs
    "output": "60-master/production-v1.wav",
    "reference_master": "60-master/pedalboard-v2.wav",              # compared in the report only
    "sections": null,          # null -> detected (written back to <config>.sections.json); or a list of
                               # {"start": s, "end": s, "type": "spoken|rap|chorus|bridge|chant"}
    "overrides": {}            # any key of DEFAULTS below
  }

Outputs: the master WAV (24-bit), <output>.json report, and <output>-premaster.wav (the remix before mastering).
Harmony layers other than an octave are not done (no pitch tracking yet).
"""
import json
import os
import sys
import time
from math import gcd

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from pedalboard import (Compressor, Delay, Distortion, HighpassFilter, HighShelfFilter, LowpassFilter,
                        LowShelfFilter, NoiseGate, Pedalboard, PeakFilter, PitchShift, Reverb)
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import butter, resample_poly, sosfiltfilt, welch

SR = 48000
DEFAULTS = {
    "vocal_boost_db": 1.0,          # lead vocal level vs the original separated stem
    "eq_strength": 0.5,             # fraction of the measured gap to the target curve that EQ closes
    "eq_cap_db": 4.0,               # no single EQ move bigger than this
    "gate_above_floor_db": 6.0,     # gate/expander threshold above the measured noise floor
    "deess_above_band_db": 4.0,     # de-esser threshold above the 5-8 kHz band's RMS
    "vocal_comp_ratio": 3.0,
    "doubles": [{"cents": 7, "delay_ms": 18, "pan": -0.7, "gain_db": -9.0},
                {"cents": -6, "delay_ms": 27, "pan": 0.7, "gain_db": -9.0}],
    "octave_gain_db": -16.0,        # octave-down layer under chant hooks / gang choruses
    "duck_db": 2.5,                 # sidechain depth on the instrument bus, 1-4 kHz
    "drum_parallel_mix": 0.35,
    "bass_sat_gain_db": -14.0,      # parallel saturated bass (harmonics above 150 Hz)
    "inst_width": 1.15,             # side gain for guitar/piano/other
    "mono_below_hz": 120.0,
    "target_lufs": -14.0,
    "ceiling_dbtp": -1.0,
    "max_limiter_gr_db": 3.0,
    "xfade_ms": 60.0,
}
BANDS = [("sub", 20, 60), ("low", 60, 250), ("lowmid", 250, 1000), ("mid", 1000, 4000),
         ("presence", 4000, 8000), ("air", 8000, 16000)]
CENTERS = {"sub": 40, "low": 120, "lowmid": 500, "mid": 2000, "presence": 5600, "air": 11000}
# Rough target band balance (dB relative to the stem's total power) for this genre; a starting point, not a law.
TARGETS = {
    "vocals": {"low": -12, "lowmid": -5, "mid": -3, "presence": -10, "air": -18},
    "guitar": {"low": -9, "lowmid": -4, "mid": -4, "presence": -12, "air": -20},
    "piano": {"low": -8, "lowmid": -4, "mid": -5, "presence": -13, "air": -21},
    "other": {"low": -8, "lowmid": -4, "mid": -5, "presence": -12, "air": -19},
    "bass": {"sub": -4, "low": -2, "lowmid": -12, "mid": -22},
}


# ----------------------------------------------------------------------------- helpers
def load(path):
    d, sr = sf.read(path, always_2d=True, dtype="float32")
    if sr != SR:
        g = gcd(sr, SR)
        d = resample_poly(d, SR // g, sr // g, axis=0).astype(np.float32)
    if d.shape[1] == 1:
        d = np.repeat(d, 2, axis=1)
    return d


def db(x):
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-12))


def lin(d):
    return 10 ** (d / 20)


def pb(chain, x):
    return Pedalboard(chain)(x.T.copy(), SR).T


def zp_split(x, hz):
    """Zero-phase low/high split that sums back exactly."""
    lo = sosfiltfilt(butter(4, hz, "lowpass", fs=SR, output="sos"), x, axis=0).astype(np.float32)
    return lo, x - lo


def band(x, lo_hz, hi_hz):
    return sosfiltfilt(butter(4, [lo_hz, hi_hz], "bandpass", fs=SR, output="sos"), x, axis=0).astype(np.float32)


def measure(x):
    m = x.mean(axis=1)
    hop = int(0.05 * SR)
    fr = np.sqrt(np.mean(m[: len(m) // hop * hop].reshape(-1, hop) ** 2, axis=1)) + 1e-12
    fr_db = 20 * np.log10(fr)
    live = fr_db[fr_db > -90]
    f, p = welch(m, fs=SR, nperseg=8192)
    tot = p[(f >= 20) & (f <= 16000)].sum() + 1e-20
    bands = {n: round(float(10 * np.log10(p[(f >= a) & (f < b)].sum() / tot + 1e-20)), 1) for n, a, b in BANDS}
    rms = db(m)
    peak = float(20 * np.log10(np.abs(m).max() + 1e-12))
    return {"rms_db": round(rms, 1), "peak_db": round(peak, 1), "crest_db": round(peak - rms, 1),
            "noise_floor_db": round(float(np.percentile(live, 10)) if len(live) else -90.0, 1),
            "active_db": round(float(np.percentile(live, 70)) if len(live) else -90.0, 1), "bands": bands}


def eq_moves(meas, target, cfg):
    moves = []
    for name, t in target.items():
        diff = (t - meas["bands"][name]) * cfg["eq_strength"]
        diff = float(np.clip(diff, -cfg["eq_cap_db"], cfg["eq_cap_db"]))
        if abs(diff) >= 0.5:
            moves.append({"band": name, "hz": CENTERS[name], "gain_db": round(diff, 1)})
    return moves


def eq_chain(moves):
    out = []
    for mv in moves:
        if mv["band"] == "sub":
            out.append(LowShelfFilter(cutoff_frequency_hz=60, gain_db=mv["gain_db"], q=0.7))
        elif mv["band"] == "air":
            out.append(HighShelfFilter(cutoff_frequency_hz=8000, gain_db=mv["gain_db"], q=0.7))
        else:
            out.append(PeakFilter(cutoff_frequency_hz=mv["hz"], gain_db=mv["gain_db"], q=0.8))
    return out


def widen(x, side_gain):
    mid, side = (x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2 * side_gain
    return np.stack([mid + side, mid - side], axis=1)


def mono_below(x, hz):
    mid, side = (x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2
    side_lo = sosfiltfilt(butter(4, hz, "lowpass", fs=SR, output="sos"), side)
    side = side - side_lo
    return np.stack([mid + side, mid - side], axis=1).astype(np.float32)


def section_masks(sections, n, xfade_ms):
    types = ["spoken", "rap", "chorus", "bridge", "chant"]
    masks = {t: np.zeros(n, np.float32) for t in types}
    for s in sections:
        a, b = int(s["start"] * SR), min(int(s["end"] * SR), n)
        if s["type"] in masks and b > a:
            masks[s["type"]][a:b] = 1.0
    w = max(1, int(xfade_ms / 1000 * SR))
    for t in types:
        masks[t] = uniform_filter1d(masks[t], w)
    return masks


# ----------------------------------------------------------------------------- sections
def detect_sections(vocals, drums, downbeats, dur):
    """Bar-level guesses: spoken (vocal, little drums) at the start/end; chorus (wide, loud vocal = gang chant);
    bridge (bright vocal in the middle of the song); rap = other vocal bars."""
    bars = list(zip(downbeats[:-1], downbeats[1:]))
    feats = []
    for a, b in bars:
        i, j = int(a * SR), int(b * SR)
        v, d = vocals[i:j], drums[i:j]
        mid, side = (v[:, 0] + v[:, 1]) / 2, (v[:, 0] - v[:, 1]) / 2
        vm = mid
        f, p = welch(vm, fs=SR, nperseg=4096) if len(vm) > 4096 else (np.array([1.0]), np.array([1.0]))
        cent = float((f * p).sum() / (p.sum() + 1e-20))
        feats.append({"start": a, "end": b, "voc": db(v), "width": db(side) - db(mid), "drums": db(d), "cent": cent})
    voc = np.array([x["voc"] for x in feats])
    active = voc > (np.median(voc) - 12)
    drm = np.array([x["drums"] for x in feats])
    wid = np.array([x["width"] for x in feats])
    cen = np.array([x["cent"] for x in feats])
    labels = ["none"] * len(feats)
    for k in range(len(feats)):
        if active[k]:
            labels[k] = "rap"
    # spoken: leading/trailing vocal bars with drums well below their median
    quiet_drums = drm < (np.median(drm) - 8)
    for rng in (range(len(feats)), range(len(feats) - 1, -1, -1)):
        for k in rng:
            if not quiet_drums[k]:
                break
            if active[k]:
                labels[k] = "spoken"
    act_idx = np.where(active)[0]
    if len(act_idx):
        thr_w = np.percentile(wid[act_idx], 75)
        thr_c = np.percentile(cen[act_idx], 85)
        for k in act_idx:
            if labels[k] == "spoken":
                continue
            pos = feats[k]["start"] / dur
            if 0.35 <= pos <= 0.8 and cen[k] >= thr_c:
                labels[k] = "bridge"
            elif wid[k] >= thr_w:
                labels[k] = "chorus"
    # keep only runs of >= 2 bars for chorus/bridge; single bars fall back to rap
    for t in ("chorus", "bridge"):
        k = 0
        while k < len(labels):
            if labels[k] == t:
                j = k
                while j < len(labels) and labels[j] == t:
                    j += 1
                if j - k < 2:
                    for q in range(k, j):
                        labels[q] = "rap"
                k = j
            else:
                k += 1
    sections = []
    for f, lab in zip(feats, labels):
        if lab == "none":
            continue
        if sections and sections[-1]["type"] == lab and abs(sections[-1]["end"] - f["start"]) < 0.05:
            sections[-1]["end"] = round(float(f["end"]), 3)
        else:
            sections.append({"start": round(float(f["start"]), 3), "end": round(float(f["end"]), 3), "type": lab})
    # gang choruses double as chant hooks for the octave layer
    sections += [{"start": s["start"], "end": s["end"], "type": "chant"} for s in sections if s["type"] == "chorus"]
    return sections


# ----------------------------------------------------------------------------- stem chains
def process_vocals(v, cfg, sections, rep):
    m = measure(v)
    gate_thr = float(np.clip(m["noise_floor_db"] + cfg["gate_above_floor_db"], -75, -35))
    x = pb([NoiseGate(threshold_db=gate_thr, ratio=2.0, attack_ms=2.0, release_ms=120.0),
            HighpassFilter(cutoff_frequency_hz=90.0)], v)
    sib = band(x, 5000, 8000)
    sib_thr = db(sib) + cfg["deess_above_band_db"]
    x = x - sib + pb([Compressor(threshold_db=sib_thr, ratio=4.0, attack_ms=1.0, release_ms=60.0)], sib)
    comp_thr = m["active_db"] - 2.0
    moves = eq_moves(m, TARGETS["vocals"], cfg)
    x = pb([Compressor(threshold_db=comp_thr, ratio=cfg["vocal_comp_ratio"], attack_ms=8.0, release_ms=120.0)]
           + eq_chain(moves), x)
    masks = section_masks(sections, len(x), cfg["xfade_ms"])
    versions = {
        "spoken": pb([Delay(delay_seconds=0.09, feedback=0.0, mix=0.12)], x),
        "rap": pb([Reverb(room_size=0.25, damping=0.5, wet_level=0.08, dry_level=1.0, width=0.7)], x),
        "bridge": pb([Delay(delay_seconds=0.32, feedback=0.25, mix=0.12),
                      Reverb(room_size=0.7, damping=0.4, wet_level=0.2, dry_level=1.0, width=1.0)], x),
    }
    versions["chorus"] = versions["rap"]
    cover = np.clip(masks["spoken"] + masks["rap"] + masks["chorus"] + masks["bridge"], 0, 1)
    out = sum(masks[t][:, None] * versions[t] for t in ("spoken", "rap", "chorus", "bridge"))
    out = out + (1 - cover)[:, None] * versions["rap"]
    del versions
    # chorus doubles and the octave layer, processed per section to save CPU
    mono = x.mean(axis=1)
    extra = np.zeros_like(x)
    pad = int(0.5 * SR)
    for s in sections:
        if s["type"] not in ("chorus", "chant"):
            continue
        a, b = max(0, int(s["start"] * SR) - pad), min(len(x), int(s["end"] * SR) + pad)
        seg = mono[a:b].copy()
        w = masks["chorus" if s["type"] == "chorus" else "chant"][a:b]
        if s["type"] == "chorus":
            for d in cfg["doubles"]:
                y = Pedalboard([PitchShift(semitones=d["cents"] / 100.0)])(seg[None, :], SR)[0]
                sh = int(d["delay_ms"] / 1000 * SR)
                y = np.concatenate([np.zeros(sh, np.float32), y[: len(y) - sh]]) * lin(d["gain_db"]) * w
                pl, pr = np.sqrt((1 - d["pan"]) / 2), np.sqrt((1 + d["pan"]) / 2)
                extra[a:b, 0] += y * pl
                extra[a:b, 1] += y * pr
        else:
            y = Pedalboard([PitchShift(semitones=-12.0), LowpassFilter(cutoff_frequency_hz=3000.0)])(seg[None, :], SR)[0]
            y = y * lin(cfg["octave_gain_db"]) * w
            extra[a:b] += y[:, None] * 0.7071
    rep["vocals"] = {"measured": m, "gate_threshold_db": round(gate_thr, 1), "deess_threshold_db": round(sib_thr, 1),
                     "comp_threshold_db": round(comp_thr, 1), "eq": moves}
    return out.astype(np.float32), extra.astype(np.float32)


def process_drums(d, cfg, rep):
    m = measure(d)
    lo, hi = zp_split(d, 120.0)  # kick/sub vs the rest (Demucs gives no separate kick)
    par = pb([Compressor(threshold_db=m["rms_db"] - 6.0, ratio=8.0, attack_ms=1.0, release_ms=80.0)], hi)
    par *= lin(db(hi) - db(par))
    hi_out = hi + cfg["drum_parallel_mix"] * par
    out = lo + hi_out
    out *= lin(m["rms_db"] - db(out))
    rep["drums"] = {"measured": m, "parallel_threshold_db": round(m["rms_db"] - 6.0, 1),
                    "parallel_mix": cfg["drum_parallel_mix"]}
    return out.astype(np.float32)


def process_bass(b, cfg, rep):
    m = measure(b)
    moves = eq_moves(m, TARGETS["bass"], cfg)
    x = pb([LowpassFilter(cutoff_frequency_hz=8000.0)] + eq_chain(moves), b)
    sat = pb([HighpassFilter(cutoff_frequency_hz=150.0), Distortion(drive_db=10.0),
              LowpassFilter(cutoff_frequency_hz=5000.0)], x)
    sat *= lin(db(x) - db(sat) + cfg["bass_sat_gain_db"])
    x = mono_below(x + sat, cfg["mono_below_hz"])
    x *= lin(m["rms_db"] - db(x))
    rep["bass"] = {"measured": m, "eq": moves, "saturation_gain_db": cfg["bass_sat_gain_db"]}
    return x.astype(np.float32)


def process_inst(name, s, cfg, rep):
    m = measure(s)
    moves = eq_moves(m, TARGETS[name], cfg)
    x = pb(eq_chain(moves) + [PeakFilter(cutoff_frequency_hz=2500.0, gain_db=-1.5, q=0.9)], s)
    sat = pb([Distortion(drive_db=6.0)], x)
    x = x + sat * lin(db(x) - db(sat) - 18.0)
    x = widen(x, cfg["inst_width"])
    x *= lin(m["rms_db"] - db(x))
    rep[name] = {"measured": m, "eq": moves, "carve_2500hz_db": -1.5, "width": cfg["inst_width"]}
    return x.astype(np.float32)


def duck(bus, vocal, vocal_floor_db, depth_db, rep):
    env_src = band(vocal, 1000, 4000).mean(axis=1)
    hop = int(0.01 * SR)
    n = len(env_src) // hop
    e = 20 * np.log10(np.sqrt(np.mean(env_src[: n * hop].reshape(n, hop) ** 2, axis=1)) + 1e-12)
    thr = vocal_floor_db + 15.0
    amt = np.clip((e - thr) / 12.0, 0, 1) * depth_db
    amt = maximum_filter1d(amt, 3)               # ~10 ms attack
    amt = uniform_filter1d(amt, 15)              # ~150 ms release smoothing
    g = lin(-np.repeat(amt, hop))
    g = np.concatenate([g, np.full(len(bus) - len(g), g[-1] if len(g) else 1.0)])[: len(bus)].astype(np.float32)
    mid_band = band(bus, 1000, 4000)
    rep["sidechain"] = {"threshold_db": round(thr, 1), "max_duck_db": depth_db,
                        "ducked_fraction": round(float(np.mean(np.repeat(amt, hop) > 0.5)), 3)}
    return bus - mid_band + mid_band * g[:, None]


# ----------------------------------------------------------------------------- master
def true_peak(x):
    return float(20 * np.log10(np.abs(resample_poly(x, 4, 1, axis=0)).max() + 1e-12))


def lufs(x):
    return float(pyln.Meter(SR).integrated_loudness(x))


def limit(x, ceiling_lin, win_ms=60.0):
    """Look-ahead peak limiter: gain never above what each sample needs, smoothed over win_ms."""
    need = np.minimum(1.0, ceiling_lin / (np.abs(x).max(axis=1) + 1e-12))
    w = max(1, int(win_ms / 1000 * SR))
    g = -maximum_filter1d(-need, w)  # moving minimum
    g = uniform_filter1d(g, w)
    g = np.minimum(g, need)
    return (x * g[:, None]).astype(np.float32), float(-20 * np.log10(g.min() + 1e-12))


def master(x, cfg, rep):
    x = pb([HighpassFilter(cutoff_frequency_hz=25.0)], x)
    act = measure(x)["active_db"]
    x = pb([Compressor(threshold_db=act + 2.0, ratio=1.5, attack_ms=30.0, release_ms=300.0)], x)
    ceiling = cfg["ceiling_dbtp"] - 0.3
    lo_g, hi_g = -12.0, 18.0
    best = None
    for _ in range(18):
        g = (lo_g + hi_g) / 2
        y, gr = limit(x * lin(g), lin(ceiling))
        lv = lufs(y)
        if gr > cfg["max_limiter_gr_db"]:
            hi_g = g
            continue
        best = (g, y, gr, lv)
        if lv < cfg["target_lufs"]:
            lo_g = g
        else:
            hi_g = g
        if abs(lv - cfg["target_lufs"]) < 0.05:
            break
    g, y, gr, lv = best
    tp = true_peak(y)
    trim = 0.0
    if tp > cfg["ceiling_dbtp"]:
        trim = cfg["ceiling_dbtp"] - tp - 0.05
        y = y * lin(trim)
        lv, tp = lufs(y), true_peak(y)
    rep["master"] = {"input_gain_db": round(g, 2), "limiter_gr_max_db": round(gr, 2), "trim_db": round(trim, 2),
                     "lufs": round(lv, 2), "true_peak_dbtp": round(tp, 2),
                     "short_of_target_lu": round(max(0.0, cfg["target_lufs"] - lv), 2)}
    return y


def lra(x):
    meter = pyln.Meter(SR)
    st = []
    for s in range(0, len(x) - 3 * SR, SR):
        f = x[s:s + 3 * SR].astype(np.float64)
        for flt in meter._filters.values():
            f = flt.apply_filter(f)
        st.append(-0.691 + 10 * np.log10((f ** 2).mean(axis=0).sum() + 1e-12))
    st = np.array(st)
    st = st[st > -70]
    st = st[st > 10 * np.log10(np.mean(10 ** (st / 10))) - 20]
    return float(np.percentile(st, 95) - np.percentile(st, 10)) if len(st) else float("nan")


def checks(x):
    L, R = x[:, 0].astype(np.float64), x[:, 1].astype(np.float64)
    mono = ((L + R) / 2)[:, None].repeat(2, axis=1)
    side_lo = sosfiltfilt(butter(4, 120, "lowpass", fs=SR, output="sos"), (L - R) / 2)
    mid_lo = sosfiltfilt(butter(4, 120, "lowpass", fs=SR, output="sos"), (L + R) / 2)
    return {"lufs": round(lufs(x), 2), "true_peak_dbtp": round(true_peak(x), 2), "lra_lu": round(lra(x), 2),
            "lr_correlation": round(float(np.corrcoef(L, R)[0, 1]), 3),
            "mono_foldown_loss_lu": round(lufs(x) - lufs(mono), 2),
            "side_vs_mid_below_120hz_db": round(db(side_lo) - db(mid_lo), 1)}


# ----------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    cfg_path = os.path.abspath(sys.argv[1])
    base = os.path.dirname(cfg_path)
    conf = json.load(open(cfg_path))
    P = lambda p: p if os.path.isabs(p) else os.path.join(base, p)  # noqa: E731
    cfg = dict(DEFAULTS, **conf.get("overrides", {}))
    rep = {"config": cfg_path, "settings": cfg}

    vocals = load(P(conf["stems"]["vocals"]))
    drums = load(P(conf["stems"]["drums"]))
    n = min(len(vocals), len(drums))
    analysis = json.load(open(P(conf["analysis"])))
    sections = conf.get("sections")
    if not sections:
        sections = detect_sections(vocals[:n], drums[:n], np.array(analysis["downbeats"]), analysis["duration_s"])
        json.dump(sections, open(cfg_path.replace(".json", "") + ".sections.json", "w"), indent=1)
        rep["sections_source"] = "detected"
    else:
        rep["sections_source"] = "config"
    rep["sections"] = sections

    levels_before = {"vocals": round(db(vocals), 1), "drums": round(db(drums), 1)}
    voc, extra = process_vocals(vocals[:n], cfg, sections, rep)
    voc_floor = rep["vocals"]["measured"]["noise_floor_db"]
    voc *= lin(levels_before["vocals"] - db(voc) + cfg["vocal_boost_db"])
    dr = process_drums(drums[:n], cfg, rep)
    del vocals, drums
    mix = voc + extra + dr
    del dr
    bus = np.zeros_like(mix)
    for name in ("bass", "guitar", "piano", "other"):
        s = load(P(conf["stems"][name]))[:n]
        levels_before[name] = round(db(s), 1)
        y = process_bass(s, cfg, rep) if name == "bass" else process_inst(name, s, cfg, rep)
        if name == "bass":
            mix += y
        else:
            bus += y
        del s, y
    bus = duck(bus, voc, voc_floor, cfg["duck_db"], rep)
    mix = mono_below(mix + bus, cfg["mono_below_hz"])
    del bus
    pre = P(conf["output"]).replace(".wav", "-premaster.wav")
    sf.write(pre, mix, SR, subtype="FLOAT")
    rep["stem_levels_db"] = {"before": levels_before}
    out = master(mix, cfg, rep)
    sf.write(P(conf["output"]), out, SR, subtype="PCM_24")
    rep["checks"] = {"production": checks(out)}
    ref = conf.get("reference_master")
    if ref and os.path.exists(P(ref)):
        rep["checks"]["reference"] = checks(load(P(ref)))
    orig = load(P(conf["original"]))
    rep["checks"]["original_unmastered"] = checks(orig)
    rep["seconds"] = round(time.time() - t0, 1)
    json.dump(rep, open(P(conf["output"]).replace(".wav", ".json"), "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("checks", "seconds")} | {"master": rep["master"]}))


if __name__ == "__main__":
    main()
