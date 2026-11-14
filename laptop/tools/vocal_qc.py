"""vocal_qc.py VOCAL_STEM LYRICS OUT_DIR [--whisper WORDS.json] [--mix FULL_MIX.wav]: vocal defect DETECTION only.

Nothing is changed. A heuristic first pass; its scores are unvalidated until Reuben labels the clips by ear.

1. Segments: faster-whisper words (given, or run here with the Glen settings and initial_prompt = the lyrics).
   Gaps longer than 0.3 s with vocal energy in them are split into onset-based segments ("(no word)").
2. Features per segment, each z-scored robustly (median / MAD) against the song itself, so they mean
   "unusual for this voice in this song". Every feature is oriented so that higher = more suspicious:
     pitch_flat     voiced segments with very little pitch movement (robotic)    -std of cents
     jitter         frame-to-frame pitch wobble                                  mean |d cents|
     pitch_jump     jumps over 3 semitones inside one word                        count
     flatness       spectral flatness (noisy, buzzy)                              mean
     low_hnr        harmonic-to-noise ratio from HPSS (breathy, noisy)            -HNR dB
     ring           2-6 kHz resonance-peak prominence (metallic ring)             peak - median dB
     timbre_jump    MFCC 1-13 distance to the neighbouring segments              mean distance
     harsh_onset    3-10 kHz spectral flux at the word onset (sharp sibilants)    max flux
     low_prob       1 - whisper word probability (mushy or unclear diction)
     glitch         clicks (sample discontinuities) + short dropouts inside voiced audio  count
3. score = sum(WEIGHTS[f] * max(z_f, 0)); a feature "fires" at z >= FIRE_Z. Labels are conservative:
     repair-candidate: score >= REPAIR_SCORE and at least 2 features fired
     review:           score >= REVIEW_SCORE with at least 2 features fired, or any feature at z >= STRONG_Z
     natural:          otherwise
   Adjacent flagged segments (gap < 0.25 s) merge into regions. The fired features become the diagnosis, mapped
   to a suggested repair (REPAIRS).
4. OUT_DIR: qc-report.json, qc-report.md, clips/ (3.5 s of the full mix around each flagged region plus 10 random
   natural controls, shuffled under neutral names clip-NN.wav), clips/labels.csv to fill in, clips/key.json (answers).
Runs in ~/band/tools/fx/.venv (CPU) through flock + heavy.sh; whisper runs in ~/band/tools/asr/.venv.
"""
import argparse
import csv
import json
import os
import random
import subprocess

import librosa
import numpy as np
import soundfile as sf

WEIGHTS = {"pitch_flat": 1.0, "jitter": 1.0, "pitch_jump": 1.5, "flatness": 1.0, "low_hnr": 1.0, "ring": 1.0,
           "timbre_jump": 1.5, "harsh_onset": 0.75, "low_prob": 1.0, "glitch": 2.0}
FIRE_Z, STRONG_Z = 3.5, 8.0
REVIEW_SCORE, REPAIR_SCORE = 10.0, 16.0
COUNT_FEATURES = {"pitch_jump", "glitch"}  # z uses a scale of at least 1 count (most words have 0)
REPAIRS = {"pitch_flat": "pitch fix (add natural drift) or repaint the phrase",
           "jitter": "pitch fix (smooth the wobble)",
           "pitch_jump": "pitch fix, or repaint the phrase if the jump is musical nonsense",
           "flatness": "dynamic EQ / denoise the band", "low_hnr": "dynamic EQ (breath/noise band)",
           "ring": "dynamic EQ (notch the 2-6 kHz ring)", "timbre_jump": "repaint the phrase",
           "harsh_onset": "de-ess", "low_prob": "repaint the phrase (diction)", "glitch": "crossfade over the glitch"}
SR = 22050


def run_whisper(stem, lyrics, out_json):
    py = os.path.expanduser("~/band/tools/asr/.venv/bin/python")
    ww = os.path.expanduser("~/band/tools/whisper_words.py")
    subprocess.run([py, ww, stem, lyrics, out_json, "medium.en"], check=True)


def robust_z(v, min_scale=0.0):
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    out = np.zeros_like(v)
    if ok.sum() < 5:
        return out
    med = np.median(v[ok])
    mad = max(1.4826 * np.median(np.abs(v[ok] - med)), min_scale) + 1e-9
    out[ok] = (v[ok] - med) / mad
    return out


def segments_from_words(words, y, sr):
    segs = [{"start": w["start"], "end": max(w["end"], w["start"] + 0.06), "text": w["word"].strip(),
             "prob": w["probability"]} for w in words]
    segs.sort(key=lambda s: s["start"])
    filled, prev_end = [], 0.0
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    thr = np.percentile(rms[rms > 0], 60) * 0.5 if np.any(rms > 0) else 0
    for s in segs + [{"start": len(y) / sr, "end": len(y) / sr, "text": None, "prob": None}]:
        if s["start"] - prev_end > 0.3:
            a, b = int(prev_end * sr), int(s["start"] * sr)
            seg = y[a:b]
            if len(seg) > sr * 0.2 and np.sqrt(np.mean(seg ** 2)) > thr:
                on = librosa.onset.onset_detect(y=seg, sr=sr, units="time", backtrack=True)
                bounds = [0.0] + [t for t in on if 0.1 < t < (b - a) / sr - 0.1] + [(b - a) / sr]
                for p, q in zip(bounds[:-1], bounds[1:]):
                    if q - p >= 0.08:
                        filled.append({"start": prev_end + p, "end": prev_end + q, "text": "(no word)", "prob": None})
        if s["text"] is not None:
            filled.append(s)
        prev_end = max(prev_end, s["end"])
    return filled


def features(y, sr, y_hi, sr_hi, segs):
    hop = 256
    f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=1000, sr=sr, frame_length=1024, hop_length=hop)
    cents = 1200 * np.log2(np.where(voiced, f0, np.nan) / 55.0)
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=hop))
    H, P = librosa.decompose.hpss(S)
    flat = librosa.feature.spectral_flatness(S=S)[0]
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(librosa.feature.melspectrogram(S=S ** 2, sr=sr)), n_mfcc=14)[1:]
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    band_ring = (freqs >= 2000) & (freqs <= 6000)
    band_hi = (freqs >= 3000) & (freqs <= 10000)
    flux_hi = np.concatenate([[0], np.maximum(np.diff(np.log1p(S[band_hi]), axis=1), 0).sum(axis=0)])
    # clicks: isolated sample-level spikes of the second difference, far above its local (5 ms) level
    from scipy.ndimage import median_filter, maximum_filter1d
    d2 = np.abs(np.diff(y_hi, 2)).astype(np.float32)
    w = max(3, int(0.005 * sr_hi)) | 1
    local = median_filter(d2, size=w) + 1e-6
    spike = d2 > 25 * local
    isolated = spike & (maximum_filter1d(spike.astype(np.uint8), w) == 1) & (d2 > 0.02)
    clicks_t = (np.where(isolated)[0] + 1) / sr_hi
    fr = lambda t: int(t * sr / hop)  # noqa: E731
    rows, mf = [], []
    for s in segs:
        a, b = fr(s["start"]), max(fr(s["end"]), fr(s["start"]) + 2)
        c = cents[a:b]
        cv = c[np.isfinite(c)]
        r = {}
        if len(cv) >= 8:
            r["pitch_flat"] = -float(np.std(cv))
            r["jitter"] = float(np.mean(np.abs(np.diff(cv))))
        else:
            r["pitch_flat"], r["jitter"] = np.nan, np.nan
        dc = np.abs(np.diff(c))
        r["pitch_jump"] = float(np.sum(dc[np.isfinite(dc)] > 300)) if s["text"] != "(no word)" else 0.0
        r["flatness"] = float(np.mean(flat[a:b]))
        h, p = H[:, a:b].sum(), P[:, a:b].sum()
        r["low_hnr"] = -float(10 * np.log10((h + 1e-9) / (p + 1e-9)))
        spec = 20 * np.log10(S[band_ring, a:b].mean(axis=1) + 1e-9)
        r["ring"] = float(spec.max() - np.median(spec))
        r["harsh_onset"] = float(flux_hi[a:min(b, a + max(1, int(0.05 * sr / hop)))].max())
        r["low_prob"] = (1 - s["prob"]) if s["prob"] is not None else np.nan
        ct = clicks_t[(clicks_t >= s["start"]) & (clicks_t < s["end"])]
        seg_rms = librosa.feature.rms(y=y[int(s["start"] * sr):int(s["end"] * sr)], frame_length=220, hop_length=220)[0]
        drop = 0
        if len(seg_rms) > 4 and np.max(seg_rms) > 0:
            q = 20 * np.log10(seg_rms / np.max(seg_rms) + 1e-9) < -40
            drop = int(np.sum(q[1:-1] & ~q[:-2] & ~q[2:]))  # isolated silent 10 ms frames inside the segment
        r["glitch"] = float(len(ct) + drop)
        rows.append(r)
        mf.append(mfcc[:, a:b].mean(axis=1))
    for i in range(len(rows)):
        dists = [np.linalg.norm(mf[i] - mf[j]) for j in (i - 1, i + 1) if 0 <= j < len(rows)]
        rows[i]["timbre_jump"] = float(np.mean(dists)) if dists else np.nan
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vocal")
    ap.add_argument("lyrics")
    ap.add_argument("out_dir")
    ap.add_argument("--whisper")
    ap.add_argument("--mix")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out_dir, "clips"), exist_ok=True)
    wjson = args.whisper or os.path.join(args.out_dir, "whisper-words.json")
    if not os.path.exists(wjson):
        run_whisper(args.vocal, args.lyrics, wjson)
    words = [w for s in json.load(open(wjson)) for w in s["words"]]
    y_hi, sr_hi = sf.read(args.vocal, always_2d=True)
    y_hi = y_hi.mean(axis=1).astype(np.float32)
    y = librosa.resample(y_hi, orig_sr=sr_hi, target_sr=SR)
    segs = segments_from_words(words, y, SR)
    rows = features(y, SR, y_hi, sr_hi, segs)
    names = list(WEIGHTS)
    min_scale = {f: 1.0 for f in COUNT_FEATURES} | {"low_prob": 0.1}  # whisper's spread is tiny; 0.1 = 10 points
    Z = {f: robust_z([r[f] for r in rows], min_scale.get(f, 0.0)) for f in names}
    for k, (s, r) in enumerate(zip(segs, rows)):
        z = {f: float(Z[f][k]) for f in names}
        fired = [f for f in names if z[f] >= FIRE_Z]
        score = sum(WEIGHTS[f] * max(z[f], 0) for f in names)
        if score >= REPAIR_SCORE and len(fired) >= 2:
            lab = "repair-candidate"
        elif (score >= REVIEW_SCORE and len(fired) >= 2) or any(z[f] >= STRONG_Z for f in names):
            lab = "review"
        else:
            lab = "natural"
        s.update({"features": {f: (None if not np.isfinite(r[f]) else round(r[f], 4)) for f in names},
                  "z": {f: round(z[f], 2) for f in names}, "score": round(score, 2), "fired": fired, "label": lab})
    order = {"natural": 0, "review": 1, "repair-candidate": 2}
    regions, cur = [], None
    for s in segs:
        if s["label"] == "natural":
            cur = None
            continue
        if cur and s["start"] - cur["end"] < 0.25:
            cur["end"] = max(cur["end"], s["end"])
            cur["words"].append(s["text"])
            cur["score"] = max(cur["score"], s["score"])
            cur["fired"] = sorted(set(cur["fired"]) | set(s["fired"]))
            if order[s["label"]] > order[cur["label"]]:
                cur["label"] = s["label"]
        else:
            cur = {"start": s["start"], "end": s["end"], "words": [s["text"]], "score": s["score"],
                   "fired": list(s["fired"]), "label": s["label"]}
            regions.append(cur)
    for r in regions:
        top = r["fired"] or ["(several weak features)"]
        r["diagnosis"] = ", ".join(top)
        r["suggested_repair"] = "; ".join(sorted({REPAIRS[f] for f in r["fired"]})) or "listen first"
        r["start"], r["end"] = round(r["start"], 2), round(r["end"], 2)
    regions.sort(key=lambda r: -r["score"])
    # clips from the full mix: flagged regions + 10 random natural controls, shuffled, neutral names
    clips = []
    if args.mix:
        mix, msr = sf.read(args.mix, always_2d=True)
        rnd = random.Random(1234)
        nat = [s for s in segs if s["label"] == "natural" and s["text"] != "(no word)"]
        ctrl = rnd.sample(nat, min(10, len(nat)))
        items = [("region", r) for r in regions] + [("control", {"start": c["start"], "end": c["end"], "label": "natural",
                                                                   "words": [c["text"]], "score": c["score"]}) for c in ctrl]
        rnd.shuffle(items)
        for n, (kind, r) in enumerate(items, 1):
            mid = (r["start"] + r["end"]) / 2
            a = max(0, int((mid - 1.75) * msr))
            b = min(len(mix), a + int(3.5 * msr))
            name = f"clip-{n:02d}.wav"
            sf.write(os.path.join(args.out_dir, "clips", name), mix[a:b], msr, subtype="PCM_24")
            clips.append({"clip": name, "kind": kind, "label": r["label"], "start": r["start"], "end": r["end"],
                          "clip_start_s": round(a / msr, 2), "words": " ".join(r["words"]), "score": r["score"]})
        with open(os.path.join(args.out_dir, "clips", "labels.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["clip", "clip_start_mmss", "your_label (ok / odd / broken)", "what you hear"])
            for c in sorted(clips, key=lambda c: c["clip"]):
                w.writerow([c["clip"], f"{int(c['clip_start_s'] // 60):02d}:{c['clip_start_s'] % 60:04.1f}", "", ""])
        json.dump(clips, open(os.path.join(args.out_dir, "clips", "key.json"), "w"), indent=1)
    counts = {k: sum(1 for r in regions if r["label"] == k) for k in ("review", "repair-candidate")}
    counts["natural_segments"] = sum(1 for s in segs if s["label"] == "natural")
    rep = {"vocal": args.vocal, "mix": args.mix, "whisper": wjson, "weights": WEIGHTS,
           "thresholds": {"fire_z": FIRE_Z, "strong_z": STRONG_Z, "review_score": REVIEW_SCORE,
                          "repair_score": REPAIR_SCORE},
           "note": "heuristic; scores are unvalidated until the clips are labelled by ear",
           "counts": counts, "regions": regions, "segments": segs}
    json.dump(rep, open(os.path.join(args.out_dir, "qc-report.json"), "w"), indent=1, default=float)
    mmss = lambda t: f"{int(t // 60):02d}:{t % 60:04.1f}"  # noqa: E731
    lines = ["# Vocal QC (detection only)", "",
             "Heuristic and unvalidated: the scores mean \"unusual for this voice in this song\", not \"broken\".",
             "Listen to clips/ (neutral names) and fill in clips/labels.csv; clips/key.json has the answers.", "",
             f"Segments: {len(segs)}; regions: {counts['repair-candidate']} repair-candidate, {counts['review']} review.", "",
             "| time | words | label | score | diagnosis | suggested repair |", "|---|---|---|---|---|---|"]
    for r in sorted(regions, key=lambda r: r["start"]):
        lines.append(f"| {mmss(r['start'])}–{mmss(r['end'])} | {' '.join(r['words'])[:60]} | {r['label']} | "
                     f"{r['score']} | {r['diagnosis']} | {r['suggested_repair']} |")
    open(os.path.join(args.out_dir, "qc-report.md"), "w").write("\n".join(lines) + "\n")
    print(json.dumps({"counts": counts, "top": [{"t": mmss(r["start"]), "label": r["label"], "score": r["score"],
                                                 "diagnosis": r["diagnosis"], "words": " ".join(r["words"])[:40]}
                                                for r in regions[:3]], "clips": len(clips)}))


if __name__ == "__main__":
    main()
