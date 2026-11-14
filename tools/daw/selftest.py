#!/usr/bin/env python3
"""Synthetic transcription test: notes we know -> audio -> Basic Pitch -> score.

    python3 tools/daw/selftest.py build/daw/gavs-song/arrangement.json build/daw/selftest

Renders each MIDI track of an arrangement with the built-in synth (a perfectly
separated "stem"), plus the full mix, transcribes them back, and scores the
result against the original notes: a note counts as found when the pitch is
exact and the onset is within 50 ms (the usual MIREX rule, offsets ignored).

This measures the transcriber at its best case (clean, synthetic, one
instrument). Real stems from ACE-Step will score lower; this is the ceiling.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import arrangement as A  # noqa: E402
import from_audio  # noqa: E402
import synth  # noqa: E402


def truth(arr, track):
    return sorted((A.beat_to_sec(arr, c["start"] + s), p) for c in track["clips"] for s, d, p, v in c["notes"])


def score(ref, est, tol=0.05, pc=False):
    used = [False] * len(est)
    hit = 0
    j0 = 0
    for t, p in ref:
        while j0 < len(est) and est[j0][0] < t - tol:
            j0 += 1
        j = j0
        while j < len(est) and est[j][0] <= t + tol:
            ok = (est[j][1] % 12 == p % 12) if pc else (est[j][1] == p)
            if ok and not used[j]:
                used[j] = True
                hit += 1
                break
            j += 1
    prec = hit / len(est) if est else 0.0
    rec = hit / len(ref) if ref else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"ref": len(ref), "est": len(est), "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}


def main(arr_path, out_dir):
    arr = A.load(arr_path)
    os.makedirs(out_dir, exist_ok=True)
    report, mix, all_ref = {}, None, []
    for tr in arr["tracks"]:
        if not tr.get("clips"):
            continue
        y = synth.render_track(arr, tr)
        mix = y if mix is None else mix + y
        wav = os.path.join(out_dir, f"{tr['role']}.wav")
        synth.write_wav(wav, y)
        role = {"bass": "bass", "lead": "guitar"}.get(tr["role"], "piano")
        _, _, fmin, fmax = from_audio.ROLES[role]
        est = sorted((s, p) for s, e, p, v in from_audio.transcribe_pitched(wav, fmin, fmax, *from_audio.THRESHOLDS[role]))
        ref = truth(arr, tr)
        all_ref += ref
        report[tr["name"]] = {"exact": score(ref, est), "pitch_class": score(ref, est, pc=True)}
        print(tr["name"], report[tr["name"]])
    wav = os.path.join(out_dir, "mix.wav")
    synth.write_wav(wav, mix)
    est = sorted((s, p) for s, e, p, v in from_audio.transcribe_pitched(wav, 27, 4200, *from_audio.THRESHOLDS["piano"]))
    report["Full mix, no separation"] = {"exact": score(sorted(all_ref), est),
                                         "pitch_class": score(sorted(all_ref), est, pc=True)}
    print("mix", report["Full mix, no separation"])
    json.dump(report, open(os.path.join(out_dir, "report.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
