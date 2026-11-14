#!/usr/bin/env python3
"""Reconstruction route: separated stems -> arrangement.json.

    python3 tools/daw/from_audio.py OUT.json --title "Deed to the Glen" --slug deed-to-the-glen \\
        --ref mix.mp3 --bpm 140 --stem bass=bass.mp3 --stem other=other.mp3 --stem drums=drums.mp3 \\
        [--stem vocals=vocals.mp3] [--sections sections.json] [--lyrics lyrics.txt] [--grid 16]

Separation is NOT done here (the laptop does it, one heavy job at a time).
This step is CPU-only and light: Basic Pitch (Spotify, Apache-2.0) for pitched
stems, a three-band onset detector for drums (kick / snare / hat), librosa for
the beat grid. Each stem keeps its audio on the track too, so the DAW shows the
original next to the transcription.

Tempo: --bpm fixes the tempo (use the bpm the song was generated at), and the
first detected beat becomes beat 0 (bar 1). Without --bpm it is estimated.
ACE-Step holds tempo well, so one constant tempo is assumed.

--grid N snaps note starts to 1/N notes (16 = sixteenths). Default: no snapping,
the performance timing is kept.

The result is an editable approximation, never the exact performance.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import arrangement as A  # noqa: E402

# role -> (instrument name, GM program, min Hz, max Hz)
ROLES = {
    "bass": ("bass", 33, 30, 400),
    "guitar": ("guitar", 25, 75, 1400),
    "piano": ("keys", 0, 27, 4200),
    "other": ("other (banjo / fiddle / pipes)", 110, 100, 3000),
    "vocals": ("vocal melody guide", 53, 80, 1100),
}
# role -> Basic Pitch (onset threshold, frame threshold, min note ms). Bass and guitar
# were tuned on the synthetic test (selftest.py); the defaults (0.5, 0.3, 80) roughly
# doubled the note count with ghost notes. The others are untested guesses.
THRESHOLDS = {"bass": (0.7, 0.4, 120), "guitar": (0.8, 0.5, 150), "piano": (0.7, 0.4, 120),
              "other": (0.6, 0.4, 100), "vocals": (0.6, 0.4, 150)}
DRUM_BANDS = [("kick", 36, 30, 150), ("snare", 38, 180, 2500), ("hat", 42, 6000, 15000)]


def detect_beats(path, bpm=None):
    import librosa
    y, sr = librosa.load(path, sr=22050, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, start_bpm=bpm or 120, tightness=400 if bpm else 100,
                                           units="time")
    tempo = float(np.atleast_1d(tempo)[0])
    if bpm:
        # keep the given tempo, find the phase that fits the detected beats best
        period = 60.0 / bpm
        phases = np.mod(beats, period)
        ang = phases / period * 2 * np.pi
        phase = (np.angle(np.mean(np.exp(1j * ang))) % (2 * np.pi)) / (2 * np.pi) * period
        first = beats[0] if len(beats) else 0.0
        offset = phase + period * np.floor((first - phase) / period + 0.5)
        return float(bpm), float(max(0.0, offset))
    return round(tempo, 2), float(beats[0] if len(beats) else 0.0)


def transcribe_pitched(path, fmin, fmax, onset=0.5, frame=0.3, min_ms=80):
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict
    _, _, events = predict(path, ICASSP_2022_MODEL_PATH, onset_threshold=onset, frame_threshold=frame,
                           minimum_note_length=min_ms, minimum_frequency=fmin, maximum_frequency=fmax,
                           melodia_trick=True)
    return [(float(s), float(e), int(p), int(np.clip(round(a * 127), 1, 127))) for s, e, p, a, *_ in events]


def transcribe_drums(path):
    import librosa
    y, sr = librosa.load(path, sr=44100, mono=True)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    out = []
    for _, pitch, lo, hi in DRUM_BANDS:
        band = S[(freqs >= lo) & (freqs < hi)]
        env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(band), sr=sr, hop_length=512)
        if env.max() <= 0:
            continue
        env = env / env.max()
        frames = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=512, delta=0.12, wait=3,
                                            backtrack=False)
        for f in frames:
            t = librosa.frames_to_time(f, sr=sr, hop_length=512)
            out.append((float(t), float(t) + 0.08, pitch, int(np.clip(30 + 97 * env[f], 1, 127))))
    return sorted(out)


def to_clips(arr, events, grid=None):
    """Seconds -> beats relative to beat 0, then one clip per section (or per 8 bars)."""
    off = arr["audio_offset_seconds"]
    notes = []
    for s, e, p, v in events:
        b0, b1 = A.sec_to_beat(arr, s - off), A.sec_to_beat(arr, e - off)
        if grid:
            q = 4.0 / grid
            b0 = round(b0 / q) * q
            b1 = max(b0 + q, round(b1 / q) * q)
        if b0 < 0:
            continue                    # before bar 1: dropped (the lead-in keeps the audio, not these notes)
        notes.append([round(b0, 4), round(max(b1 - b0, 0.05), 4), p, v])
    bounds = [(s["name"], A.bar_to_beat(arr, s["start_bar"]), A.bar_to_beat(arr, s["start_bar"] + s["bars"]))
              for s in arr["sections"]]
    if not bounds:
        end = max((n[0] + n[1] for n in notes), default=0)
        span = A.bar_to_beat(arr, 8)
        bounds = [(f"bars {int(i / A.beats_per_bar(arr)) + 1}-", i, i + span) for i in np.arange(0, end + 1e-6, span)]
    clips = []
    for name, a, b in bounds:
        inside = [[round(n[0] - a, 4), round(min(n[1], b - n[0]), 4), n[2], n[3]] for n in notes if a <= n[0] < b]
        if inside:
            clips.append({"name": name, "start": a, "length": b - a, "notes": inside})
    return clips


def sections_from_seconds(arr, rows):
    """[{name, start, end}] in seconds -> bar-aligned sections."""
    bpb = A.beats_per_bar(arr)
    out = []
    for r in rows:
        a = round(A.sec_to_beat(arr, r["start"] - arr["audio_offset_seconds"]) / bpb)
        b = round(A.sec_to_beat(arr, r["end"] - arr["audio_offset_seconds"]) / bpb)
        if b > a >= 0:
            out.append({"name": r["name"], "start_bar": int(a), "bars": int(b - a), "chords": [],
                        "lyrics": r.get("lyrics", ""), "dynamics": None})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--title", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--ref", help="reference mix (also used for the beat grid)")
    ap.add_argument("--bpm", type=float)
    ap.add_argument("--timesig", default="4/4")
    ap.add_argument("--key")
    ap.add_argument("--stem", action="append", default=[], help="role=path; roles: drums " + " ".join(ROLES))
    ap.add_argument("--transcribe", default="bass,guitar,piano,other,drums",
                    help="roles to turn into MIDI (others stay audio only)")
    ap.add_argument("--sections", help="json list of {name, start, end[, lyrics]} in seconds")
    ap.add_argument("--grid", type=int)
    a = ap.parse_args()

    num, den = map(int, a.timesig.split("/"))
    arr = A.new(a.slug, a.title, a.bpm or 120, (num, den), a.key)
    out_dir = os.path.dirname(os.path.abspath(a.out))
    rel = lambda p: os.path.relpath(os.path.abspath(p), out_dir)  # noqa: E731
    beat_src = a.ref or dict(s.split("=", 1) for s in a.stem).get("drums")
    if beat_src:
        bpm, off = detect_beats(beat_src, a.bpm)
        num_, den_ = map(int, a.timesig.split("/"))
        off %= (60.0 / bpm) * num_ * 4.0 / den_          # earliest bar line in the file: intros stay on the grid
        arr["tempo_bpm"], arr["audio_offset_seconds"] = bpm, off
        print(f"tempo {bpm:g} bpm, beat 0 at {off:.3f} s")
    if a.ref:
        arr["reference_audio"] = rel(a.ref)
    if a.sections:
        arr["sections"] = sections_from_seconds(arr, json.load(open(a.sections)))

    wanted = set(a.transcribe.split(","))
    for spec in a.stem:
        role, path = spec.split("=", 1)
        if role == "drums":
            name, prog = "Drums", 0
            inst = {"name": "drum kit (kick 36, snare 38, hat 42)", "program": 0, "drums": True}
        else:
            iname, prog, fmin, fmax = ROLES[role]
            name = role.capitalize()
            inst = {"name": iname, "program": prog, "drums": False}
        track = {"name": name, "role": role, "origin": "transcribed" if role in wanted else "audio",
                 "instrument": inst, "clips": [], "audio": rel(path)}
        if role in wanted:
            ev = transcribe_drums(path) if role == "drums" else \
                transcribe_pitched(path, fmin, fmax, *THRESHOLDS[role])
            track["clips"] = to_clips(arr, ev, a.grid)
            n = sum(len(c["notes"]) for c in track["clips"])
            track["notes"] = (f"{n} notes, Basic Pitch" if role != "drums" else f"{n} hits, band onsets") + \
                (f", snapped to 1/{a.grid}" if a.grid else ", unquantised")
            print(f"{name}: {n} notes")
        arr["tracks"].append(track)
    arr["provenance"] = {"route": "reconstruction", "source": a.ref or "stems", "grid": a.grid,
                         "transcriber": "basic-pitch 0.4 (ICASSP 2022 model)"}
    A.save(arr, a.out)
    print(a.out)


if __name__ == "__main__":
    main()
