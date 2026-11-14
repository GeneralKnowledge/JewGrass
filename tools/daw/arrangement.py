"""JewGrass arrangement: the DAW-independent description of a song.

Every route into a DAW goes through this one format:

    Song Design (yaml) --from_design.py-->  arrangement.json  --export.py--> REAPER .rpp
    stems (audio)      --from_audio.py--->       (this)                  --> multitrack + per-track MIDI
                                                                         --> audio package, lyrics, sections

Time is in quarter-note beats from the downbeat of bar 1 (beat 0), so the
format never depends on a DAW's tick resolution. Seconds are derived from the
tempo. The schema is plain JSON:

{
  "format": "jewgrass.arrangement/1",
  "slug": "gavs-song", "title": "Gav's Song",
  "tempo_bpm": 76,                 # quarter notes per minute
  "time_signature": [6, 8],
  "key": "D major",
  "audio_offset_seconds": 0.0,     # where beat 0 falls inside the reference audio
  "sections": [ {"name", "start_bar" (0-based), "bars", "chords": [[beat, "Bm"], ...],
                 "lyrics", "dynamics"} ],
  "tracks": [ {"name", "role", "origin": "design" | "transcribed" | "audio",
               "instrument": {"name", "program" (GM 0-127), "drums": bool},
               "clips": [ {"name", "start", "length", "notes": [[start, dur, pitch, vel], ...]} ],
               "audio": "relative/path.mp3"  # audio tracks only
               "notes": "free text: how this track was made, what to trust" } ],
  "reference_audio": "relative/path.mp3" | null,
  "provenance": { ... }            # what made it, with which settings
}

Note starts inside a clip are relative to the clip start.
"""
import json

FORMAT = "jewgrass.arrangement/1"


def beats_per_bar(arr):
    num, den = arr["time_signature"]
    return num * 4.0 / den


def bar_to_beat(arr, bar):
    return bar * beats_per_bar(arr)


def beat_to_sec(arr, beat):
    return beat * 60.0 / arr["tempo_bpm"]


def sec_to_beat(arr, sec):
    return sec * arr["tempo_bpm"] / 60.0


def total_beats(arr):
    end = 0.0
    for s in arr["sections"]:
        end = max(end, bar_to_beat(arr, s["start_bar"] + s["bars"]))
    for t in arr["tracks"]:
        for c in t.get("clips", []):
            end = max(end, c["start"] + c["length"])
    return end


def new(slug, title, bpm, time_signature=(4, 4), key=None):
    return {
        "format": FORMAT,
        "slug": slug,
        "title": title,
        "tempo_bpm": float(bpm),
        "time_signature": list(time_signature),
        "key": key,
        "audio_offset_seconds": 0.0,
        "sections": [],
        "tracks": [],
        "reference_audio": None,
        "provenance": {},
    }


def validate(arr):
    problems = []
    if arr.get("format") != FORMAT:
        problems.append(f"format should be {FORMAT}")
    if not 20 <= arr["tempo_bpm"] <= 300:
        problems.append("tempo_bpm out of range")
    names = set()
    for t in arr["tracks"]:
        if t["name"] in names:
            problems.append(f"duplicate track name {t['name']}")
        names.add(t["name"])
        for c in t.get("clips", []):
            for n in c["notes"]:
                s, d, p, v = n
                if not (0 <= p <= 127 and 1 <= v <= 127 and d > 0 and s >= 0):
                    problems.append(f"{t['name']}/{c['name']}: bad note {n}")
                    break
    return problems


def save(arr, path):
    probs = validate(arr)
    if probs:
        raise ValueError("; ".join(probs))
    with open(path, "w") as f:
        json.dump(arr, f, indent=1)


def load(path):
    with open(path) as f:
        arr = json.load(f)
    probs = validate(arr)
    if probs:
        raise ValueError("; ".join(probs))
    return arr


# --- small music helpers shared by the routes ---------------------------------

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_QUALITY = {
    "": [0, 4, 7], "m": [0, 3, 7], "7": [0, 4, 7, 10], "m7": [0, 3, 7, 10],
    "maj7": [0, 4, 7, 11], "sus2": [0, 2, 7], "sus4": [0, 5, 7], "dim": [0, 3, 6],
    "5": [0, 7],
}


def parse_chord(sym):
    """'Bm' -> (11, [0, 3, 7]). Returns None if it isn't a chord symbol."""
    import re
    m = re.fullmatch(r"([A-G])([#b]?)(m7|maj7|m|7|sus2|sus4|dim|5)?", sym)
    if not m:
        return None
    root = _PC[m.group(1)] + {"#": 1, "b": -1, "": 0}[m.group(2)]
    return root % 12, _QUALITY[m.group(3) or ""]


def pitch_name(p):
    return f"{NOTE_NAMES[p % 12]}{p // 12 - 1}"
