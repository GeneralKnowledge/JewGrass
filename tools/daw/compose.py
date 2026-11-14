#!/usr/bin/env python3
"""Composition engine, first cut: Song Design + parts file -> arrangement.json.

    python3 tools/daw/compose.py docs/album/design/brass-plate.yaml docs/album/parts/brass-plate.yaml \\
        build/daw/brass-plate/arrangement.json

The design gives sections, bar counts, tempo, key, meter, dynamics and lyrics.
The parts file (docs/album/parts/<slug>.yaml) gives the exact chord chart per
section and, for each instrument, which generator writes it and where it plays.
Generators are small, deterministic functions (seeded, so the same input always
gives the same notes) that write one bar at a time from the chart.

Everything is authored, nothing is inferred from audio: every track gets
origin "composed".
"""
import os
import random
import re
import sys

import yaml

sys.path.insert(0, os.path.dirname(__file__))
import arrangement as A  # noqa: E402

DRUM = {"kick": 36, "snare": 38, "low_tom": 41, "hat": 42, "ride_bell": 53, "triangle": 81, "clap": 39}


# ------------------------------------------------------------------ chart helpers

def parse_slash(sym):
    chord, _, bass = sym.partition("/")
    root, ivs = A.parse_chord(chord)
    bass_pc = A.parse_chord(bass)[0] if bass else root
    return root, ivs, bass_pc


def place(pc, low):
    """Lowest pitch with pitch class pc at or above `low`."""
    return low + (pc - low) % 12


def chord_tones(sym, low):
    root, ivs, _ = parse_slash(sym)
    return sorted(place((root + iv) % 12, low) for iv in ivs)


def chart_for(charts, name):
    base = re.sub(r"\s*\d+$", "", name)
    for k in (name, base, base.split()[-1]):
        if k in charts:
            return charts[k]
    raise KeyError(f"no chart for section '{name}'")


def plays(spec, name):
    if spec in ("all", None):
        return True
    for p in spec:
        if p == name or (p.endswith("*") and re.sub(r"\s*\d+$", "", name) == p[:-1].strip()):
            return True
    return False


# ------------------------------------------------------------------ generators
# Each: gen(ctx) -> list of [start_beat_in_bar, dur, pitch, vel] for ONE bar.
# ctx: sym, next_sym, bar (index in section), bars, section, dyn, rng, param, is_last

def typewriter(c):
    out = []
    sparse = c["param"] == "sparse"
    step = 0.5 if sparse else 0.25
    n = int(4 / step)
    for i in range(n):
        if c["section"] == "intro" and c["bar"] >= 2 and i % 2:
            continue                                 # thin out when the harpsichord enters
        if c["rng"].random() < (0.25 if sparse else 0.12):
            continue                                 # a typist, not a machine
        v = (70 if i % 4 == 0 else 48) + c["rng"].randint(-8, 8) - (20 if sparse else 0)
        out.append([i * step, 0.1, DRUM["hat"], max(10, v)])
    if c["bar"] % 2 == 1 and not sparse:
        out.append([3.5, 0.5, DRUM["ride_bell"], 70])   # carriage-return bell
    return out


def office_drums(c):
    out = []
    if c["bar"] == 0:
        out.append([0, 1, DRUM["triangle"], 90])       # desk bell at each section start
    if c["section"] == "ending":
        return out + [[0, 1, DRUM["kick"], 120], [2, 1, DRUM["snare"], 120], [2, 1, DRUM["low_tom"], 120]]
    vel = 60 + 10 * c["dyn"]
    for b in (1, 3):                                  # the rubber stamp
        out += [[b, 0.25, DRUM["snare"], min(127, vel)], [b, 0.25, DRUM["low_tom"], min(127, vel - 10)]]
    kicks = [0, 2] if "chorus" not in c["section"] else [0, 1.5, 2, 3.5]
    out += [[k, 0.25, DRUM["kick"], min(127, vel + 5)] for k in kicks]
    return out


MONEY = [(0, 0.25, 0), (0.25, 0.25, 7), (0.5, 0.25, 1), (0.75, 1.25, 0)]   # C G Db C (from motifs.yaml)


def harpsichord(c):
    sym = c["sym"]
    t = chord_tones(sym, 60)
    root = place(parse_slash(sym)[0], 60)
    if c["section"] == "intro" and c["bar"] < 2:
        return []                                     # typewriter alone first
    if c["section"].startswith("verse") and c["bar"] % 8 == 7 and c["section"] != "verse 3":
        tonic = 72                                    # money flourish closes every 8 bars of verse
        riff = [[0, 0.45, root, 70], [0.5, 0.45, t[1], 55], [1, 0.45, t[2], 55], [1.5, 0.45, t[1], 50]]
        return riff + [[2 + s, d, tonic + iv, 90] for s, d, iv in MONEY]
    if c["section"] == "ending":
        return [[0, 3, p, 100] for p in chord_tones(sym, 60)]
    pat = [root, t[2], t[1], t[2], root + 12, t[2], t[1], t[2]]
    vel = 52 + 6 * c["dyn"]
    out = [[i * 0.5, 0.45, p, vel + (12 if i % 4 == 0 else 0)] for i, p in enumerate(pat)]
    out.insert(1, [0.25, 0.2, root + 2, vel - 10])     # little ornament after the downbeat
    if c["section"] == "verse 3":
        out = [n for n in out if n[0] in (0, 1, 2, 3)]   # he slows down
    return out


def march_bass(c):
    root, ivs, bass = parse_slash(c["sym"])
    b = place(bass, 31)                               # tuba range, G1..F#2
    vel = 70 + 6 * c["dyn"]
    if c["section"] == "ending":
        return [[0, 3, place(0, 31), 110]]
    if c["section"] == "verse 3":
        return [[0, 3.8, b, vel - 10]]                # held pedal C: the floor drops away
    fifth = place((root + 7) % 12, b - 4)               # chord's fifth, close to the bass note
    nxt = parse_slash(c["next_sym"])[2] if c["next_sym"] else bass
    target = place(nxt, 31)
    approach = target + (1 if target < b else -1) if target != b else b + 2
    return [[0, 0.9, b, vel], [1, 0.4, b, vel - 25], [2, 0.9, fifth, vel - 8], [3, 0.9, approach, vel - 12]]


def pizzicato(c):
    t = chord_tones(c["sym"], 55)
    vel = 45 + 8 * c["dyn"]
    if c["section"] == "verse 3":
        return [[i, 0.3, t[i % 2], 50 + c["rng"].randint(-5, 5)] for i in range(4)]
    seq = [t[0], t[1], t[0], t[2]] * 2
    return [[i * 0.5, 0.2, p, vel + (10 if i % 2 == 0 else 0)] for i, p in enumerate(seq)]


def brass_stabs(c):
    tones = chord_tones(c["sym"], 58) + [place(parse_slash(c["sym"])[0], 58) + 12]
    if c["section"] == "ending":
        return [[0, 3.5, p, 115] for p in tones + [tones[0] - 12]]
    vel = 70 + 8 * c["dyn"]
    out = [[0, 0.35, p, min(127, vel)] for p in tones]
    if c["section"] == "final chorus" and c["bar"] % 2 == 1:
        out += [[2.5, 0.35, p, min(127, vel - 10)] for p in tones]
    return out


def sub_hits(c):
    root = parse_slash(c["sym"])[0]
    return [[0, 1.5, place(root, 24), 110]]


def synth_stabs(c):
    if c["bar"] % 4 != 3:
        return []
    top = chord_tones(c["sym"], 72)
    return [[3.5, 0.25, p, 95] for p in top[-2:]]


GENERATORS = {f.__name__: f for f in (typewriter, office_drums, harpsichord, march_bass, pizzicato,
                                      brass_stabs, sub_hits, synth_stabs)}


# ------------------------------------------------------------------ build

def build(design, parts):
    r = design["render"]
    ts = {"2": (2, 4), "3": (3, 4), "4": (4, 4), "6": (6, 8)}[str(r["timesig"])]
    if ts != (4, 4):
        raise SystemExit("compose.py generators are written for 4/4 so far")
    arr = A.new(design["slug"], design["title"], r["bpm"], ts, r["key"])
    rng = random.Random(parts.get("seed", 0))
    tracks = []
    for p in parts["parts"]:
        drums = p.get("program") in (None, "drums")
        tracks.append({"name": p["name"], "role": p["generator"], "origin": "composed",
                       "instrument": {"name": p["name"].lower(), "program": 0 if drums else int(p["program"]),
                                      "drums": drums},
                       "clips": [], "notes": f"generator {p['generator']}"})
    bar = 0
    for s in design["sections"]:
        n = int(s["bars"])
        chart = chart_for(parts["charts"], s["name"])
        seq = [chart[i % len(chart)] for i in range(n)]
        start = A.bar_to_beat(arr, bar)
        arr["sections"].append({"name": s["name"], "start_bar": bar, "bars": n,
                                "chords": [[start + 4 * i, c] for i, c in enumerate(seq)],
                                "lyrics": (s.get("lyrics") or "").strip(), "dynamics": s.get("dynamics")})
        nxt_section_first = None
        idx = design["sections"].index(s)
        if idx + 1 < len(design["sections"]):
            nxt_section_first = chart_for(parts["charts"], design["sections"][idx + 1]["name"])[0]
        for p, tr in zip(parts["parts"], tracks):
            if not plays(p.get("play", "all"), s["name"]):
                continue
            gen = GENERATORS[p["generator"]]
            param = (p.get("params") or {}).get(s["name"])
            notes = []
            for i, sym in enumerate(seq):
                ctx = {"sym": sym, "next_sym": seq[i + 1] if i + 1 < n else nxt_section_first, "bar": i, "bars": n,
                       "section": s["name"], "dyn": int(s.get("dynamics") or 3), "rng": rng, "param": param}
                notes += [[4 * i + a, d, int(pp), int(max(1, min(127, v)))] for a, d, pp, v in gen(ctx)]
            if notes:
                tr["clips"].append({"name": s["name"], "start": start, "length": 4.0 * n, "notes": notes})
        bar += n
    arr["tracks"] = tracks
    arr["provenance"] = {"route": "composition", "source": f"docs/album/design/{design['slug']}.yaml + "
                         f"docs/album/parts/{design['slug']}.yaml", "seed": parts.get("seed", 0)}
    return arr


if __name__ == "__main__":
    design, parts, out = sys.argv[1:4]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    arr = build(yaml.safe_load(open(design)), yaml.safe_load(open(parts)))
    A.save(arr, out)
    print(f"{out}: {sum(len(c['notes']) for t in arr['tracks'] for c in t['clips'])} notes, "
          f"{A.beat_to_sec(arr, A.total_beats(arr)):.0f} s")
    for t in arr["tracks"]:
        print(f"  {t['name']:<18} {', '.join(c['name'] for c in t['clips'])}")
