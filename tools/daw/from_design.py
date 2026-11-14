#!/usr/bin/env python3
"""Composition route: Song Design yaml -> arrangement.json.

    python3 tools/daw/from_design.py docs/album/design/gavs-song.yaml build/daw/gavs-song/arrangement.json

Chords: if a section has `chords: "Bm G D A | Bm G A A"` (one chord per bar,
bars cycle if short) that exact chart is used. Otherwise the free-text
`harmony` block is read as best it can be (chord symbols in order, "last line"
phrases go at the end of the section), which is approximate.

What it makes, all from the design and nothing from audio:
  * sections with bar positions, dynamics and lyrics (from `sections`)
  * a chord track (sustained triads, one chord per bar, cycled from `harmony`)
  * a bass track (root and fifth, in the song's meter)
  * a fingerpicked guitar track (arpeggios of the same chords)

This is a sketch, not a finished part: it proves the design can be played as
editable MIDI. Instrument entries/exits from `roles` are not applied yet.
Section lengths come from the design's bar counts, so the MIDI follows the
written plan, not the timing of any ACE-Step take.
"""
import re
import sys

import yaml

sys.path.insert(0, __import__("os").path.dirname(__file__))
import arrangement as A  # noqa: E402

TIMESIG = {"2": (2, 4), "3": (3, 4), "4": (4, 4), "6": (6, 8)}


def chord_tokens(text):
    text = re.sub(r"\(.*?\)", "", str(text))
    return [t for t in re.findall(r"[A-G][#b]?(?:maj7|m7|m|7|sus2|sus4|dim|5)?(?![a-z])", text)]


def section_chords(design, name, key):
    """Chord list for a section, from the design's free-text harmony."""
    h = {k.replace("_", " "): v for k, v in (design.get("harmony") or {}).items()}
    base = re.sub(r"\s*\d+$", "", name)              # 'verse 1' -> 'verse'
    text = h.get(name) or h.get(base) or h.get(base.split()[-1])   # 'final chorus' -> 'chorus'
    if text is None and base == "outro":
        text = h.get("ending")
    if text is None:
        tonic = key.split()[0] + ("m" if "minor" in key else "")
        return [tonic], []
    if str(text).startswith("as "):                   # 'as chorus, but the last line resolves to D'
        ref = str(text)[3:].split(",")[0].strip()
        loop, tail = section_chords(design, ref, key)
        extra = chord_tokens(str(text).split(",", 1)[1]) if "," in str(text) else []
        return loop, (tail[:-len(extra)] + extra) if (tail and extra) else (extra or tail)
    loop, tail = [], []
    for phrase in re.sub(r"\(.*?\)", "", str(text)).split(","):
        toks = chord_tokens(phrase)
        (tail if "last line" in phrase else loop).extend(toks)
    return loop or tail, tail


def voicing(sym, centre=60):
    root, ivs = A.parse_chord(sym)
    notes = []
    for iv in ivs[:3] if len(ivs) > 3 else ivs:
        p = centre - 7 + ((root + iv - (centre - 7)) % 12)
        notes.append(p)
    return sorted(notes)


def build(design):
    r = design["render"]
    ts = TIMESIG[str(r["timesig"])]
    arr = A.new(design["slug"], design["title"], r["bpm"], ts, r["key"])
    bpb = A.beats_per_bar(arr)
    eighth = 0.5
    chords_t = {"name": "Chords", "role": "harmony", "origin": "design",
                "instrument": {"name": "chord guide", "program": 48, "drums": False}, "clips": [],
                "notes": "One chord per bar, cycled from the design's harmony text. A guide, not a part."}
    bass_t = {"name": "Bass", "role": "bass", "origin": "design",
              "instrument": {"name": "upright bass", "program": 32, "drums": False}, "clips": [],
              "notes": "Root on the downbeat, fifth on the second pulse."}
    gtr_t = {"name": "Guitar", "role": "lead", "origin": "design",
             "instrument": {"name": "fingerpicked acoustic guitar", "program": 25, "drums": False}, "clips": [],
             "notes": "Eighth-note arpeggio of each chord."}

    bar = 0
    for s in design["sections"]:
        n = int(s["bars"])
        if s.get("chords"):                          # optional exact chart: one chord per bar
            seq = chord_tokens(s["chords"])
            seq = [seq[i % len(seq)] for i in range(n)]
        else:                                        # fallback: read the free-text harmony
            loop, tail = section_chords(design, s["name"], r["key"])
            seq = [loop[i % len(loop)] for i in range(n)]
            if tail and len(tail) < n:
                seq[n - len(tail):] = tail
        vel = 46 + 12 * int(s.get("dynamics", 3))
        start = A.bar_to_beat(arr, bar)
        sec = {"name": s["name"], "start_bar": bar, "bars": n,
               "chords": [[start + i * bpb, c] for i, c in enumerate(seq)],
               "lyrics": (s.get("lyrics") or "").strip(), "dynamics": s.get("dynamics")}
        arr["sections"].append(sec)
        cn, bn, gn = [], [], []
        for i, sym in enumerate(seq):
            b0 = i * bpb
            v = voicing(sym, 62)
            cn += [[b0, bpb * 0.98, p, max(1, vel - 20)] for p in v]
            root = 28 + (A.parse_chord(sym)[0] - 28) % 12    # chord root, E1..D#2
            fifth = root + 7
            if ts == (6, 8):
                bn += [[b0, 1.4, root, vel], [b0 + 1.5, 1.4, fifth if i % 2 else root, vel - 8]]
                g = voicing(sym, 55)
                pat = [root + 12, g[1], g[2], g[0] + 12, g[2], g[1]]
            else:
                half = bpb / 2
                bn += [[b0, half * 0.95, root, vel], [b0 + half, half * 0.95, fifth, vel - 8]]
                g = voicing(sym, 55)
                pat = [root + 12, g[1], g[2], g[0] + 12, g[2], g[1], g[0] + 12, g[2]][: int(bpb / eighth)]
            for k, p in enumerate(pat):
                gn.append([b0 + k * eighth, eighth * 1.8, p, vel - (0 if k == 0 else 12)])
        for t, notes in ((chords_t, cn), (bass_t, bn), (gtr_t, gn)):
            t["clips"].append({"name": s["name"], "start": start, "length": n * bpb, "notes": notes})
        bar += n

    arr["tracks"] = [gtr_t, bass_t, chords_t]
    arr["provenance"] = {"route": "composition", "source": f"docs/album/design/{design['slug']}.yaml",
                         "design_version": design.get("version", 1)}
    return arr


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    import os
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    arr = build(yaml.safe_load(open(src)))
    A.save(arr, out)
    secs = A.beat_to_sec(arr, A.total_beats(arr))
    print(f"{out}: {len(arr['sections'])} sections, {len(arr['tracks'])} tracks, {secs:.0f} s at {arr['tempo_bpm']:g} bpm")
    for s in arr["sections"]:
        print(f"  bar {s['start_bar'] + 1:>3}  {s['name']:<13} {' '.join(c for _, c in s['chords'])}")
