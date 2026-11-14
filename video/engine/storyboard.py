"""Storyboard loading and validation.

storyboard.yaml (written by the creative director, reviewed by a human):
  song, style, preset, seed, motifs[], shots[]
  shot: id, at (a section name) or bars: [first, last_exclusive],
        plate, camera, weather, motifs, text, react, grade, transition

timing.yaml (from alignment; hand-fixable):
  sections: [{name, start_bar, bars}]
  lines:    [{section, text, words: [[time_s, word], ...]}]

Rules enforced (docs/video/video-engine.md section 3, stage 2):
  * every shot starts and ends on a bar line, shots don't overlap, and
    together they cover the whole song;
  * each shot is between MIN_BARS and MAX_BARS long (unless the song section
    itself is shorter);
  * every motif used is declared; every lyric line falls in a shot with text.
"""
import yaml

MIN_BARS, MAX_BARS = 2, 16
KNOWN_TEXT_MODES = {"chorus-sign", "spoken-title", "slam", "quiet", "voicemail", "none"}
KNOWN_MOVES = {"push-in", "pull-out", "drift-left", "drift-right", "hold", "rise"}


class StoryboardError(ValueError):
    pass


def load(storyboard_path, timing_path, analysis):
    sb = yaml.safe_load(open(storyboard_path))
    tm = yaml.safe_load(open(timing_path))
    return resolve(sb, tm, analysis)


def bar_time(analysis, bar):
    db = analysis["downbeats"]
    if bar < len(db):
        return db[bar]
    # extrapolate past the last detected downbeat
    period = (db[-1] - db[0]) / max(len(db) - 1, 1)
    return db[-1] + (bar - len(db) + 1) * period


def resolve(sb, tm, analysis):
    errors = []
    sections = {s["name"]: s for s in tm["sections"]}
    total_bars = max(s["start_bar"] + s["bars"] for s in tm["sections"])
    declared = set(sb.get("motifs", []))
    shots = []
    for sh in sb["shots"]:
        if "at" in sh:
            if sh["at"] not in sections:
                errors.append(f"{sh['id']}: unknown section {sh['at']!r}")
                continue
            s = sections[sh["at"]]
            b0, b1 = s["start_bar"], s["start_bar"] + s["bars"]
            if "split" in sh:  # e.g. split: [0, 4] -> bars 0..4 of the section
                b0, b1 = s["start_bar"] + sh["split"][0], s["start_bar"] + sh["split"][1]
        else:
            b0, b1 = sh["bars"]
        n = b1 - b0
        if n < MIN_BARS or n > MAX_BARS:
            errors.append(f"{sh['id']}: {n} bars (allowed {MIN_BARS}-{MAX_BARS})")
        cam = sh.get("camera", {})
        if cam.get("move", "hold") not in KNOWN_MOVES:
            errors.append(f"{sh['id']}: unknown camera move {cam.get('move')!r}")
        mode = (sh.get("text") or {}).get("mode", "none")
        if mode not in KNOWN_TEXT_MODES:
            errors.append(f"{sh['id']}: unknown text mode {mode!r}")
        for m in (sh.get("motifs") or {}):
            if m not in declared:
                errors.append(f"{sh['id']}: motif {m!r} not declared in motifs:")
        shots.append(dict(sh, b0=b0, b1=b1, t0=bar_time(analysis, b0), t1=bar_time(analysis, b1)))

    shots.sort(key=lambda s: s["b0"])
    pos = 0
    for s in shots:
        if s["b0"] != pos:
            errors.append(f"{s['id']}: starts at bar {s['b0']}, expected {pos} (gap or overlap)")
        pos = s["b1"]
    if pos != total_bars:
        errors.append(f"shots end at bar {pos}, song has {total_bars} bars")

    # The first shot starts at 0 s and the last ends at the end of the audio.
    if shots:
        shots[0]["t0"] = 0.0
        shots[-1]["t1"] = analysis["duration"]

    lines = tm.get("lines", [])
    for ln in lines:
        if not ln.get("words"):
            continue
        t = ln["words"][0][0]
        hit = [s for s in shots if s["t0"] <= t < s["t1"]]
        if hit and (hit[0].get("text") or {}).get("mode", "none") == "none":
            errors.append(f"line {ln['text'][:30]!r} at {t:.1f}s has no text treatment (shot {hit[0]['id']})")
    if errors:
        raise StoryboardError("storyboard problems:\n  " + "\n  ".join(errors))
    return {"meta": {k: v for k, v in sb.items() if k != "shots"}, "shots": shots,
            "lines": lines, "sections": tm["sections"]}
