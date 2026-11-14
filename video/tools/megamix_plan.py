"""Plan a lyric video for a megamix: one shot per song excerpt, a title card as
each song comes in, lyrics only where the recogniser confirms them.

  python tools/megamix_plan.py SONG_DIR --tracks tracks.yaml --lyrics-md lyrics.md [--album "Natural Capital"]

Reads SONG_DIR/media/: cues.json [{title, start_s, end_s, source_file,
source_offset_s, ...}], beats.json, optional whisper-words.json, master.*.
Writes SONG_DIR/timing.yaml, SONG_DIR/grid.json and SONG_DIR/storyboard.yaml
(the storyboard only if it doesn't exist yet, so hand edits survive).

Lyrics for each song come from video/songs/<slug>/media/lyrics-performed.txt
or lyrics.txt if we have them, then tracks.yaml, then the ``` block under the
song's heading in docs/lyrics.md. A line is shown only if most of its words
were heard inside that song's excerpt: a megamix cuts songs mid-verse, and a
wrong line on screen is worse than none.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from align_lyrics import bars_from_beats  # noqa: E402
from whisper_timing import load_whisper, norm, sim  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def slugify(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def lyric_lines(text):
    out = []
    for raw in text.splitlines():
        ln = raw.strip()
        if not ln or ln.startswith(("#", "```")) or re.fullmatch(r"\[[^\]]*\]", ln):
            continue
        if ln.startswith("(") and ln.endswith(")"):
            ln = ln[1:-1]  # sung asides
        out.append(ln)
    return out


def find_lyrics(slug, title, tracks, lyrics_md):
    for name in ("lyrics-performed.txt", "lyrics.txt"):
        p = os.path.join(HERE, "songs", slug, "media", name)
        if os.path.exists(p):
            return lyric_lines(open(p, encoding="utf-8").read())
    for t in tracks:
        if (t.get("slug") == slug or t.get("title", "").lower() == title.lower()) and t.get("lyrics"):
            return lyric_lines(t["lyrics"])
    if lyrics_md:
        md = open(lyrics_md, encoding="utf-8").read()
        m = re.search(r"^### " + re.escape(title) + r"\b.*?$(.*?)(?=^### |\Z)", md, re.M | re.S | re.I)
        if m:
            # the lyrics are the longest fenced block (others are captions)
            blocks = re.findall(r"```(.*?)```", m.group(1), re.S)
            if blocks:
                return lyric_lines(max(blocks, key=lambda b: b.count("\n")))
    return []


def local_align(lyr, heard, gap=-0.6, min_sim=0.6):
    """Like whisper_timing.align_words, but skipping lyric words before and
    after the excerpt is free (the mix starts and ends mid-song)."""
    n, m = len(lyr), len(heard)
    S = np.zeros((n + 1, m + 1), np.float32)
    B = np.zeros((n + 1, m + 1), np.int8)
    S[0, 1:] = gap * 0.5 * np.arange(1, m + 1)
    B[1:, 0], B[0, 1:] = 1, 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = sim(lyr[i - 1], heard[j - 1][2])
            diag = S[i - 1, j - 1] + (2.0 * s if s >= min_sim else -1.0)
            up = S[i - 1, j] + (0.0 if j == m else gap)  # trailing lyric words: free
            left = S[i, j - 1] + gap * 0.5
            if diag >= up and diag >= left:
                S[i, j], B[i, j] = diag, 0
            elif up >= left:
                S[i, j], B[i, j] = up, 1
            else:
                S[i, j], B[i, j] = left, 2
    match = {}
    i, j = n, m
    while i > 0 and j > 0:
        if B[i, j] == 0:
            if sim(lyr[i - 1], heard[j - 1][2]) >= min_sim:
                match[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif B[i, j] == 1:
            i -= 1
        else:
            j -= 1
    return match


def song_lines(lines, heard, keep=0.6):
    """Timed lines for one excerpt: [{text, words: [[t, w], ...]}]."""
    words, owner = [], []
    for li, ln in enumerate(lines):
        for w in ln.split():
            words.append(norm(w))
            owner.append(li)
    if not words or not heard:
        return []
    match = local_align(words, heard)
    out = []
    for li, ln in enumerate(lines):
        ks = [k for k, o in enumerate(owner) if o == li]
        got = [k for k in ks if k in match]
        if len(got) < max(2, keep * len(ks)) and not (len(ks) <= 2 and len(got) == len(ks)):
            continue
        t = np.array([heard[match[k]][0] if k in match else np.nan for k in ks])
        known = np.where(~np.isnan(t))[0]
        if np.any(np.diff(t[known]) < 0):
            continue
        idx = np.arange(len(ks))
        t = np.interp(idx, known, t[known])
        t[: known[0]] = t[known[0]] - 0.28 * (known[0] - idx[: known[0]])
        t[known[-1] + 1:] = t[known[-1]] + 0.28 * (idx[known[-1] + 1:] - known[-1])
        out.append({"text": ln, "words": [[round(float(x), 3), w] for x, w in zip(t, ln.split())]})
    # a chorus repeated in the lyrics can match the same heard words twice: keep time order, no overlaps
    out.sort(key=lambda L: L["words"][0][0])
    clean = []
    for L in out:
        if clean and L["words"][0][0] < clean[-1]["words"][-1][0]:
            continue
        clean.append(L)
    return clean


# one look per song (Reuben, 14 Nov: fractals + lyrics only, no landscape):
# neighbours differ in palette turn, speed and kaleidoscope folds
LOOKS = [
    {"hue": 0.0, "speed": 0.7}, {"hue": 0.35, "speed": 1.0, "kaleido": 6},
    {"hue": 0.7, "speed": 0.8}, {"hue": 0.15, "speed": 1.2, "kaleido": 4},
    {"hue": 0.5, "speed": 0.6, "zoom": 0.85}, {"hue": 0.85, "speed": 1.1, "kaleido": 8},
    {"hue": 0.25, "speed": 0.9}, {"hue": 0.6, "speed": 1.3, "kaleido": 5},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song_dir")
    ap.add_argument("--tracks")
    ap.add_argument("--lyrics-md")
    ap.add_argument("--album", default="Natural Capital")
    a = ap.parse_args()
    M = os.path.join(a.song_dir, "media")
    cues = json.load(open(os.path.join(M, "cues.json")))
    cues = cues["cues"] if isinstance(cues, dict) else cues
    tracks = yaml.safe_load(open(a.tracks))["tracks"] if a.tracks else []
    num = {t["slug"]: t["n"] for t in tracks}
    master = sorted(glob.glob(os.path.join(M, "master.*")))[0]
    import soundfile
    try:
        duration = soundfile.info(master).duration
    except Exception:
        import librosa
        duration = librosa.get_duration(path=master)
    beats, downs, tempo = bars_from_beats(os.path.join(M, "beats.json"), duration)
    wp = os.path.join(M, "whisper-words.json")
    heard_all = load_whisper(wp) if os.path.exists(wp) else []

    sections, lines, report = [], [], []
    for c in cues:
        st = float(c.get("start_s", c.get("start")))
        en = float(c.get("end_s", c.get("end")))
        title = c["title"]
        slug = c.get("slug") or slugify(title)
        b0 = int(np.argmin(np.abs(downs - st)))
        if sections and b0 <= sections[-1]["start_bar"]:
            b0 = sections[-1]["start_bar"] + 2
        name = slug if all(s["name"] != slug for s in sections) else f"{slug}-{len(sections)}"
        sections.append({"name": name, "start_bar": int(b0), "title": title, "slug": slug, "start_s": st})
        heard = [h for h in heard_all if st - 0.3 <= h[0] < en + 0.3]
        ls = song_lines(find_lyrics(slug, title, tracks, a.lyrics_md), heard)
        lines += [dict(L, section=name) for L in ls]
        report.append(f"  {name:<24} {st:7.1f}-{en:7.1f}s  {len(heard):>3} words heard, {len(ls):>2} lines shown")
    sections[0]["start_bar"] = 0
    nbars = int(np.searchsorted(downs, duration))
    for k, s in enumerate(sections):
        s["bars"] = (sections[k + 1]["start_bar"] if k + 1 < len(sections) else nbars) - s["start_bar"]

    with open(os.path.join(a.song_dir, "timing.yaml"), "w") as fh:
        yaml.safe_dump({"note": "megamix: sections = song excerpts (tools/megamix_plan.py)", "tempo": round(float(tempo), 2),
                        "sections": [{k: s[k] for k in ("name", "start_bar", "bars")} for s in sections],
                        "lines": [{"section": L["section"], "text": L["text"], "words": L["words"]} for L in lines]},
                       fh, sort_keys=False, allow_unicode=True, width=200)
    json.dump({"tempo": float(tempo), "beats": [round(float(x), 4) for x in beats],
               "downbeats": [round(float(x), 4) for x in downs if x >= -0.01]},
              open(os.path.join(a.song_dir, "grid.json"), "w"))

    sb_path = os.path.join(a.song_dir, "storyboard.yaml")
    if not os.path.exists(sb_path):
        shots = []
        for k, s in enumerate(sections):
            n = num.get(s["slug"])
            kicker = f"{a.album} · {n:02d}" if n else a.album
            sh = {"id": s["name"], "at": s["name"], "plate": "fractal",
                  "text": {"mode": "lyric"}, "grade": {"exposure": 0.9}, "psych": dict(LOOKS[k % len(LOOKS)]),
                  "title": {"text": s["title"], "kicker": kicker}}
            if k:
                sh["transition"] = "dissolve"
                sh["start_s"] = s["start_s"]  # exact cue time, not the nearest bar
            shots.append(sh)
        sb = {"song": os.path.basename(os.path.normpath(a.song_dir)), "style": "glen-gothic", "preset": "lyric",
              "seed": 1707, "max_shot_bars": 200, "motifs": [], "shots": shots,
              "thumbnail": {"shot": sections[0]["name"], "at": 0.5, "title": a.album.upper(), "laird": False}}
        with open(sb_path, "w") as fh:
            fh.write("# Megamix lyric video: generated by tools/megamix_plan.py, edit freely.\n")
            yaml.safe_dump(sb, fh, sort_keys=False, allow_unicode=True, width=200)
    print(f"{len(sections)} songs, {len(lines)} lyric lines, {nbars} bars, {duration:.1f}s")
    print("\n".join(report))


if __name__ == "__main__":
    main()
