"""Turn docs/album/tracks.yaml plus the notebook's settings into a list of render jobs.

Pure Python (PyYAML only), so it runs anywhere and can be checked without a GPU:
  python kaggle/plan.py docs/album/tracks.yaml --tracks queue --takes 2 --first-seed 8001 --gpus 2

Each job is one take: slug, seed, caption, lyrics, seconds, bpm, key, timesig, gpu.
"""
import argparse
import json
import re
import sys

import yaml

# Caption phrases that make ACE-Step add hiss/static (heard on Brass Plate and Bought and Sold, Nov 14).
# Any comma-separated caption item, or item inside a [section - cue] tag, that matches is dropped.
STATIC_PRONE = re.compile(r"gramophone|crackl|typewriter clack|vinyl hiss|tape hiss|record hiss", re.I)


def scrub_caption(caption):
    """Drop static-prone comma items from a caption. Returns (clean caption, list of dropped items)."""
    items = [i.strip() for i in caption.replace("\n", " ").split(",")]
    kept = [i for i in items if i and not STATIC_PRONE.search(i)]
    dropped = [i for i in items if i and STATIC_PRONE.search(i)]
    return ", ".join(kept), dropped


def _scrub_tag(match, dropped):
    inner = match.group(1)
    if " - " not in inner:
        return match.group(0)
    head, cue = inner.split(" - ", 1)
    items = [i.strip() for i in cue.split(",")]
    kept = [i for i in items if not STATIC_PRONE.search(i)]
    dropped += [i for i in items if STATIC_PRONE.search(i)]
    return f"[{head} - {', '.join(kept)}]" if kept else f"[{head}]"


def clean_lyrics(lyrics):
    """Strip stage directions (lines fully in parentheses) and static-prone cues in section tags."""
    dropped = []
    out = []
    for line in lyrics.splitlines():
        s = line.strip()
        if s.startswith("(") and s.endswith(")"):
            continue
        out.append(re.sub(r"\[([^\]]*)\]", lambda m: _scrub_tag(m, dropped), line))
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
    return text, dropped


def pick_tracks(doc, tracks):
    """tracks: 'queue' (every status=generate track, queue order), 'all' (album order), or a list of slugs."""
    by_slug = {t["slug"]: t for t in doc["tracks"]}
    if tracks == "queue":
        slugs = [s for s in doc.get("queue", []) if by_slug.get(s, {}).get("status") == "generate"]
        slugs += [t["slug"] for t in doc["tracks"] if t.get("status") == "generate" and t["slug"] not in slugs]
    elif tracks == "all":
        slugs = [t["slug"] for t in doc["tracks"] if t.get("status") == "generate"]
    else:
        slugs = list(tracks)
        missing = [s for s in slugs if s not in by_slug]
        if missing:
            raise SystemExit(f"not in tracks.yaml: {missing}. Known: {sorted(by_slug)}")
        nolyrics = [s for s in slugs if not by_slug[s].get("lyrics")]
        if nolyrics:
            raise SystemExit(f"no caption/lyrics in tracks.yaml (existing/voice/pending tracks): {nolyrics}")
    return [by_slug[s] for s in slugs]


def plan(doc, tracks="queue", takes=2, first_seed=8001, seeds=None, gpus=1, max_seconds=300):
    """seeds: optional {slug: [seed, ...]} that overrides takes/first_seed for that track."""
    seeds = seeds or {}
    jobs, notes = [], []
    for t in pick_tracks(doc, tracks):
        caption, cap_drop = scrub_caption(t["caption"])
        lyrics, lyr_drop = clean_lyrics(t["lyrics"])
        if cap_drop or lyr_drop:
            notes.append(f"{t['slug']}: dropped static-prone phrases {cap_drop + lyr_drop}")
        secs = min(float(t.get("seconds") or 180), max_seconds)
        for seed in seeds.get(t["slug"]) or [first_seed + i for i in range(takes)]:
            jobs.append({
                "slug": t["slug"], "title": t.get("title", t["slug"]), "seed": int(seed),
                "caption": caption, "lyrics": lyrics, "seconds": secs,
                "bpm": t.get("bpm"), "key": t.get("key") or "", "timesig": str(t.get("timesig") or ""),
            })
    # Longest jobs first, then deal them out to the GPU with the least work, so both cards finish together.
    load = [0.0] * gpus
    for j in sorted(jobs, key=lambda j: -j["seconds"]):
        g = load.index(min(load))
        j["gpu"] = g
        load[g] += j["seconds"]
    return jobs, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks_yaml")
    ap.add_argument("--tracks", default="queue", help="queue, all, or comma-separated slugs")
    ap.add_argument("--takes", type=int, default=2)
    ap.add_argument("--first-seed", type=int, default=8001)
    ap.add_argument("--seeds", default="{}", help='JSON, e.g. {"brass-plate": [7102, 7103]}')
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--out", help="write jobs as JSON here")
    a = ap.parse_args()
    doc = yaml.safe_load(open(a.tracks_yaml))
    tracks = a.tracks if a.tracks in ("queue", "all") else [s.strip() for s in a.tracks.split(",") if s.strip()]
    jobs, notes = plan(doc, tracks, a.takes, a.first_seed, json.loads(a.seeds), a.gpus)
    for n in notes:
        print("NOTE", n, file=sys.stderr)
    for j in jobs:
        print(f"gpu{j['gpu']}  {j['slug']:<24} seed {j['seed']}  {j['seconds']:.0f}s  bpm {j['bpm']}  "
              f"key {j['key'] or '-'}  timesig {j['timesig'] or '-'}", file=sys.stderr)
    if a.out:
        json.dump(jobs, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
