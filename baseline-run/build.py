"""Build baseline-run/tracks.yaml: every act's songs (lyrics, length, bpm, key from the act's snapshot in sources/)
with the overnight-style captions from captions.yaml. One file, so one Kaggle run renders everything.

    python3 baseline-run/build.py          # writes baseline-run/tracks.yaml and prints a summary
    python3 baseline-run/build.py --check  # fails if tracks.yaml is out of date

Old captions are never touched: sources/ holds the acts' tracks.yaml as they were on Nov 15.
"""
import argparse
import os
import re
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
# Words that add static (Nov 14) or a circus feel (Nov 14), plus the boilerplate this run drops on purpose.
BANNED = re.compile(r"crackl|hiss|static|vinyl|gramophone|typewriter|\btape\b|cassette|tuba|pizzicato|woodblock|"
                    r"marching snare|oom-pah|circus|comic|theatrical|cohesive|blended mix|not overproduced|"
                    r"not over-produced|polished", re.I)


def drop_outro(lyrics):
    """Remove a trailing [outro] section (the spoken sign-offs); the song then ends on the section before."""
    parts = re.split(r"(?m)^(?=\[)", lyrics.strip())
    if parts and parts[-1].lower().startswith("[outro"):
        parts = parts[:-1]
    return "".join(parts).strip() + "\n"


def build():
    cfg = yaml.safe_load(open(os.path.join(HERE, "captions.yaml")))
    tracks, queue, problems = [], [], []
    for act, a in cfg["acts"].items():
        src = yaml.safe_load(open(os.path.join(HERE, a["source"])))
        by_slug = {t["slug"]: t for t in src["tracks"]}
        for e in a.get("extra", []):
            t = {k: v for k, v in e.items() if k != "lyrics_file"}
            t["status"] = "generate"
            t["lyrics"] = open(os.path.join(HERE, e["lyrics_file"])).read()
            by_slug[t["slug"]] = t
        wanted = [s for s in a["captions"]]
        missing = [s for s in wanted if s not in by_slug]
        unrendered = [t["slug"] for t in src["tracks"] if t.get("status") == "generate"
                      and t["slug"] not in a["captions"] and t["slug"] not in a.get("skip", [])]
        if missing:
            problems.append(f"{act}: captions for unknown songs {missing}")
        if unrendered:
            problems.append(f"{act}: songs with no new caption {unrendered}")
        for slug in wanted:
            s = by_slug.get(slug)
            if not s:
                continue
            cap = " ".join(a["captions"][slug].split())
            if act != "natural-capital" and BANNED.search(cap):
                problems.append(f"{slug}: banned word '{BANNED.search(cap).group(0)}'")
            elif act == "natural-capital" and re.search(r"crackl|hiss|static|vinyl|gramophone|typewriter|\btape\b", cap, re.I):
                problems.append(f"{slug}: static-prone word in the overnight caption")
            lyrics = s["lyrics"]
            if a.get("drop_outro"):
                lyrics = drop_outro(lyrics)
            t = {"act": act, "slug": a.get("prefix", "") + slug, "title": s.get("title", slug), "status": "generate",
                 "seconds": s["seconds"], "bpm": s.get("bpm")}
            if not a.get("no_key"):
                for k in ("key", "timesig"):
                    if s.get(k):
                        t[k] = s[k]
            t["caption"] = cap + "\n"
            t["previous_caption"] = " ".join((s.get("caption") or "(none)").split()) + "\n"
            t["lyrics"] = lyrics
            tracks.append(t)
            queue.append(t["slug"])
    doc = {"album": "Baseline run (Nov 15): every act in the overnight caption style",
           "note": "Built by baseline-run/build.py from captions.yaml + sources/. Edit those, not this file.",
           "queue": queue, "tracks": tracks}
    return doc, problems


class _Lit(str):
    pass


def _dump(doc):
    yaml.add_representer(_Lit, lambda d, s: d.represent_scalar("tag:yaml.org,2002:str", s, style="|"))
    for t in doc["tracks"]:
        for k in ("caption", "previous_caption", "lyrics"):
            t[k] = _Lit(t[k])
    return yaml.dump(doc, sort_keys=False, allow_unicode=True, width=10000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    doc, problems = build()
    for p in problems:
        print("PROBLEM", p, file=sys.stderr)
    if problems:
        sys.exit(1)
    text = _dump(doc)
    out = os.path.join(HERE, "tracks.yaml")
    if a.check:
        sys.exit(0 if os.path.exists(out) and open(out).read() == text else "tracks.yaml is out of date: run build.py")
    open(out, "w").write(text)
    acts = {}
    for t in doc["tracks"]:
        n, s = acts.get(t["act"], (0, 0))
        acts[t["act"]] = (n + 1, s + float(t["seconds"]))
    total = sum(s for _, s in acts.values())
    for act, (n, s) in acts.items():
        print(f"{act:<16} {n:>2} songs  {s / 60:5.1f} min")
    print(f"{'total':<16} {len(doc['tracks']):>2} songs  {total / 60:5.1f} min of music per take")


if __name__ == "__main__":
    main()
