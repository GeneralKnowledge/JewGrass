#!/usr/bin/env python3
"""Compile a Song Design (docs/album/design/<slug>.yaml) into what ACE-Step reads.

    python3 tools/design/compile_design.py docs/album/design/gavs-song.yaml [OUT_ROOT]

Writes OUT_ROOT/<slug>/design-v<N>/ (default OUT_ROOT: build/design):
  caption.txt   the caption
  lyrics.txt    lyrics with each section tag carrying its arrangement cue, e.g. [verse - guitar and bass only]
  meta.env      SLUG, BPM, KEY, TIMESIG, SECONDS (gen.py reads BAND_KEY / BAND_TIMESIG)
  run.sh        the exact laptop command for one seed

Only needs PyYAML. Checks the things that have bitten us before.
"""
import os
import re
import sys

import yaml


def compile_design(path, out_root="build/design"):
    d = yaml.safe_load(open(path))
    r = d["render"]
    problems = []

    caption = " ".join(d["caption"].split())
    if len(caption) > 900:
        problems.append(f"caption is {len(caption)} chars; keep it under ~900")
    if not 30 <= r["seconds"] <= 300:
        problems.append("seconds should be 30-300 (one pass on the laptop)")
    if str(r["timesig"]) not in ("2", "3", "4", "6"):
        problems.append("timesig must be 2, 3, 4 or 6 (ACE-Step codes)")

    blocks = []
    for s in d["sections"]:
        tag = s["tag"].strip()
        if len(tag.split()) > 16:
            problems.append(f"tag for {s['name']} is long ({len(tag.split())} words)")
        text = (s.get("lyrics") or "").strip("\n")
        for line in text.splitlines():
            if re.fullmatch(r"\s*\(.*\)\s*", line):
                problems.append(f"{s['name']}: line '{line.strip()}' is all in brackets and the laptop loop strips it")
        blocks.append(f"[{tag}]" + ("\n" + text if text else ""))
    lyrics = "\n\n".join(blocks) + "\n"

    out = os.path.join(out_root, d["slug"], f"design-v{d.get('version', 1)}")
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "caption.txt"), "w").write(caption + "\n")
    open(os.path.join(out, "lyrics.txt"), "w").write(lyrics)
    open(os.path.join(out, "meta.env"), "w").write(
        f"SLUG={d['slug']}\nBPM={r['bpm']}\nKEY='{r['key']}'\nTIMESIG={r['timesig']}\nSECONDS={r['seconds']}\n"
    )
    name = f"{d['slug']}-design-v{d.get('version', 1)}"
    open(os.path.join(out, "run.sh"), "w").write(
        "#!/usr/bin/env bash\n"
        "# usage: run.sh SEED   (run from this folder on the laptop)\n"
        "set -euo pipefail\n"
        'cd "$(dirname "$0")"\n'
        "seed=${1:?seed}\n"
        f"BAND_KEY='{r['key']}' BAND_TIMESIG={r['timesig']} \\\n"
        f"  flock ~/band/.gpu.lock ~/band/heavy.sh ~/band/gen.sh caption.txt lyrics.txt {r['seconds']} \"$seed\" "
        f"{name}-s$seed {r['bpm']}\n"
    )
    os.chmod(os.path.join(out, "run.sh"), 0o755)
    return out, problems, caption, lyrics


if __name__ == "__main__":
    out, problems, caption, lyrics = compile_design(*sys.argv[1:3])
    print(f"wrote {out}  (caption {len(caption)} chars, {lyrics.count(chr(10))} lyric lines)")
    for p in problems:
        print("WARNING:", p)
    sys.exit(1 if problems else 0)
