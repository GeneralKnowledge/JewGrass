"""megamix_config.py OUT_JSON: build megamix.py's config from ~/band/final, tracks.yaml and the analyses."""
import json
import os
import sys

import yaml

B = os.path.expanduser("~/band")
y = yaml.safe_load(open(f"{B}/album-tracks.yaml"))
info = {t["slug"]: t for t in y["tracks"]}
seeds = {}
for slug in info:
    p = f"{B}/songs/{slug}/DONE.json"
    if os.path.exists(p):
        d = json.load(open(p))
        if d.get("chosen_seed"):
            seeds[slug] = d["chosen_seed"]
special = {"deed-to-the-glen": ("10-analysis/analysis-v2-8003.json", 140),
           "swim-at-your-own-risk": ("10-analysis/analysis-5555.json", 140),
           "cold-hame": ("10-analysis/analysis-master.json", 140),
           "we-bought-the-hill": ("10-analysis/analysis-master.json", 170)}
tracks = []
for f in sorted(os.listdir(f"{B}/final")):
    if not f.endswith(".wav") or not f[:2].isdigit():
        continue
    slug = f[3:-4]
    if slug in special:
        an, bpm = special[slug]
    else:
        an, bpm = f"10-analysis/analysis-{seeds[slug]}.json", int(info[slug]["bpm"])
    an = f"{B}/songs/{slug}/{an}"
    a = json.load(open(an))
    tracks.append({"slug": slug, "title": info.get(slug, {}).get("title", slug.replace("-", " ").title()),
                   "file": f"{B}/final/{f}", "analysis": an, "bpm": bpm,
                   "skit": slug in ("parcel-of-rogues", "viewing"), "avoid": a.get("dropouts_s", [])})
cfg = {"tracks": tracks, "first": "parcel-of-rogues", "last": ["gavs-song", "we-bought-the-hill"],
       "out_dir": f"{B}/final/megamix", "target_min_s": 540, "target_max_s": 720}
json.dump(cfg, open(sys.argv[1], "w"), indent=1)
print(len(tracks), [(t["slug"], t["bpm"], os.path.basename(t["analysis"])) for t in tracks])
