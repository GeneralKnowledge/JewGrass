#!/usr/bin/env python3
"""Build tools/player/catalog.json: which act each song belongs to, its title, and the other slugs its takes
were rendered under, so the take server shows every take of a song together.

    python3 tools/player/build_catalog.py baseline-run/tracks.yaml baseline-run/sources/*.yaml

Reads the baseline run's tracks.yaml first (its slugs are the canonical ones), then each act's own tracks.yaml
(earlier rounds). A song from an earlier round whose slug differs only by the act prefix (Natural Capital's
album renders were "swim-at-your-own-risk", the baseline is "nc-swim-at-your-own-risk") becomes an alias.
Needs PyYAML; the server itself only reads the JSON this writes.
"""

import json
import os
import sys

import yaml

ACTS = {
    "natural-capital": "Natural Capital",
    "book-of-names": "Book of Names",
    "after-hours": "AFTER HOURS",
    "mr-sidequest": "MR. SIDEQUEST",
    "nightshift-fm": "NIGHTSHIFT FM",
    "iron-canticle": "IRON CANTICLE",
    "last-carriage": "LAST CARRIAGE",
    "lowbridge": "LOWBRIDGE",
}
# act prefixes the baseline run added to slugs that had none before
PREFIXES = {"natural-capital": "nc-"}
# slugs the take server makes up for takes without a sidecar, or one-off checks, that belong to a known song
EXTRA_ALIASES = {
    "nc-swim-at-your-own-risk": ["swim", "swim-check"],  # laptop outputs/swim-len240-1111.wav, Kaggle swim check
}


def tracks(path):
    doc = yaml.safe_load(open(path))
    return doc.get("tracks", []) if isinstance(doc, dict) else doc


def main(paths):
    songs, alias_of = {}, {}
    baseline, sources = paths[0], paths[1:]
    for t in tracks(baseline):
        songs[t["slug"]] = {"title": t.get("title") or t["slug"], "act": t["act"], "aliases": []}
    for path in sources:
        act = os.path.splitext(os.path.basename(path))[0]
        prefix = PREFIXES.get(act, "")
        for t in tracks(path):
            slug = t["slug"]
            if slug in songs:
                continue
            canonical = prefix + slug if prefix + slug in songs else None
            if canonical:
                songs[canonical]["aliases"].append(slug)
            else:  # a song the baseline left out (Brass Plate, Deed to the Glen, voice notes): list it under its act
                songs[slug] = {"title": t.get("title") or slug, "act": act, "aliases": []}
    for canonical, extra in EXTRA_ALIASES.items():
        if canonical in songs:
            songs[canonical]["aliases"] += [a for a in extra if a not in songs[canonical]["aliases"]]
    for slug, s in songs.items():
        for a in s["aliases"]:
            if a in alias_of or a in songs:
                sys.exit(f"alias {a} is used twice ({alias_of.get(a, a)} and {slug})")
            alias_of[a] = slug
    out = {
        "note": "Built by tools/player/build_catalog.py. The take server groups every take whose slug is a song's "
                "slug or one of its aliases under that song, and shows the act.",
        "acts": ACTS,
        "songs": dict(sorted(songs.items())),
    }
    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "catalog.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"{len(songs)} songs, {len(alias_of)} aliases -> {dest}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
