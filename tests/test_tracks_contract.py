"""The tracks.yaml contract that kaggle/plan.py and the laptop loop rely on, plus the caption rules that apply
to every act (Nov 14 lessons). Checks committed files only; no build, no audio."""
import glob
import os
import re

import pytest
import yaml

from conftest import ROOT

FILES = sorted(
    os.path.relpath(p, ROOT)
    for p in glob.glob(os.path.join(ROOT, "*", "tracks.yaml")) + [os.path.join(ROOT, "docs", "album", "tracks.yaml")]
)
NEEDED = ("slug", "status", "seconds", "caption", "lyrics")  # plan.py reads bpm/key/timesig with .get()

# Words that put static/hiss or a circus feel into a render. Everything else stays an act-level choice.
BANNED = re.compile(r"\b(crackle|crackling|hiss|gramophone|typewriter|circus|oom-?pah|clean polished)\b", re.I)


def load(rel):
    return yaml.safe_load(open(os.path.join(ROOT, rel)))


@pytest.mark.parametrize("rel", FILES)
def test_shape(rel):
    doc = load(rel)
    slugs = [t["slug"] for t in doc["tracks"]]
    assert len(slugs) == len(set(slugs)), "duplicate slugs"
    assert set(doc.get("queue", [])) <= set(slugs), "queue names a slug with no track"
    for t in doc["tracks"]:
        if t.get("status") == "generate":
            missing = [k for k in NEEDED if not t.get(k)]
            assert not missing, f"{t['slug']} is missing {missing}"
            assert 10 <= float(t["seconds"]) <= 300, f"{t['slug']}: {t['seconds']} s"
            assert str(t.get("timesig") or "4") in ("2", "3", "4", "6"), f"{t['slug']}: timesig {t.get('timesig')}"


@pytest.mark.parametrize("rel", [f for f in FILES if f != "docs/album/tracks.yaml"])
def test_no_static_or_circus_words(rel):
    """docs/album/tracks.yaml is the pre-design baseline Reuben likes, so it is kept as it is."""
    bad = [(t["slug"], m.group(0)) for t in load(rel)["tracks"] if t.get("status") == "generate"
           for m in BANNED.finditer(t["caption"] + " " + " ".join(re.findall(r"\[([^\]]*)\]", t["lyrics"])))]
    assert not bad, bad
