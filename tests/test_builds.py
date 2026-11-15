"""Regression tests for the per-act compilers (<act>/build.py).

Every act folder with a build.py is found automatically, so acts merged later (AFTER HOURS, NIGHTSHIFT FM,
IRON CANTICLE...) are covered without touching this file. Each build runs on a temporary copy, so the
committed tracks.yaml is never rewritten by a test.
"""
import glob
import os
import shutil
import subprocess
import sys

import pytest
import yaml

from conftest import ROOT

ACTS = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(ROOT, "*", "build.py")))
ACTS_WITH_BARS = [a for a in ACTS if "planned = bars" in open(os.path.join(ROOT, a, "build.py")).read()]


def run_build(folder):
    return subprocess.run([sys.executable, os.path.join(folder, "build.py")], capture_output=True, text=True)


@pytest.fixture
def act_copy(tmp_path):
    def copy(act):
        dst = tmp_path / act
        shutil.copytree(os.path.join(ROOT, act), dst)
        return dst
    return copy


@pytest.mark.parametrize("act", ACTS)
def test_build_is_clean_and_matches_committed_tracks(act, act_copy):
    """The build passes its own checks and reproduces the committed tracks.yaml byte for byte.

    If this fails after an edit to a design, rerun `python3 <act>/build.py` and commit the new tracks.yaml."""
    folder = act_copy(act)
    r = run_build(folder)
    assert r.returncode == 0, r.stdout + r.stderr
    committed = open(os.path.join(ROOT, act, "tracks.yaml")).read()
    assert (folder / "tracks.yaml").read_text() == committed, f"{act}/tracks.yaml is stale; rerun {act}/build.py"


def write_6_8_design(folder, bpm=76, bars=104, seconds=250):
    """Swap the act's designs for one 6/8 song: 104 bars at 76 bpm is 246 s (3 quarter notes a bar)."""
    designs = sorted((folder / "design").glob("*.yaml"))
    d = yaml.safe_load(designs[0].read_text())
    for p in designs:
        p.unlink()
    per = bars // len(d["sections"])
    for s in d["sections"]:
        s["bars"] = per
    d["sections"][-1]["bars"] += bars - per * len(d["sections"])
    d["render"].update({"bpm": bpm, "timesig": "6", "seconds": seconds})
    (folder / "design" / designs[0].name).write_text(yaml.safe_dump(d, sort_keys=False))


@pytest.mark.parametrize("act", ACTS_WITH_BARS)
def test_6_8_duration_counts_three_beats_a_bar(act, act_copy):
    """timesig "6" is ACE-Step's code for 6/8. A 6/8 bar is 3 quarter-note beats, not 6, so a 250 s render of
    104 bars at 76 bpm is long enough (old formula said 493 s and warned)."""
    folder = act_copy(act)
    write_6_8_design(folder)
    r = run_build(folder)
    assert "shorter than the planned" not in r.stdout, r.stdout


@pytest.mark.parametrize("act", ACTS_WITH_BARS)
def test_too_short_6_8_render_still_warns(act, act_copy):
    folder = act_copy(act)
    write_6_8_design(folder, seconds=200)
    r = run_build(folder)
    assert "shorter than the planned 246 s" in r.stdout, r.stdout
