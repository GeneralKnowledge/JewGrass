"""tools/design/compile_design.py: every album Song Design compiles, and the outputs the laptop reads exist."""
import glob
import os
import subprocess
import sys

import pytest

from conftest import ROOT

DESIGNS = sorted(glob.glob(os.path.join(ROOT, "docs", "album", "design", "*.yaml")))


@pytest.mark.parametrize("path", DESIGNS, ids=os.path.basename)
def test_design_compiles(path, tmp_path):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "design", "compile_design.py"), path,
                        str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    outs = {os.path.basename(p) for p in glob.glob(str(tmp_path / "*" / "*" / "*"))}
    assert {"caption.txt", "lyrics.txt", "meta.env", "run.sh"} <= outs, outs


def test_gavs_song_compiles_as_6_8(tmp_path):
    """Gav's Song is the only 6/8 design; the laptop must get TIMESIG=6 (ACE-Step's 6/8 code)."""
    path = os.path.join(ROOT, "docs", "album", "design", "gavs-song.yaml")
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "design", "compile_design.py"), path, str(tmp_path)],
                   check=True, capture_output=True)
    (meta,) = glob.glob(str(tmp_path / "*" / "*" / "meta.env"))
    assert "TIMESIG=6\n" in open(meta).read()
