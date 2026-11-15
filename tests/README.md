# Checks

    pip install pyyaml pytest ruff
    pytest -q          # about 2 s, CPU only
    ruff check .       # syntax errors and undefined names only

CI runs the same on every pull request (`.github/workflows/ci.yml`). No audio is generated.

What is covered:

- **Every `<act>/build.py`** (found automatically, so new acts are covered once merged): it passes its own
  checks and reproduces the committed `tracks.yaml` exactly. If that fails after you edit a design, rerun
  `python3 <act>/build.py` and commit the new `tracks.yaml`.
- **6/8 timing**: `timesig: "6"` is ACE-Step's 6/8 code. A 6/8 bar is 3 quarter-note beats, so the
  "render shorter than planned" check counts 3 beats a bar, not 6.
- **The tracks.yaml contract** that `kaggle/plan.py` and the laptop loop read: `queue` (slugs) and
  `tracks[]` with `slug`, `status`, and for `status: generate` also `seconds`, `caption`, `lyrics`
  (`bpm`, `key`, `timesig` optional). `status: voice` tracks are skipped by renders.
- **Caption rules for every act**: no crackle/hiss/gramophone/typewriter/circus/oom-pah/"clean polished"
  (`docs/album/tracks.yaml`, the pre-design baseline, is left as it is).
- **Song Designs** in `docs/album/design/` compile with `tools/design/compile_design.py`.
