# JewGrass Video Engine

Turns a finished song into an art-directed music video. Design and reasoning:
[`docs/video/video-engine.md`](../docs/video/video-engine.md).

Status: **skeleton**. The analysis, storyboard validation and deterministic
compositor work end to end on a synthetic test song with procedural
placeholder art. Not built yet: lyric alignment from the real vocal,
generated (SD 1.5) plates, characters, the deed/logo motifs, reading the
presets from `styles/presets.yaml` (the budgets are currently hard-coded in
`engine/compositor.py`), thumbnail and vertical cut.

## Layout

```
engine/
  analysis.py     stage 1: beats, bars, per-stem envelopes, macro curves -> analysis.json
  storyboard.py   load + validate storyboard.yaml against timing.yaml
  plates.py       painted backgrounds as depth layers (procedural now; generated later)
  motifs.py       crow, red boundary line (drawn in code, identical every video)
  typography.py   lyric sprites and text modes
  fx.py           fog, particles, paper, grain, misregistration, vignette
  compositor.py   stage 4: frame(i), a pure function of the inputs
  render.py       parallel chunks -> ffv1 segments -> H.264 MP4 + contact sheet + frames.md5
styles/presets.yaml        per-song visual signatures
motifs/natural-capital.yaml recurring motifs for the album
assets/fonts/              OFL fonts (Oswald, Libre Baskerville, Caveat)
songs/<slug>/              storyboard.yaml, timing.yaml (+ analysis.json, master, out/ are not committed)
tools/make_test_song.py    synthetic 30 s test song
```

## Try it

CPU only, Python 3.10+, ffmpeg. In its own venv (never ACE-Step's):

```
pip install numpy scipy librosa soundfile opencv-python-headless pillow pyyaml
cd video
python tools/make_test_song.py songs/test-glen
python -m engine.analysis songs/test-glen/master.wav songs/test-glen/analysis.json --beats songs/test-glen/beats.json
python -m engine.render songs/test-glen --stills      # contact sheet only, seconds
python -m engine.render songs/test-glen --jobs 4      # full video
```

On the laptop, wrap the render in `~/band/heavy.sh` and only run it when no
music job is using the machine.

## Determinism

`out/frames.md5` holds a checksum per frame of the lossless render. Two
renders of the same inputs must produce identical files:
`diff run1/frames.md5 run2/frames.md5`.
