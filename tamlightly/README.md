# TAM LIGHTLY

A fictional, lighthearted rap act: **Tam Lightly**, twenty-three, from Auchterbeg (a made-up small town in
central Scotland). He raps about tiny everyday wins as if they were cup finals: total confidence, very low
stakes, and the joke is always on him. Bright, bouncy, catchy and daft, with a bit of heart.

Everything here is original. The vibe (playful, fun, internet-catchy rap) is the only reference; no real
artist, record, person, brand or place is named or imitated in any file that goes into generation.

The pilot uses the LOWBRIDGE layout (artist DNA + Song Designs, compiled into a `tracks.yaml` the Kaggle
notebook renders as is).

| File | What it is |
|---|---|
| [`artist-dna.yaml`](artist-dna.yaml) | Who Tam is, his world, sound, rules, the shared caption core, acceptance criteria |
| [`design/meal-deal-champion.yaml`](design/meal-deal-champion.yaml) | 01 Meal Deal Champion. 146 bpm, F major, 105 s. The single: the last sandwich, told like a cup final |
| [`design/ten-thousand-steps.yaml`](design/ten-thousand-steps.yaml) | 02 Ten Thousand Steps. 160 bpm, E minor, 90 s. The fast one: a smartwatch step goal at midnight |
| [`design/nan-says-im-gorgeous.yaml`](design/nan-says-im-gorgeous.yaml) | 03 Nan Says I'm Gorgeous. 92 bpm, C major, 175 s. The warm singalong: his only fan is his nan |
| [`build.py`](build.py) | Compiles the DNA and designs into `tracks.yaml` and checks them |
| [`tracks.yaml`](tracks.yaml) | Generated. Same shape as `docs/album/tracks.yaml` |

Built in from what we've learned: every song starts straight in on the hook (`build.py` refuses an intro
section), no spoken outros, three different lengths and tempos, and hiss-prone words are refused in captions
and tags.

```
python3 tamlightly/build.py      # rebuild tracks.yaml after editing a design; exits 1 on any warning
```

## Rendering on Kaggle (round 1)

Two takes per song in XL turbo, seeds 6101/6102, 6201/6202, 6301/6302:

```
python3 kaggle/laptop/kjob.py run JOB=render \
  TRACKS_YAML=https://raw.githubusercontent.com/GeneralKnowledge/JewGrass/claude/project-thread-o6z35j/tamlightly/tracks.yaml \
  SEEDS='{"tamlightly-meal-deal-champion":[6101,6102],"tamlightly-ten-thousand-steps":[6201,6202],"tamlightly-nan-says-im-gorgeous":[6301,6302]}'
```

About 6 minutes of music, so roughly 20 to 30 minutes of session time including the install.
