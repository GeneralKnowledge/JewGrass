# MR. SIDEQUEST

Reuben's spec (Nov 14): a genuinely talented British rapper who finds the ridiculous in everyday life and treats
ordinary life as a series of side quests. Funny because of the writing and delivery, technically good, and able
to turn sincere. Everything is original; genre references only.

| File | What it is |
|---|---|
| [`artist-dna.yaml`](artist-dna.yaml) | Persona, voice, delivery, musical modes, production and writing rules, caption core, acceptance |
| [`design/main-character-syndrome.yaml`](design/main-character-syndrome.yaml) | 01 Main Character Syndrome. 104 bpm, G minor, 220 s. Full lyrics. Renders first |
| [`build.py`](build.py) | Compiles DNA + designs into `tracks.yaml`; refuses intro sections and hiss-prone words |
| [`tracks.yaml`](tracks.yaml) | Generated, same shape as `docs/album/tracks.yaml` |

*I've Got a System* (138 bpm, C minor) and *Laughing at Myself* (92 bpm, D major) get full lyrics if the first
song works. Changes from the spec for Reuben's standing rules: no spoken intros, bridges rapped on the beat,
endings are rapped or hook lines with a hard stop.

Round 1 on Kaggle (spare card during the reuben2 voice retrain), XL turbo, seeds 6401 and 6402:

```
TRACKS_YAML=https://raw.githubusercontent.com/GeneralKnowledge/JewGrass/claude/project-thread-o6z35j/mr-sidequest/tracks.yaml
SEEDS='{"mr-sidequest-main-character-syndrome":[6401,6402]}'
```
