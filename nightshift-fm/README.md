# NIGHTSHIFT FM: PHANTOM FREQUENCY

An anonymous producer-led project: ten transmissions from a fictional late-night radio station. Memphis-style
phonk, drift phonk, experimental hip-hop and one Brazilian funk hybrid, almost entirely instrumental. The voice
is an instrument: short original phrases, chopped, pitched, stuttered and reversed. Genre and technique words
only; no real artist, record, sample or melody is named or imitated.

| File | What it is |
|---|---|
| [`artist-dna.yaml`](artist-dna.yaml) | Identity, palette, album motifs, voices, vocal rules, the five production families (each with its own drums), caption core, avoid lists, acceptance tests, round 1 plan |
| `design/01-...10-*.yaml` | One Song Design per track: render settings, palette, harmony, drums, bass, vocal motif and treatments, the moment, sections (tag + lyrics + bars + performer), ending, transmission layer, caption |
| [`build.py`](build.py) | Copied from `lowbridge/build.py`; compiles the DNA and designs into `tracks.yaml` and checks them |
| [`tracks.yaml`](tracks.yaml) | Generated. Same shape as `docs/album/tracks.yaml`, so the Kaggle notebook renders it unchanged |

```
python3 nightshift-fm/build.py      # rebuild after editing a design; exits 1 on any warning
```

## How it reaches ACE-Step

- Caption = DNA `caption_core` + the track's family caption (memphis, drift, experimental, ambient, baile) +
  the track's own caption. Key, tempo, meter and length go in as real conditioning.
- Lyrics hold only the words. A chop is written as the stutter it should become ("Can't, can't, can't get
  through"). Treatment goes in the section tag. Instrumental sections have a tag that says "instrumental" and
  no lyric lines (build.py checks both ways).
- Chord charts, drum patterns and bass movement are for the DAW stage; ACE-Step only hears them as words.
- No crackle, hiss or static anywhere, in renders or in post (Reuben's call). The radio concept is kept out of captions, because those
  words put hiss over whole songs before. Each design has a `transmission_layer` saying where those events go;
  they get added on the laptop after rendering, on the exact bar: sweeps, filtering, dropouts, pitch drift, tape stops. Section tags may name one event
  ("radio tuning sweep", "tape stop"), never a constant bed.
- Dead Channel was 96 bars in the sketch (320 s at 72 bpm); it is scaled to 68 bars to keep 240 s.
  Lost Transmission is 225 s (80 bars at 86 is 223 s). Static Dreams has no spare seconds: check its outro.

## Round 1

No Reception, Redline and Baile Fantasma (one per production family most likely to fail differently),
2 takes each on Kaggle, XL turbo. Seeds 4201/4202, 4301/4302, 4701/4702 (4000 + track number x 100 + take).
The other seven render only after Reuben has listened.

```
python3 kaggle/laptop/kjob.py run JOB=render \
  TRACKS_YAML=https://raw.githubusercontent.com/GeneralKnowledge/JewGrass/claude/project-thread-e17buf/nightshift-fm/tracks.yaml \
  TRACKS='["nightshift-no-reception","nightshift-redline","nightshift-baile-fantasma"]' \
  SEEDS='{"nightshift-no-reception":[4201,4202],"nightshift-redline":[4301,4302],"nightshift-baile-fantasma":[4701,4702]}'
```
