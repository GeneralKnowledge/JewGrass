# LAST CARRIAGE: OVERNIGHT NORTH (pilot)

An anonymous instrumental act seeded by a long instrumental livestream mix of a friend's experimental music, shared by Reuben (Nov 14). One overnight sleeper
train from London to the far north-west Highlands, told in music only: nine tracks that segue into each other,
so the album plays as one continuous mix like the original. It leaves in D minor, the key drifts by semitones
through the night, and it arrives back in D minor at first light. Act and album names are working titles.

The seed mix was measured with librosa (key path, tempo estimates, low-end balance, where it drops and rebuilds);
the findings are in `artist-dna.yaml` under `seed_analysis`, and each album track maps to a stretch of it.

| File | What it is |
|---|---|
| [`artist-dna.yaml`](artist-dna.yaml) | Concept, seed analysis, palette, motifs, the four families (heavy, glide, pulse, drift), caption core, avoid and banned lists, nine-track outline, round 1 plan |
| `design/04-beattock.yaml`, `05-over-the-border.yaml`, `08-rannoch.yaml` | The three pilot Song Designs |
| [`build.py`](build.py) | From `nightshift-fm/build.py`; also rejects circus instruments, "clean/polished" mix words and opening intros |
| [`tracks.yaml`](tracks.yaml) | Generated; same shape as `docs/album/tracks.yaml` |

```
python3 last-carriage/build.py      # rebuild after editing a design; exits 1 on any warning
```

## v2 (Nov 15): heavier and darker

Reuben heard round 1 and asked for heavier, darker instruments. The sound is now industrial metal weight
(down-tuned distorted guitars, palm-muted chugs, distorted synth bass, pounding drums) plus dark noir melody
(lonely minor piano, a mournful saxophone in the quiet moments). Touchstones are described in genre words and
never named; build.py rejects the names. Round 2 seeds: 2411/2412, 2511/2512, 2811/2812, to project files
last-carriage/round2/.

## v3 (Nov 15): heavier and grittier

Reuben liked v2 but found it too clean after the anti-hiss work. Grit now comes from distortion and saturation
words (raw, gritty, saturated, overdriven, fuzz, dirty, grimy, abrasive, wall of sound) in the caption core and
families. The hiss words stay banned. Round 3 seeds: 2421/2422, 2521/2522, 2821/2822, to last-carriage/round3/.

## Rules carried over from Nov 14

No crackle/hiss/static/vinyl/tape/radio words; no "train" or "railway" words either (they may pull in sound
effects), so the rails live in the eighth-note bass. No circus or oom-pah instruments. Warm, cohesive, blended
mix, never clean or polished. Few named solo instruments. Songs start straight in. No spoken parts at all.

## Round 1 (not launched; needs Reuben's go in the laptop thread)

Beattock (230 s), Over the Border (200 s), Rannoch (150 s), 2 takes each, Kaggle XL turbo.
Seeds 2401/2402, 2501/2502, 2801/2802 (2000 + track number x 100 + take).

```
python3 kaggle/laptop/kjob.py run JOB=render \
  TRACKS_YAML=https://raw.githubusercontent.com/GeneralKnowledge/JewGrass/claude/project-thread-ym6960/last-carriage/tracks.yaml \
  TRACKS='["lastcarriage-beattock","lastcarriage-over-the-border","lastcarriage-rannoch"]' \
  SEEDS='{"lastcarriage-beattock":[2401,2402],"lastcarriage-over-the-border":[2501,2502],"lastcarriage-rannoch":[2801,2802]}'
```

Sound reference (friend's permission confirmed Nov 14; the album was the friend's idea): the full mix and a
one-minute excerpt per pilot are on the never-merged `claude/reference-inputs` branch, as
`reference-inputs/overnight-north-ref-beattock.mp3`, `-over-the-border.mp3` and `-rannoch.mp3`.
