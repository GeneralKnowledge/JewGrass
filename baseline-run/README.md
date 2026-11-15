# Baseline run (Nov 15): every act in the overnight caption style

Reuben, Nov 15: "Seems like we have went too clean across the board. I was just trying to get rid of static.
We should rethink all prompts based on our initial overnight run."

The overnight Natural Capital run (Nov 14 night, takes in project files `album-run1/` and `album-xl/`) and
Swim at Your Own Risk are the sound he likes. This folder rewrites every act's captions in that style and puts
all of them in one `tracks.yaml`, so one Kaggle run renders everything.

## What the overnight style is

One line: genre fusion first, then the voice (accent and attitude), then a handful of concrete instruments with
character (distorted 808, hard trap drums, heavy guitars on the chorus, a gang chant), one or two colours that
belong to that song, `raw mix`, tempo. About 30 to 55 words.

What went in after the overnight run and is now dropped: the shared boilerplate at the front of every caption
("warm cohesive blended mix", "raw saturated live-sounding instruments, not overproduced, gritty warm blended
mix", long instrument lists repeated on every song), which made everything sound alike and too clean.

Rules kept, because they fix real problems:

- no crackle, hiss, static, vinyl, gramophone, typewriter or tape words (they put static in the render)
- no circus or oom-pah instruments (tuba, pizzicato, woodblock, marching snare, comic, theatrical)
- songs start straight in and end sung (Natural Capital's spoken "due Friday" outros are dropped;
  its spoken intros stay, as on the overnight run)
- key and time signature still reach the model through `key`/`timesig`, not the caption

## The acts

| act | songs | source (snapshot in `sources/`) |
|---|---|---|
| Natural Capital (recreated) | 9 overnight songs, Brass Plate left out, plus Swim at Your Own Risk as the control | `claude/project-thread-59abul` docs/album/tracks.yaml |
| Book of Names | 12 | `rla07g` natural-capital-2/tracks.yaml |
| AFTER HOURS | 3 | `hhd444` after-hours/tracks.yaml |
| MR. SIDEQUEST | 3 | `o6z35j` mr-sidequest/tracks.yaml |
| NIGHTSHIFT FM | 10 | `e17buf` nightshift-fm/tracks.yaml |
| IRON CANTICLE | 10 | `pku6cs` iron-canticle/tracks.yaml |
| LAST CARRIAGE | 3 | `ym6960` last-carriage/tracks.yaml |
| LOWBRIDGE | 3 | `in6kwa` lowbridge/tracks.yaml |

Natural Capital's captions are the overnight text word for word, except four static-prone phrases:
Ermine and Ink "typewriter and printing press as drums" became "heavy stamp thuds as percussion"; Gav's Song
lost "subtle tape hiss, gentle glitch fade at the end" (now "ends on the sung chorus"); Parcel of Rogues
"digital noise" became "stutter edits"; Viewing lost "tape warble". Swim keeps its own laptop caption.

Lyrics, lengths, bpm and keys are the acts' own, unchanged. Each song in `tracks.yaml` also carries its
`previous_caption` so the two can be compared. Nothing in the acts' own folders was changed.

## Build and run

```
python3 baseline-run/build.py          # captions.yaml + sources/ -> tracks.yaml (checks banned words)
python3 kaggle/plan.py baseline-run/tracks.yaml --takes 2 --first-seed 1501 --gpus 2 --models xl-turbo
```

54 songs, 176 min of music per take. Two XL turbo takes each (seeds 1501 and 1502) = 108 takes, 352 min of
music, dealt evenly to both T4s (176 min each). About 2 GPU hours by the speed of run 4 (54 takes, 180 min of
music, 0.9 h).

On the laptop:

```
python3 kaggle/laptop/kjob.py run JOB=render BRANCH=claude/project-thread-7xq7od TRACKS_YAML=baseline-run/tracks.yaml TRACKS=queue TAKES=2 FIRST_SEED=1501 DIT_MODELS='["xl-turbo"]'
```

Results go to project files `baseline-run/<act>/` (file names `<slug>-<seed>-xl-turbo.mp3`).
