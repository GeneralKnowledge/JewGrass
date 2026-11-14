# Matching Jackets

A fictional "new age" boy band from Ashbrae, a fictional Scottish new town: five lads who met in the
leisure centre choir. Late-90s / early-2000s boy band songwriting (stacked harmonies, pre-choruses that lift,
the key change, standing up off the stools) with modern production and one light, funny rap section per song.
They know every boy band trope and use them on purpose, with a wink, but mean the feelings underneath.

Everything is original: the genre and era are the only references. No real artist, song, person, brand or
place is named or imitated in any file that goes into generation.

| Member | Role | Voice |
|---|---|---|
| **Cal** | lead vocal, the sincere one | warm clear tenor |
| **Ollie** | pre-choruses and hooks, the youngest | bright breathy pop tenor |
| **Dev** | falsetto, ad-libs, bridges, the key change | soaring falsetto |
| **Big Jon** | low harmony anchor | deep baritone |
| **Kez** | the rapper, the funny one | cheeky conversational British rap |

## Pilot

| # | Song | Purpose | Tempo / key | Length |
|---|---|---|---|---|
| 1 | [Left on Read](design/left-on-read.yaml) | the upbeat single: opens on the hook a cappella, full key change at the end | 112, E major | 3:30 |
| 2 | [Meal Deal](design/meal-deal.yaml) | the fun one: Kez's song, 2-step garage pop, the crush on the girl on the till | 132, F major | 2:20 |
| 3 | [Kept Your Hoodie](design/kept-your-hoodie.yaml) | the ballad: nineteen days in her hoodie, and she opens the door wearing his | 70, Bb major | 3:50 |

Every song starts straight in (voice or hook first), ends sung or played (nothing spoken), and has a different
length, tempo and hook shape.

## Files

| File | What it is |
|---|---|
| [`band-dna.yaml`](band-dna.yaml) | Members, home town, sound, production and writing rules, shared caption core, acceptance criteria |
| `design/*.yaml` | One Song Design per track: palette, harmony, the moment, sections with performer and tag, lyrics, caption |
| [`build.py`](build.py) | Compiles DNA + designs into `tracks.yaml` and checks them (tag length, lines per bar, hiss-prone words) |
| [`tracks.yaml`](tracks.yaml) | Generated. Same shape as `docs/album/tracks.yaml`, so the Kaggle notebook renders it as is |

```
python3 matching-jackets/build.py     # rebuild tracks.yaml after editing a design; exits 1 on any warning
```

## Rendering

Round 1 on Kaggle (XL turbo, the default), two takes per song, seeds 9501 and 9502:

```
TRACKS_YAML = "https://raw.githubusercontent.com/GeneralKnowledge/JewGrass/claude/project-thread-hhd444/matching-jackets/tracks.yaml"
TRACKS = "queue"
SEEDS = {"mj-left-on-read": [9501, 9502], "mj-meal-deal": [9501, 9502], "mj-kept-your-hoodie": [9501, 9502]}
```

About 10 minutes of music in total. The main thing to listen for: does it sound like a group (several voices,
stacked chorus) or one singer with backing, and does the rap come out light and funny.
