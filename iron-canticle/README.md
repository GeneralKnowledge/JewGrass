# IRON CANTICLE: CANDLES FOR A DEAD SUN

A grimdark sci-fi concept album sung as anthemic heavy/power metal: a powerful clean male lead, huge gang-choir
choruses, galloping double kick and heavy guitars. Every line is sung; no rap. (v2, Nov 15: Reuben found v1's
hymn-trap rap verses "too much rap like singing"; v1 is kept in `previous/v1-hymn-trap/`.) Inspired by the genre in general and fully original: no franchise names, factions, places,
terms or quotes anywhere (`build.py` checks every lyric, title and caption against `banned_lore`).

**The world.** Vigil is a city covering a whole world, twelve thousand years old, lit by candles because its
sun is dying. The Synod, a priesthood, keeps the faith, the ledgers and the levy: every tenth child goes to the
war at the edge of the system. **Kell** is taken, survives a frozen moon, deserts, and comes home to the
Undercroft, where a murder cult sings to Cull, the Red Saint. He joins. The cult rises, the Synod falls, and
the new rulers light the same candles in the same spire. Nothing is saved; the sun keeps going out.

## Tracklist (outline; only the three pilots are written)

| # | Title | Family | Story | Length |
|---|---|---|---|---|
| 01 | **Tithe Bell** (pilot) | cathedral | Levy day. The Herald reads the order, a mother asks the count to skip her door | ~1:50 |
| 02 | Every Tenth Child | cathedral | The mother, alone, the night after | ~2:30 |
| 03 | **Trench Psalm** (pilot) | war | The frozen moon. Teague dies, the psalm doesn't change, the sun is a red ember | ~2:30 |
| 04 | Engine Litany | machine | The factory-chapels that build the guns, sung by the machine-priests | ~3:00 |
| 05 | Deserter | war | Kell runs, stowed on a supply ship home | ~2:45 |
| 06 | The Undercroft | undercroft | Under the city: almost still, a heartbeat and a bell | ~1:20 |
| 07 | **Prayer** (pilot) | cult | The cult's hymn. A friend's song, adapted with permission | ~3:35 |
| 08 | The Silent Court | cult | Kell's oath and first killing, told by Kell | ~3:00 |
| 09 | When Kingdoms Fall | war + cult | The rising. The spire burns. Heaviest track | ~4:00 |
| 10 | Candles for a Dead Sun | cathedral | The new rulers light the same candles. The bell rings once and isn't answered | ~3:30 |

| File | What it is |
|---|---|
| [`artist-dna.yaml`](artist-dna.yaml) | World, cast and voices, palette, motifs, five production families, caption core, avoid lists, banned franchise terms, round 1 plan |
| `design/*.yaml` | One Song Design per track (render settings, palette, harmony, drums, sections with tags and lyrics, ending, caption, lyric audit) |
| [`build.py`](build.py) | Copied from `lowbridge/build.py` (via NIGHTSHIFT FM); compiles DNA + designs into `tracks.yaml` and checks them |
| [`tracks.yaml`](tracks.yaml) | Generated. Same shape as `docs/album/tracks.yaml`, so the Kaggle notebook renders it unchanged |

```
python3 iron-canticle/build.py      # rebuild after editing a design; exits 1 on any warning
```

## Prayer

Written by a friend of Reuben's and used with permission. The original names a trademarked god from a fantasy
game setting and some of his lore, so six lines are changed (listed under `changes` in
[`design/07-prayer.yaml`](design/07-prayer.yaml)); everything else is the friend's words. The friend should
OK the changes before anything is released. Their mix is the ACE-Step sound reference for the laptop test,
kept only on the never-merged branch `claude/reference-inputs` (`reference-inputs/prayer-friend.mp3`).

## Round 1

Tithe Bell, Trench Psalm and Prayer (one per family), 2 takes each on Kaggle, XL turbo. v1 seeds 3101/3102,
3301/3302, 3701/3702 (in project files iron-canticle/round1/). v2 seeds 3111/3112, 3311/3312, 3711/3712. Plus one laptop render of Prayer with the friend's mix
as reference, for a blind A/B against the plain take.

```
python3 kaggle/laptop/kjob.py run JOB=render \
  TRACKS_YAML=https://raw.githubusercontent.com/GeneralKnowledge/JewGrass/claude/project-thread-pku6cs/iron-canticle/tracks.yaml \
  TRACKS='["canticle-tithe-bell","canticle-trench-psalm","canticle-prayer"]' \
  SEEDS='{"canticle-tithe-bell":[3111,3112],"canticle-trench-psalm":[3311,3312],"canticle-prayer":[3711,3712]}'
```
