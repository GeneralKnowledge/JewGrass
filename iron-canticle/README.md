# IRON CANTICLE: CANDLES FOR A DEAD SUN

A grimdark sci-fi concept album sung as anthemic heavy/power metal: a powerful clean male lead, huge gang-choir
choruses, galloping double kick and thick, gritty guitars. Every line is sung; no rap. Raw and heavy, never
clean or glossy. (v2, Nov 15: Reuben found v1's hymn-trap rap verses "too much rap like singing"; v1 is kept in
`previous/v1-hymn-trap/`. v3, Nov 15: v2 renders were "too clean", so the captions now ask for a raw, gritty,
not-overproduced sound and build.py rejects clean/polished/glossy/symphonic wording.) Inspired by the genre in general and fully original: no franchise names, factions, places,
terms or quotes anywhere (`build.py` checks every lyric, title and caption against `banned_lore`).

**The world.** Vigil is a city covering a whole world, twelve thousand years old, lit by candles because its
sun is dying. The Synod, a priesthood, keeps the faith, the ledgers and the levy: every tenth child goes to the
war at the edge of the system. **Kell** is taken, survives a frozen moon, deserts, and comes home to the
Undercroft, where a murder cult sings to Cull, the Red Saint. He joins. The cult rises, the Synod falls, and
the new rulers light the same candles in the same spire. Nothing is saved; the sun keeps going out.

## Tracklist (all ten designed; Reuben's round 2 picks: Tithe Bell 3112, Trench Psalm 3312, Prayer 3711)

| # | Title | Family | Story | Length |
|---|---|---|---|---|
| 01 | **Tithe Bell** | cathedral | Levy day. The Herald reads the order, a mother asks the count to skip her door | ~1:50 |
| 02 | **Every Tenth Child** | cathedral | The mother, alone, the night after | ~2:30 |
| 03 | **Trench Psalm** | war | The frozen moon. Teague dies, the psalm doesn't change, the sun is a red ember | ~2:30 |
| 04 | **Engine Litany** | machine | The factory-chapels that build the guns, sung by the machine-priests | ~3:00 |
| 05 | **Deserter** | war | Kell runs, stowed on a supply ship home | ~2:45 |
| 06 | **The Undercroft** | undercroft | Under the city: almost still, a heartbeat and a bell | ~1:20 |
| 07 | **Prayer** | cult | The cult's hymn. A friend's song, adapted with permission | ~3:35 |
| 08 | **The Silent Court** | cult | Kell's oath and first killing, told by Kell | ~3:00 |
| 09 | **When Kingdoms Fall** | war + cult | The rising. The spire burns. Heaviest track | ~4:00 |
| 10 | **Candles for a Dead Sun** | cathedral | The new rulers light the same candles. The bell rings once and isn't answered | ~3:30 |

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

## Renders

- Round 1 (v1, hymn trap): seeds 3101/3102, 3301/3302, 3701/3702, project files `iron-canticle/round1/`.
- Round 2 (v2, sung metal): seeds 3111/3112, 3311/3312, 3711/3712, project files `iron-canticle/round2/`.
- Round 3 (v3, gritty, whole album): two takes per track, project files `iron-canticle/round3/`. The pilots keep
  Reuben's picked seed plus one new one. Seeds are 3000 + track x 100 + 11/12.

```
python3 kaggle/laptop/kjob.py run JOB=render \
  TRACKS_YAML=https://raw.githubusercontent.com/GeneralKnowledge/JewGrass/claude/project-thread-pku6cs/iron-canticle/tracks.yaml \
  TRACKS='["canticle-tithe-bell","canticle-every-tenth-child","canticle-trench-psalm","canticle-engine-litany","canticle-deserter","canticle-the-undercroft","canticle-prayer","canticle-the-silent-court","canticle-when-kingdoms-fall","canticle-candles-for-a-dead-sun"]' \
  SEEDS='{"canticle-tithe-bell":[3112,3113],"canticle-every-tenth-child":[3211,3212],"canticle-trench-psalm":[3312,3313],"canticle-engine-litany":[3411,3412],"canticle-deserter":[3511,3512],"canticle-the-undercroft":[3611,3612],"canticle-prayer":[3711,3713],"canticle-the-silent-court":[3811,3812],"canticle-when-kingdoms-fall":[3911,3912],"canticle-candles-for-a-dead-sun":[4011,4012]}'
```

A laptop render of Prayer with the friend's mix as the sound reference is still pending (seed 3711).
