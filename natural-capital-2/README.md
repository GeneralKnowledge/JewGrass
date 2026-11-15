# Book of Names (Natural Capital, part 2)

The second album by the band that made *Natural Capital*. Reuben asked for it on 2026-11-15: an album about "the files" and how everyone is guilty, with religious themes and the obvious conspiracies.

## Ground rules

- **The case is fictional.** The real public scandal around a dead financier's files, flight logs, private island and sweetheart plea deal gives the shape. Every person, island, lodge and institution here is invented. No real person is named, portrayed or accused. Before release, search every invented name to make sure nobody real shares it.
- **"Everyone is guilty" is about the system, not a list of names.** The guests, the bank, the prosecutor, the paper, the college, the guards, the neighbour, and the public who turned it into memes and a fight between sides. Villains are jobs ("the Banker", "the Prosecutor"), not people.
- **The conspiracies are sung as questions, not answers.** Broken cameras, sleeping guards, a skipped minute of tape, a client list that exists and then doesn't. *Two Guards Asleep* asks them; its bridge asks whether blaming one dead man is just easier.
- **Survivors are treated with care.** Nothing is explicit. The survivor, Rose, is the only character who isn't guilty, and she gets the album's biggest ballad.
- **Same band, same sound, more singing.** Of the 12 songs, 8 are sung-led and 4 are rap-led.

## The story

Morag, the reporter from *Natural Capital*, now works in London. An unmarked parcel arrives: the files of **Silas Varne**, a financier who has just been arrested again. There are flight logs, dinner lists and a black book. Her brother Ewan recognises the name. Varne owned **Corran Lodge** across the loch from their glen, and every August the helicopters came and the village looked away. Morag follows the book: the TV preacher who blessed his foundation, the bank that moved his money for twenty years, the dinners on his island **Sabbath Cay**, and the prosecutor who gave him a sweetheart deal the first time. A survivor, Rose, tells her story. Then Varne dies in his cell before trial, with two guards asleep and the cameras off, and every question is left open. When the files are finally released, the powerful names are blacked out and a survivor's name is printed by mistake. The band's verdict is that there was never one monster. It took a whole system, and everyone who knew is in the book, including us. In the end Morag comes home, and the survivors write their own names in a different book.

## The service

The album runs like a church service and a trial at once: *All Rise* (the call to worship and to court), *Confession*, *Indulgences* (absolution for sale), *The Last Supper*, *Pilate* (the hand-washing), *Redacted* (the revelation that reveals nothing), *Judgment Day* (with a verse of the *Dies irae*, public domain) and the *Book of Names*, the book of life that answers the black book.

**The running joke.** Part 1's villains signed off "due Friday". Here every villain ends with their own excuse and an "Amen", sung as the last line: "No further comment. Amen." "Tax-deductible. Amen." "Grace is said. Amen." "Case closed. Amen." The finale answers them: "Write it in the book of names. Amen."

## Cast

| Character | Who they are | Song | Voice |
|---|---|---|---|
| **The band** | Narrators and judges | *All Rise*, *The Black Book*, *Two Guards Asleep*, *Judgment Day* | Frontman's angry Scottish rap, Ghost's haunting vocal, choir |
| **Ewan** | Morag's brother, still in the glen | Two voice notes | Reuben's own speaking voice (or the RVC voice) |
| **Morag** | The reporter | *Redacted*, *Book of Names* | Scottish female lead |
| **Rose** | A survivor | *Rose*, bridges of *Confession* and *Last Supper*, *Book of Names* | Fragile female vocal (Ghost) |
| **Silas Varne** | The financier, the host | *Last Supper on Sabbath Cay* | Silky deep half-sung R&B |
| **The Reverend** | A TV preacher on the guest list | *Confession* | Powerful gospel lead and choir |
| **The Banker** | His bank for twenty years | *Indulgences* | Smug posh English rap |
| **The Prosecutor** | Gave him the sweetheart deal | *Sweetheart Deal* | Calm deadpan rap |
| **The Editor, the College, the Neighbour** | The people who looked away | *Pilate* | Soft sung verses |

## Track list (14 tracks, 41:30)

| # | Title | Length | Style | Sung or rap |
|---|---|---|---|---|
| 1 | All Rise | 2:00 | Gospel folk-trap hymn, organ and choir | Sung |
| 2 | Voice Note: Helicopters | 0:40 | Ewan | Voice |
| 3 | The Black Book | 3:40 | Boom bap into trap, banjo loop | Rap, sung chorus |
| 4 | Confession | 3:00 | Dark gospel soul over trap | Sung |
| 5 | Indulgences | 3:20 | Glossy dark UK trap | Rap, sung hook |
| 6 | Last Supper on Sabbath Cay | 4:00 | Slow dark R&B trap | Half-sung |
| 7 | Rose | 4:20 | Piano and strings ballad | Sung |
| 8 | Sweetheart Deal | 2:40 | Cold sparse trap | Rap, sung hook |
| 9 | Pilate | 2:50 | Soulful folk trap | Sung, rap bridge |
| 10 | Voice Note: Cell | 0:30 | Ewan | Voice |
| 11 | Two Guards Asleep | 3:10 | Nu-metal trap | Rap and gang chant |
| 12 | Redacted | 3:30 | Dark glitch pop | Sung |
| 13 | Judgment Day | 4:20 | Epic trap, choir, nu-metal chorus | Rap verses, sung chorus |
| 14 | Book of Names | 3:30 | Celtic folk hymn, full pipes on the last chorus | Sung |

Pipes are kept back for the very end: they only play in the last chorus of *Book of Names*.

## Files and rendering

- `design/NN-slug.yaml`: one file per track: who sings it, its purpose, render settings, caption and lyrics. Voice notes hold Ewan's script.
- `queue.yaml`: render order, strongest songs first.
- `build.py`: checks the caption rules (no crackle, hiss, static, vinyl, gramophone or typewriter; nothing circus-like; no clean or polished mix; start straight in; end sung; 260 s or less) and writes `tracks.yaml`. Run `python3 natural-capital-2/build.py` after editing a design.
- `tracks.yaml`: GENERATED, the same shape as `docs/album/tracks.yaml`. Kaggle renders it as a `RENDER_FIRST` batch:

```python
{"tracks_yaml": "https://raw.githubusercontent.com/GeneralKnowledge/JewGrass/claude/project-thread-rla07g/natural-capital-2/tracks.yaml",
 "tracks": "queue", "takes": 2, "first_seed": 5101, "models": ["xl-turbo"]}
```

That is 12 songs × 2 takes = 24 XL-turbo takes (about 41 minutes of audio per take set), split across both GPUs. Results go to project files `book-of-names/round1/`.

- `alt-lifeline/`: the first idea for part 2 (water company and ferry fiasco), started before Reuben redirected it. Kept, not used.
