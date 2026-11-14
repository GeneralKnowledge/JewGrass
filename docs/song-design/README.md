# Song Design

A stage before generation. Every song gets a design file that decides what the music *does*, not just what it sounds like, and that file is compiled into what ACE-Step reads.

```
story beat ──> Song Design (docs/album/design/<slug>.yaml) ──> compile ──> caption + tagged lyrics + key/meter/tempo/length ──> laptop
                      ^                                                                                                     |
                      └──────────────── arrangement review (listening checklist below) <──────── generated takes <──────────┘
```

Pilot: [Gav's Song](../album/design/gavs-song.yaml), [Brass Plate](../album/design/brass-plate.yaml) and [Bought and Sold](../album/design/bought-and-sold.yaml), three very different problems (a folk ballad, a comic patter song, a monumental protest track). The rest of the album waits until we've heard these.

## What a design contains

| Field | What we decide |
|---|---|
| `approach` | The songwriting approach (see the table below). No two neighbouring tracks share one. |
| `identity` | One sentence: what the song is. |
| `render` | Tempo, key, meter, length. These four go straight to ACE-Step. |
| `palette` | The instrument world, and `left_out`: what this song deliberately does *not* use. |
| `roles` | Each instrument's job, and where it enters and leaves. |
| `harmony` | Chords per section, and how the song resolves. |
| `motifs` | Which [library motifs](motifs.md) appear and how they're transformed. |
| `sections` | Per section: the tag (with its arrangement cue), a dynamics level 1-5, the one idea and one image of each verse, and the lyrics. |
| `ending` | How it ends on purpose. |
| `caption` | The caption, written from all of the above, describing the arc not just the palette. |
| `lyric_audit` | What the lyric edit changed and why, with scores. |

## What ACE-Step actually hears (be honest about this)

ACE-Step reads text, not notation. So the design reaches the model through five channels:

1. **Key and meter** as real conditioning (`keyscale`, `timesignature`), plus tempo and length. These were never set before; the model guessed. Needs a two-line change to `gen.py` on the laptop (read `BAND_KEY` and `BAND_TIMESIG` from the environment and pass them as `keyscale=` and `timesignature=`).
2. **A focused palette.** Each song now asks for its own instruments and leaves the album's stock mix (banjo, fiddle, 808, trap drums, glitch, nu-metal) out unless the design gives them a job. This is probably the change you'll hear most.
3. **Arc in the caption** ("guitar alone at the start... bridge drops to guitar and voice... instruments fall away one by one").
4. **Arrangement cues in the section tags**, e.g. `[bridge - everything drops out, guitar and voice only, almost whispered]`.
5. **The lyrics**, which shape phrasing and section length.

What it can't do yet: follow a written chord chart or play a specific melody. The chords and motifs in the designs are the composer's plan, and they reach the model only as descriptions ("a slow rising bagpipe melody"). If the pilots work, the next step for real motifs is cheap: render a motif as a short audio intro and let ACE-Step continue from it with the repaint tool we already have, so the actual tune is in the song. No new installs needed.

## Songwriting approaches

| Approach | What it achieves | Pilot |
|---|---|---|
| First-person confession | Character reveals more than intended | Gav's Song |
| Narrative storytelling | A sequence of events, not a list of accusations | Gav's Song, Brass Plate |
| Dialogue | Two characters arguing or negotiating | (Exit Strategy, Ermine and Ink) |
| Dark comedy | The villain exposes himself | Brass Plate |
| Emotional ballad | The human cost | Gav's Song |
| Protest anthem | Something for the audience to sing | Bought and Sold chorus |
| Spoken documentary | History without forced rhymes | Bought and Sold verses |

## Lyric edit (scored separately, 1-5)

1. **Character voice.** Could only this person say these lines? Dialect is consistent, vocabulary fits their world.
2. **Narrative progression.** Does each verse move the story on? One main idea per verse.
3. **Imagery.** One memorable, concrete, original image per verse. No stock phrases ("ancient glory" is allowed only when it's Burns).
4. **Singability.** Lines fit the tempo; stressed syllables land on the beat; verses that aren't sung are allowed to breathe instead of rhyming.
5. **Hook.** A chorus or line the listener can sing back after one listen.

Also check: the song's sign-off ("due Friday") has moved on from the last time it was used ([the arc](motifs.md#due-friday)); no spoken outros; no real names.

## Arrangement review (after a take is generated)

Listen with these questions, in order:

1. Is the chorus clearly different from the verse (instruments, energy, register)?
2. Is there a recognisable instrumental hook?
3. Did the instruments the design left out stay out?
4. Do instruments fight for the same space (for example a fiddle and a vocal in the same range)?
5. Does it build towards the climax the design asked for, and drop where it asked to drop?
6. Does the ending sound intentional?
7. Is there enough change across the whole length?

The laptop's existing analysis (`tools/analyze.py`, section loudness and vocal activity) can answer 5 and 7 with numbers; the rest is listening.

## Instrument worlds (sound follows the money)

| World | Tracks | Instruments | Feel |
|---|---|---|---|
| History | Parcel of Rogues, Bought and Sold | Pipe lament, low strings, marching drums, deep male choir, distorted old textures | Old, monumental, angry |
| The fund | Natural Capital, Sunshine Trust | Rhodes, glassy synths, soft percussion, clean sub, lounge piano, expensive strings, artificial hold music | Beautiful, wealthy, empty |
| Paperwork and power | Brass Plate, Ermine and Ink | Typewriter, harpsichord, pizzicato, brass, mechanical rhythms, sharp electronic stabs | Bureaucratic, theatrical, ridiculous |
| Industry | Cold Hame, Swim at Your Own Risk | Industrial bass, metallic percussion, distorted guitars, hydraulic noise, dark synths | Powerful, oppressive |
| The village | Gav's Song, We Bought the Hill | Acoustic guitar, fiddle, accordion, bodhran, upright bass, whistles, then full pipes | Human, communal, imperfect |

The band's own country trap sound (banjo, 808, trap drums, glitch) belongs to the band and the villains who borrow it, not to every track. Instruments worth trying in later designs: hammered dulcimer, pump organ, concertina, bouzouki, cello, brass band, prepared piano.

## Tools

```
python3 tools/design/compile_design.py docs/album/design/<slug>.yaml build/design   # caption, lyrics, meta.env, run.sh
python3 tools/design/motif_audition.py docs/song-design/motifs.yaml build/motifs    # rough WAVs of the five motifs
```

Both need only PyYAML and numpy. The compiled caption, lyrics, key and meter are also written into [`tracks.yaml`](../album/tracks.yaml) for the laptop.

Not adopted yet: music21, Tonal, Strudel, isobar, Basic Pitch, Surge XT. Each becomes worth installing only for a specific job (Basic Pitch to pull a melody out of a good take so it can become a motif; Surge XT if we render motif intros with better sounds). Nothing here needed them.
