# LOWBRIDGE

A fictional UK drill collective from Southmere, a fictional British city: **Rook** (lead voice), **Vex**
(technical), **Vale** (reflective) and **LATCH** (producer). Cold, stripped-back, hard-hitting drill.
Everything here is original: the genre is the only reference, and no real artist, record, person or place
is named or imitated in any file that goes into generation.

The pilot, *Under the Bridge*, uses the existing JewGrass pipeline with a small amount of artist-specific
configuration. No band system yet: first we find out whether the concept works musically.

| File | What it is |
|---|---|
| [`artist-dna.yaml`](artist-dna.yaml) | The persistent identity: members and their delivery, shared world, sound, production rules, the shared caption core, writing rules, mix targets, visual identity, variation rules, acceptance criteria |
| [`design/no-signal.yaml`](design/no-signal.yaml) | 01 No Signal, Rook featuring Vale. 140 bpm, F minor. The introduction |
| [`design/weight-of-it.yaml`](design/weight-of-it.yaml) | 02 Weight of It, Rook, Vex and Vale. 144 bpm, C minor. The hard single |
| [`design/last-bus-home.yaml`](design/last-bus-home.yaml) | 03 Last Bus Home, Vale featuring Rook. 138 bpm, D minor. The reflective track |
| [`build.py`](build.py) | Compiles the DNA and designs into `tracks.yaml` and checks them |
| [`tracks.yaml`](tracks.yaml) | Generated. Same shape as `docs/album/tracks.yaml`, so the Kaggle notebook and the laptop render it as is |

Each design has the Song Design fields (render, palette, roles, harmony, sections with arrangement cues in
the tags, ending, caption, lyric audit) plus drum pattern, bass movement, the song's one planned "moment",
and a `performer` per section.

## How the songs reach ACE-Step

`build.py` joins the DNA's `caption_core` (the shared LOWBRIDGE sound) with each song's caption (its
tempo, key, instruments, voices and moment). Key, tempo, meter and length go in as real conditioning.
Arrangement moments ("drums and 808 cut out, voice and piano only") ride in the section tags, and each
member's delivery is written into the tag of every section he raps.

What we've learned from JewGrass and built in:
- ACE-Step follows key and tempo but not chord charts, so harmony, drum patterns and bass movement in the
  designs are the plan for the DAW stage; they reach the model only as words.
- Words for constant noise textures caused hiss before, so `build.py` refuses captions and tags with
  words like static, crackle, vinyl, texture or atmosphere.
- Spoken outros don't come through, so every outro is a short rapped line or instrumental.

```
python3 lowbridge/build.py      # rebuild tracks.yaml after editing a design; exits 1 on any warning
```

## The three voices

The pilot doesn't assume ACE-Step can keep three consistent voices in one recording. Round 1 renders whole
songs (2 takes each) with each member's delivery described in his section tags, and we listen for whether
it sounds like three people. Only if it doesn't do we go further: render each verse separately with its
member's profile and assemble them on the laptop with the repaint tool. A member voice counts as stable only
once it passes blind listening across several generations.

## Judging the pilot

Against `pilot_acceptance` in the DNA: group identity, member identity, production, lyrics, hooks,
arrangement, audio quality, and whether a second take keeps the same identity.
