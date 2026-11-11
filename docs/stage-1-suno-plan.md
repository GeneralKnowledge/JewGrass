# Stage 1: prove the sound on free Suno

**Goal:** find out whether free Suno can produce a convincing original band in
dark country trap, bluegrass and glitch. We aren't aiming for album
production, distribution or perfect voice consistency yet.

**Budget:** 14 generations (about 28 outputs, since Suno usually returns two),
spread over a few sessions as the free daily credits allow. Use **Custom mode**
and **v6-mini** for everything.

**Rules for every prompt**
- No artist names, song titles or lyrics from the reference catalogue. Describe
  the musical parts instead.
- Change one thing at a time. If a result is better, we need to know whether
  the music, the voice or the words made it better.
- Keep every output you score, even the bad ones, and log it in
  [`experiment-log.md`](experiment-log.md).

Lyrics live in [`lyrics.md`](lyrics.md). The parked local-generation plan is in
[`stage-2-local-option.md`](stage-2-local-option.md).

---

## How the band works (recommended)

**Yes: three performers who collaborate on records, not three AI voices taking
turns.** In practice that means:

- **Every song has an owner.** The Frontman owns most tracks. The others show up
  for a reason, such as a contrast verse, a hook or a bridge, and not to fill
  a quota.
- **Roles stay stable.** The Frontman carries verses and chant hooks. The
  Wildcard is the disruption: one feature verse, ad-libs, a breakdown. The Ghost
  is the melody: choruses, harmonies, intros and outros, plus the occasional
  solo song.
- **No song splits 8 bars three ways.** Even trade-offs sound like a showcase,
  not a band.
- **Recurring identity across songs matters more than variety inside one song.**
  The test of a member is whether you recognise them on a different track.

A possible five-track EP shape:

| # | Owner | Others |
|---|---|---|
| 1 | Frontman | Ghost on the outro |
| 2 | Frontman | Wildcard feature verse |
| 3 | Wildcard | Frontman on the hook |
| 4 | Ghost (solo) | none |
| 5 | Frontman | all three, the only full-band track |

Stage 1 tests whether Suno can support this. If two voices in one generation
keep blurring together, we'll stitch sections in an editor later instead
of dropping the idea.

---

## How to fill in Suno Custom mode

- **Lyrics:** paste from `lyrics.md`, tags included. Suno reads `[Verse]`,
  `[Chorus]` and `[Intro]` in any capitalisation. Notes such as
  `[Verse - fast, distorted]` are requests and may be ignored.
- **Styles:** the prompt from this doc. Suno weights the first few descriptors
  most, so the most important words go first.
- **Exclude styles** (if your plan shows it):
  `country pop, polished, glossy, clean vocals, EDM, stadium reverb`
- **Title:** use our working title plus the generation ID, e.g.
  `Dirt Under the Nails S1`, so outputs are easy to match to the log.

In multi-voice songs Suno doesn't know our member names. Before pasting
`Gravel and Static` or `Porchlight Funeral`, replace each name in the section
tags with that member's **vocal block** from Step 2 (e.g.
`[Verse - Frontman]` becomes `[Verse - low gravelly male rap, southern drawl]`).

---

## Step 1: Sound (5 generations)

**Question:** can Suno combine banjo, country instrumentation, aggressive trap
drums, distorted 808s and glitch convincingly?

**Lyrics:** *Dirt Under the Nails* for all five, unchanged.

| ID | Styles |
|---|---|
| S1 | **A, core:** Dark underground country trap, aggressive southern hip-hop, raw banjo riffs, distorted electric guitar, deep sliding 808 bass, hard-hitting trap drums, rapid hi-hats, gritty outlaw atmosphere, rough masculine rap vocals, rhythmic Southern drawl, explosive chant chorus, dirty analogue saturation, cinematic rural darkness |
| S2 | **B, electronic:** Experimental industrial trap fused with outlaw country, heavily distorted sub bass, glitching banjo samples, chopped acoustic guitar, aggressive syncopated drums, metallic percussion, sudden beat switches, stuttering vocal chops, chaotic digital textures, menacing low-register rap vocals, raw underground mix |
| S3 | **C, bluegrass:** Dark Appalachian bluegrass fused with southern trap, frantic banjo picking, fiddle melodies, dirty slide guitar, deep 808 bass, punchy half-time drums, rough acoustic textures, ominous cinematic atmosphere, aggressive country rap, unusual melodic hooks, raw live-band energy |
| S4 | **D, melodic:** Haunting outlaw Americana blended with dark southern trap, atmospheric acoustic guitar, mournful fiddle, distorted low bass, restrained trap percussion, haunting female country vocals, gritty male rap verses, layered harmonies, emotionally charged chorus, cinematic night-time atmosphere, experimental production |
| S5 | **Push the winner.** Take the best of S1–S4, keep its Styles and add **one** of the experimental pushes below |

**Pushes for S5.** Pick the one that fixes the winner's weakest point:

| If the winner... | Add this |
|---|---|
| lacks weight | `clipped distorted 808 bass that breaks up on every hit, tape saturation, sparse drums` |
| sounds too normal | `chopped stuttering banjo samples, bitcrushed guitar, sudden dropouts and tape stops, digital noise between phrases` |
| feels samey all the way through | `beat switch halfway, second half double-time bluegrass breakdown with frantic banjo and claps` |
| sounds too polished | `lo-fi porch recording, foot stomps as percussion, cassette hiss, warped tape, unpolished` |
| sounds like country pop | Move the trap and 808 words to the front, delete `melodic`, `haunting` and `emotionally charged` |
| lost the banjo | Put `banjo` in the first three words and change `[Instrumental]` to `[Banjo Solo]` |

**After Step 1:** write the winning Styles prompt into the log as **CORE**. Steps 2
and 3 build on it.

---

## Step 2: Members (6 generations)

**Question:** can we get three voices that are clearly different people?

Styles = **vocal block first, then CORE**. If CORE mentions a vocal, delete it.

### Frontman (lyrics: *Dirt Under the Nails*)

| ID | Vocal block |
|---|---|
| M1 | low gravelly male baritone rap, heavy southern drawl, laid-back behind-the-beat flow, dry close vocal |
| M2 | deep menacing male rap, quiet controlled almost whispered verses exploding into a shouted chant chorus, heavy drawl, low ad-libs |

Pick the voice that could carry five songs without getting tiring, then use
its vocal block for the Frontman from here on.

### Wildcard (lyrics: *Rattlesnake Sunday*)

| ID | Vocal block |
|---|---|
| M3 | fast raspy male rap, higher register, erratic triplet flow, abrupt stops, manic laughter ad-libs, distorted bitcrushed vocal, dark humour |
| M4 | nasal sneering male rap, rapid-fire delivery, stuttering vocal edits, megaphone distortion, confrontational |

### Ghost (lyrics: *Creek Don't Remember*)

| ID | Vocal block |
|---|---|
| M5 | haunting female alto, Appalachian folk inflection, high lonesome tone, sparse vibrato, close intimate vocal, layered harmonies on the chorus |
| M6 | haunting male tenor, high lonesome Appalachian tone, cracked fragile falsetto, gospel-tinged harmonies |

For the Ghost, also trim CORE down: drop `aggressive` and `explosive`, and add
`restrained trap drums, distorted 808 entering late`.

**Distinctness check:** play the three picks back to back with your eyes shut.
If any two could be the same person, push that member's vocal block further
before Step 3.

**If your plan offers Personas,** save one from each member's best output and
use it in Step 3. If not, reuse the exact vocal block word for word. Either way,
Step 3 tells us how well the voice survives on a different song.

---

## Step 3: Songs (3 generations)

**Question:** can we make complete tracks with hooks you remember, not just
interesting 30-second ideas? This step also tests whether members are recognisable
on a second song.

| ID | Song | Styles |
|---|---|---|
| T1 | *Dirt Under the Nails*, full version. Add a `[Bridge]` and a third chorus, and run to 3+ minutes | Frontman vocal block + CORE + push from S5 if it helped |
| T2 | *Gravel and Static* | `two contrasting male rappers, first voice` + Frontman block + `, second voice` + Wildcard block + `, call and response chant chorus,` + CORE |
| T3 | *Porchlight Funeral* | Frontman block for verses + `, haunting` + Ghost block (female or male, whichever won) + `on the chorus,` + CORE with the Ghost trims |

---

## Scoring

Rate each output 1–5 per category. Score out of 100 =
(Genre×5) + (Production and bass×4) + (Vocal character×4) + (Country instrumentation×3) + (Originality×2) + (Hook×2).
Above 80 is worth developing. A track that's huge on production but weak on
vocal identity isn't automatically a keeper.

Three extra checks, logged as yes/no:

- **Hook test** (Step 3): the next day, can you hum the chorus without replaying it?
- **Same person?** (Steps 2–3): does each member sound like the same person across both outputs and across songs?
- **Different people?** (Steps 2–3): could any member be mistaken for another?

---

## Stage 1 decision: what to bring back here

After the 14 generations, share the best outputs (links or audio) and the
filled-in log. Then we decide, part by part:

| Part | Keep on Suno if... | Consider local (Stage 2) if... |
|---|---|---|
| Sound | Step 1 has an output above 80 | Banjo, bass and glitch never coexist |
| Voices | Members pass both identity checks | Members drift or blur between songs |
| Songs | T1–T3 hold up past the first minute | Arrangements collapse or repeat |
| Mix | Rough in a good way | Always too glossy, even with excludes |

Commercial note: free-plan outputs don't get commercial rights later. Anything
we might release must be regenerated on a paid plan.
