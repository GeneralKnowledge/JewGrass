# DAW export

Goal: every JewGrass song can be opened in a DAW as an editable project: one track per part, real MIDI notes, sections, tempo and meter, lyrics, and the original audio next to it for comparison.

## Shape

```
Song Design yaml ──from_design.py──┐                          ┌─> REAPER project (.rpp)          first exporter
  (composition route)              ├─> arrangement.json ──export.py──> multitrack MIDI (SMF type 1)   any DAW
stems from the laptop ─from_audio.py┘   (DAW-independent)      ├─> per-track MIDI
  (reconstruction route)                                        └─> audio, lyrics, sections, README
```

- **arrangement.json is the source of truth**, not MIDI and not any DAW's file. Schema at the top of [`tools/daw/arrangement.py`](../tools/daw/arrangement.py). Time is in quarter-note beats from bar 1, so it never depends on a DAW's tick rate. Later exporters (DAWproject, Ardour, an Ableton template) read the same file.
- **Composition route** (the long-term one): the Song Design already holds sections, bar counts, tempo, key, meter, harmony and lyrics. `from_design.py` turns that into playable parts. Today it makes three: a fingerpicked guitar arpeggio, a bass (root and fifth) and a chord guide. These are sketches proving the pipe works; real part writing (motifs, fills, entries and exits from `roles`) is the next step.
- **Reconstruction route**: the laptop separates stems (one heavy job at a time); `from_audio.py` runs in the cloud on CPU. Basic Pitch (Apache-2.0) transcribes pitched stems; drums use a three-band onset detector (kick, snare, hat). Each transcribed track keeps its stem audio, muted, so you can A/B. It is an editable approximation, not the exact performance.
- **Why REAPER first**: its project file is plain text, so we write it directly with no library, no template and no GUI automation. MIDI lives inside the project and is editable. REAPER costs about $60 but its evaluation is fully functional. For a free DAW, the multitrack MIDI file opens in Ardour, LMMS and others.

## Composition engine (first cut): Brass Plate

`compose.py` reads the Song Design plus a **parts file** (`docs/album/parts/<slug>.yaml`) and writes every note itself. Nothing is inferred from audio.

The parts file has two blocks:
- `charts`: the exact chord per bar for each section, with optional slash bass (`Cm/Bb`). The design's harmony text can't be played as written, so this is where it becomes notes.
- `parts`: one entry per instrument, giving the generator that writes it, its GM program, and the sections it plays in. The entries and exits come from the design's `roles`.

Brass Plate has 8 tracks and about 3,000 notes:
- **Typewriter**: 16th-note clacks with a carriage-return bell.
- **Stamp and drums**: the rubber stamp on 2 and 4, plus a desk bell at each section start.
- **Harpsichord riff**: the money motif `C G Db C` closes every 8 bars of verse.
- **Tuba**: oom-pah with chromatic approach notes, walking C Bb Ab G in the verses, holding a pedal C under the corruption progression in verse 3.
- **Pizzicato strings**: drop to uneasy quarter notes in verse 3.
- **Brass**: stabs on every chorus chord change, and the final C major chord.
- **808 hits**: chorus downbeats only.
- **Synth stabs**: on the punchlines from verse 2.

Generators are deterministic: the same seed gives the same notes. They're 4/4 only so far.

**Sound**: `export.py --preview` renders the MIDI through a General MIDI SoundFont (TimGM6mb, which ships inside pretty_midi) with tinysoundfont, a pure-Python renderer. The render is added to the REAPER project as a muted reference track. It's a cheap GM sound, there to check the notes. It won't compete with an ACE-Step render on sound. The way to get both is to give ACE-Step the composed render as its audio reference (repaint or cover), so the arrangement is ours and the performance is ACE-Step's.

## Usage

```bash
pip install mido numpy soundfile pyyaml          # export and composition route
pip install pretty_midi; pip install --no-deps tinysoundfont   # --preview (SoundFont render)
pip install "basic-pitch[onnx]" librosa           # reconstruction route only (CPU, Python 3.11)

python3 tools/daw/from_design.py docs/album/design/gavs-song.yaml build/daw/gavs-song/arrangement.json
python3 tools/daw/export.py build/daw/gavs-song/arrangement.json "build/daw/gavs-song/Gav's Song - DAW project" --preview

python3 tools/daw/from_audio.py build/daw/deed/arrangement.json --title "Deed to the Glen" --slug deed-to-the-glen \
    --ref mix.mp3 --bpm 140 --stem bass=bass.mp3 --stem guitar=guitar.mp3 --stem other=other.mp3 \
    --stem drums=drums.mp3 --stem vocals=vocals.mp3 --transcribe bass,guitar,other,drums

python3 tools/daw/compose.py docs/album/design/brass-plate.yaml docs/album/parts/brass-plate.yaml build/daw/brass-plate/arrangement.json

python3 tools/daw/selftest.py build/daw/gavs-song/arrangement.json build/daw/selftest
```

If the basic-pitch install pulls in TensorFlow and it fails to import, uninstall TensorFlow (the ONNX model is used instead) and install `setuptools<81` (resampy needs `pkg_resources`).

## Synthetic test (known notes in, notes out)

`selftest.py` renders the Gav's Song sketch with a tiny synth, transcribes it back and scores it (pitch exact, onset within 50 ms). This is the best case: clean, synthetic, one instrument per stem.

| Part | Notes | Found (recall) | Correct (precision) | F1 |
|---|---|---|---|---|
| Guitar arpeggio, own stem | 624 | 100% | 90% | 0.95 |
| Bass, own stem | 208 | 99% | 70% | 0.82 |
| Chord guide, own stem | 312 | 99% | 57% | 0.72 |
| All three mixed, no separation | 1144 | 71% | 54% | 0.61 |

What it tells us: separation matters (0.61 without, up to 0.95 with); Basic Pitch rarely misses a note but adds ghost notes, so the per-role thresholds in `from_audio.py` were raised (the defaults doubled the note count). Real ACE-Step stems will score lower than this ceiling.

## Real test: Deed to the Glen (reconstruction)

The stems came from the laptop: BS-RoFormer vocals plus htdemucs_6s drums, bass, guitar, piano and other, at 140 bpm. Sections were estimated from the laptop's vocal analysis and are marked "(est.)". The guitar and piano stems are nearly silent (-48 and -42 dB), because Demucs put the banjo, fiddle and guitars into "other". So those two stay audio-only.

The transcriber isn't scored against true notes here, because there are none. Instead, each stem is compared, frame by frame, with its own MIDI on the loud frames:

| Part | Notes | Loud time covered by a MIDI note | Covered time where MIDI has the stem's main pitch class |
|---|---|---|---|
| Bass (808) | 236 | 40% | 94% |
| Other (banjo, fiddle, pipes, guitars) | 1026 | 88% | 92% |
| Drums (kick, snare, hat) | 1526 hits | n/a | n/a |

Where there is a MIDI note, it's mostly the right one. The 808 is the weak spot: long, gliding sub notes. Lower thresholds raise coverage to 70%, but at 1000 notes and 89% agreement, which means more mess to clean up, so the cleaner setting is the default. "Other" is several instruments in one stem, so its MIDI is a blend of all of them, not one playable part.

## Not done yet

- Instruments: MIDI tracks in the .rpp have no instrument loaded. The safe way is a REAPER track template made in Reuben's REAPER (with ReaSynth / ReaSamplOmatic or a free SFZ player per role), which the exporter then copies in. Writing plugin state blind would risk a project that doesn't open.
- Verified only by parsing: the cloud cannot download REAPER, so the .rpp was checked with an RPP parser and against the MIDI file (same 1144 notes), not opened in REAPER.
- Tempo maps (one constant tempo is assumed), automation, DAWproject export, Ardour session export.
