# Laptop pipeline

How the songs are made locally on the band laptop (Dell G3 17, GTX 1060 6 GB,
8 GB RAM, headless Ubuntu Server). Everything lives in `~/band` on the laptop.
The scripts themselves are copied into [`laptop/`](../laptop/) in this repo, with install steps in [`laptop/README.md`](../laptop/README.md).

## 1. Generate (ACE-Step 1.5, 2B turbo)

```
~/band/gen.sh <caption-file> <lyrics-file> <seconds> <seed> <name> <bpm>
```

- Licence: ACE-Step 1.5 is MIT, so commercial use is allowed.
- A full 240 s song takes about 5–6 minutes end to end in one pass. VRAM stays
  flat at about 5.9 GB whatever the length.
- Lyrics use plain section tags: `[intro] [verse] [chorus] [bridge] [breakdown] [outro]`.
- The language model is off. Captions come from `docs/lyrics.md` and the song files.

### What we changed to make it run on a Pascal card

The stock install doesn't work on a GTX 1060. These fixes are in the `gen.py`
patch and are switched by environment variables:

| Problem | Fix | Variable |
|---|---|---|
| torch cu128 has no Pascal kernels | install the cu126 wheels and always run `uv run --no-sync` | none |
| fp16 gives all-NaN latents on Pascal | keep weights in fp16, but compute each layer in fp32 | `BAND_FP32_COMPUTE=1` |
| Model load took 11 minutes on 8 GB RAM | load the DiT tensor by tensor straight to CUDA | `BAND_DIT_DIRECT=1` |
| Offloading the DiT hung | never offload it | `BAND_OFFLOAD_DIT=0` |
| Eager attention blew up VRAM on long songs | memory-efficient SDPA attention | `BAND_ATTN=sdpa` (default) |

Downloads: the home connection drops, so use `HF_HUB_DISABLE_XET=1`, a retry
loop and a sha256 check on the big files.

## 2. Song folders

Each song gets its own folder:

```
~/band/songs/<slug>/
  00-gen/        raw takes, one per seed
  10-analysis/   tempo, beat grid, loudness, gaps, vocal activity
  20-stems/      vocals / instrumental (BS-RoFormer), dereverbed vocal
  30-vocals/     vocal work (later: the RVC-converted vocal)
  40-edits/      repaints and patches
  60-master/     masters; old versions are kept, new ones get -v2, -v3...
```

## 3. Analyse and pick a take

`tools/analyze.py`, `tools/stemcheck.py` and `tools/check_audio.py` (CPU venv
`tools/fx`: librosa, beat-this, pyloudnorm) report, for each take:

- tempo and how steady it is;
- silences and dropouts (for example 1.7 s of silence after an intro);
- where the vocal stem is active, which shows whether the spoken intro and
  outro are really there;
- loudness and true peak.

The best take is the one with no gaps, a steady tempo and the vocal parts present.

## 4. Repaint (fix a section without regenerating the song)

```
cd ~/band/ACE-Step-1.5 && uv run --no-sync python ../repaint.py SRC.wav caption.txt lyrics.txt BPM START END OUT_DIR SEED[,SEED...]
```

- Uses ACE-Step's `task_type=repaint` on a full 4-minute song.
- VAE-encodes in 5 s chunks, otherwise the 6 GB card runs out of memory.
- One process per seed, because a second seed in the same process also runs out.
- `tools/window_beats.py` snaps the window to bar lines, and `tools/repaint_check.py`
  compares the window before and after.
- Used for the Glen dropped beat (patch 7001, approved) and for missing spoken
  outros: repaint the last ~20 s with "spoken word outro, smug posh speaking
  voice" added to the caption, three seeds, then check the vocal stem for speech.
  That worked for Glen but not for Swim, and it is slow, so spoken outros are no
  longer chased: songs keep the ending they were generated with.

## 5. Master

`tools/master.py` (pedalboard): high-pass, gentle glue compression, EQ,
limiter, to -14 LUFS integrated and -1 dBTP. Reuben preferred this over
Matchering on Glen, so the Glen pedalboard master is the reference for the EP's sound.

Also available:

- **Stem master** (`tools/stem_master.py`, then `master.py`): split vocal and
  instrumental; vocal gets gate, 90 Hz high-pass, de-ess, 3:1 compression,
  presence EQ and a short reverb; instrumental gets a high-pass and glue; remix,
  then master to -14 LUFS / -1 dBTP → `60-master/stem-master.wav`. Done for Glen and Swim.
- **One sound for the EP:** Matchering a song against the Glen pedalboard master
  → `matchering-glenref.wav`. Done for Swim.

### Production upgrade (`tools/produce.py`)

A full stem-based production pass, driven by a per-song JSON config
(example: [`laptop/songs/deed-to-the-glen/production-v1.config.json`](../laptop/songs/deed-to-the-glen/production-v1.config.json)).

```
flock ~/band/.gpu.lock ~/band/heavy.sh ~/band/tools/fx/.venv/bin/python ~/band/tools/produce.py <config.json>
```

1. **Stems.** BS-RoFormer gives the lead vocal. Demucs `htdemucs_6s` (through
   audio-separator in the sep venv) splits the RoFormer instrumental into drums, bass,
   guitar, piano and other. On Glen the banjo, fiddle and guitars mostly land in "other".
2. **Measured, not fixed.** Each stem is measured first: noise floor, active level,
   crest factor and band balance. The gate sits 6 dB above the noise floor. The
   compressor threshold follows the active level. EQ closes half the gap to a
   genre target curve, capped at ±4 dB. Every value can be overridden in the config.
3. **Per stem.**
   - Drums: parallel compression above 120 Hz.
   - Bass: low-pass, parallel saturation for harmonics that read on small
     speakers, mono below 120 Hz.
   - Guitar, piano, other: EQ, a small 2.5 kHz carve, light saturation, slight widening.
   - The instrument bus ducks up to 2.5 dB at 1–4 kHz when the vocal is active.
4. **Vocal production.** Gate, high-pass, de-ess, compression and EQ, then a
   different space per section type:
   - spoken: dry with a short slapback;
   - rap: a small room;
   - bridge: a longer delay and reverb.

   Choruses get two detuned, delayed doubles panned left and right. Chant hooks get
   a quiet octave-down layer. Sections are detected automatically and written to
   `<config>.sections.json`, or can be given by hand in the config.
5. **Gentler master.** It aims for -14 LUFS / -1 dBTP, but the limiter may take at
   most 3 dB. If that leaves the song quieter than -14, it stays quieter rather than being squashed.
6. **Report.** LUFS, true peak, loudness range, L/R correlation, mono fold-down
   loss and low-end width, against the pedalboard master and the unmastered original.

First result, Glen `production-v1.wav` against `pedalboard-v2.wav`:

| | production-v1 | pedalboard-v2 |
|---|---|---|
| Loudness | -14.56 LUFS | -14.01 LUFS |
| True peak | -1.05 dBTP | -1.13 dBTP |
| Loudness range | 10.3 LU | 5.6 LU |
| Mono fold-down loss | 0.34 LU | 0.44 LU |
| Side vs mid below 120 Hz | -35 dB | -19 dB |

Known weaknesses for v2:

- Section detection is rough: it found only one chorus on Glen. Hand-written
  sections or lyric timestamps would be better.
- EQ hits its ±4 dB cap on near-empty stems (Glen's guitar and piano); stems more
  than 20 dB below the mix should skip EQ.
- The sidechain ducks 72% of the time. Its threshold should follow the vocal's
  active level, not its noise floor.
- No harmonies other than the octave yet, because there's no pitch tracking.

Always compare versions at matched loudness, because louder always sounds better.

## 6. Keeping the laptop connected

Heavy jobs used to drop the connection. The cause was CPU saturation and
thermal throttling (about 45,000 throttle events) starving `claude remote-control`,
not Wi-Fi or memory.

- Every heavy job, GPU or CPU, runs one at a time through the lock and `heavy.sh`:
  `flock ~/band/.gpu.lock ~/band/heavy.sh ...`. A CPU master running alongside a
  GPU repaint caused a second drop on 2026-11-14 (RAM 6.1 of 7.6 GB, throttling).
- `heavy.sh` now runs each job in a systemd user scope (no sudo) capped at
  CPUQuota=500%, 6 GB of RAM and 2 GB of swap, with 6 threads, `nice 15`, idle I/O
  and `oom_score_adj 500`. The copy in `laptop/heavy.sh` is the earlier, simpler version.
- Optional, needs the user: a systemd user unit `claude-remote-control.service`
  so Remote Control restarts by itself, turning Wi-Fi power saving off, and
  cleaning the fans and vents.

## 7. Your voice (RVC)

- Recording script: [`voice/recording-script.md`](voice/recording-script.md).
- Setup: RVC in its own venv `~/band/tools/rvc` (torch cu126, fp32 on Pascal),
  with ContentVec/HuBERT, RMVPE and the v2 pretrained models.
- Folders: `~/band/voice/{raw,clean,dataset,models}`.
- Flow: clean the recordings (dereverb, trim, normalise), train, then convert the
  separated vocal stem of a song and remix it with the instrumental.
- Only train on your own voice, or on someone who has agreed in writing.

## 8. Next songs

- Vary the lengths: mix short songs (~2–2.5 minutes) with longer ones (~5 minutes,
  in one pass or with a repaint extension).
- Add a little bagpipe where it fits: a distant drone, a short melody under a
  chorus, kept low in the mix.
