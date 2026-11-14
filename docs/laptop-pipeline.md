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

## 5. Master

`tools/master.py` (pedalboard): high-pass, gentle glue compression, EQ,
limiter, to -14 LUFS integrated and -1 dBTP. Reuben preferred this over
Matchering on Glen, so the Glen pedalboard master is the reference for the EP's sound.

Planned and in progress:

- **Stem-based master:** split vocal and instrumental, then
  - vocal: gate, 90 Hz high-pass, de-ess, 3:1 compression, presence EQ, short reverb;
  - instrumental: high-pass and glue;
  - remix both stems and master to -14 LUFS / -1 dBTP → `60-master/stem-master.wav`.
- **One sound for the EP:** Matchering each new song against the Glen pedalboard
  master → `matchering-glenref.wav`.
- **Automatic quality checks:** gap, tempo and outro checks after every generation,
  with an automatic repaint when a check fails.

Always compare versions at matched loudness, because louder always sounds better.

## 6. Keeping the laptop connected

Heavy jobs used to drop the connection. The cause was CPU saturation and
thermal throttling (about 45,000 throttle events) starving `claude remote-control`,
not Wi-Fi or memory.

- Every heavy job runs through `heavy.sh`: `nice 15`, idle I/O priority, 8 of 12
  threads, `oom_score_adj 500`.
- GPU jobs take a lock so only one runs at a time: `flock ~/band/.gpu.lock`.
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
