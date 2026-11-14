# Laptop scripts

Copies of the scripts in `~/band` on the band laptop (GTX 1060 6 GB, 8 GB RAM,
headless Ubuntu Server). What they do and why is explained in
[`../docs/laptop-pipeline.md`](../docs/laptop-pipeline.md). Paths assume the
layout below. Secrets and IP addresses have been removed.

```
~/band/
  ACE-Step-1.5/          upstream checkout + its .venv (torch 2.10.0+cu126), never modified
  gen.py                 = upstream-style generate script + patches/gen.py.patch
  gen.sh run.sh heavy.sh monitor.sh outro-job.sh sweep.sh rvc.sh repaint.py
  tools/fx/.venv         CPU: analysis + mastering (tools/*.py run here)
  tools/sep/.venv        GPU: audio-separator (stems, dereverb)
  tools/rvc/             Applio (RVC) + its .venv
  models/separator/      separator checkpoints
  songs/<slug>/{00-gen,10-analysis,20-stems,30-vocals,40-edits,60-master}
  voice/{raw,clean,dataset,models}
```

## Files

| File | What it does |
|---|---|
| `gen.sh` | `gen.sh caption.txt lyrics.txt seconds seed name [bpm]`: one song into `~/band/outputs/name.wav` |
| `patches/gen.py.patch` | The Pascal fixes: fp32 compute, direct-to-GPU load, no DiT offload, SDPA attention (`BAND_*` env vars). Note: `gen.py.orig` was saved after a first edit, so it already reads `BAND_OFFLOAD_DIT`; the very first version had a plain `offload_dit_to_cpu=True`. |
| `repaint.py` | Repaint `[start, end]` of an existing song. Text encoder on CPU, 5 s VAE encode chunks |
| `run.sh` | Run `gen.sh` low-priority with a log and start/end stamps |
| `heavy.sh` | Wrap any heavy job: nice 15, idle I/O, 8 threads, `oom_score_adj 500` |
| `outro-job.sh` | Repaint a window with several seeds (one process each), then separate the window to check for vocals |
| `sweep.sh` | Length sweep (one-shot generations at several lengths) |
| `monitor.sh` | GPU memory/temperature and RAM every 15 s |
| `rvc.sh` | Run Applio's command line in fp32, low priority, under the GPU lock |
| `tools/analyze.py` | Beats, downbeats, BPM, key, LUFS, true peak, loudness range, dropouts, where audio ends |
| `tools/check_audio.py` | Quick check: loudness per 15 s, tempo per minute, whether the ending fades or is cut off |
| `tools/stemcheck.py` | Null test of the stems and vocal presence per 5 s |
| `tools/vocal_profile.py` | Vocal level per second, longest vocal run |
| `tools/window_beats.py` | Odd beats/bars around a time window (to place repaints) |
| `tools/repaint_check.py` | Null test outside a repaint window, change inside, seam levels |
| `tools/master.py` | Pedalboard master to -14 LUFS / -1 dBTP, plus Matchering against a reference |
| `tools/stem_master.py` | Process vocal and instrumental stems separately, then remix (master afterwards with `master.py`) |
| `tools/produce.py` | Full stem-based production pass: Demucs stems, measured per-stem processing, vocal doubles and octave, sidechain, gentler master, report. Config example in `songs/deed-to-the-glen/` |
| `systemd/claude-remote-control.service` | Optional user unit so Remote Control restarts by itself (written, not enabled) |
| `systemd/wifi-powersave-off.sh` | Optional, needs sudo, not applied: turn Wi-Fi power saving off |
| `NOTES.md` | Standing preferences for new songs |
| `songs/deed-to-the-glen/production-v1.config*.json` | `produce.py` config for Glen, and the sections it detected |
| `songs/swim-at-your-own-risk/caption.txt` | Example caption. The lyrics are in [`docs/lyrics.md`](../docs/lyrics.md) |

Every heavy job, GPU or CPU, runs one at a time: `flock ~/band/.gpu.lock ~/band/heavy.sh ...`. On the laptop, `heavy.sh` now also wraps the job in a systemd user scope (CPUQuota=500%, MemoryMax=6G, 2 GB of swap, 6 threads); the copy here is the earlier version.

## Installing the tool venvs (uv, Python 3.12)

```
# tools/fx (CPU: analysis and mastering)
cd ~/band/tools/fx && uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python pedalboard pyloudnorm matchering librosa soundfile beat-this
# resolved: pedalboard 0.9.26, pyloudnorm 0.2.0, matchering 2.0.6, librosa 1.0.1, soundfile 0.14.0,
#   beat-this 1.1.0, numpy 2.5.4, scipy 1.18.1. beat_this downloads its 78 MB checkpoint on first use.

# tools/sep (GPU: stem separation and dereverb)
cd ~/band/tools/sep && uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv/bin/python "audio-separator[cpu]"
uv pip install --python .venv/bin/python audioread   # missing dependency of audio-separator 0.48.0
uv pip install --python .venv/bin/python torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126 --reinstall-package torchvision   # resolver pulls cu128, which has no Pascal kernels
# models (downloaded on first use into ~/band/models/separator):
#   model_bs_roformer_ep_317_sdr_12.9755.ckpt, dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt
# usage: audio-separator IN.wav -m <model> --model_file_dir ~/band/models/separator --output_dir OUT --output_format WAV

# tools/rvc (GPU: Applio command line, MIT)
cd ~/band/tools/rvc && git clone --depth 1 https://github.com/IAHispano/Applio.git Applio
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu126
printf 'torch==2.10.0\ntorchaudio==2.10.0\n' > overrides.txt   # Applio pins a cu128 torch, which may drop Pascal
(cd Applio && uv pip install --python ../.venv/bin/python -r requirements-engine.txt --override ../overrides.txt \
   --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match)
echo '{"precision": "fp32"}' > applio-config.json   # Pascal has no usable fp16; used via rvc.sh
```

## Voice (RVC) flow, not yet run end to end

1. One time: `rvc.sh prerequisites --pretraineds-hifigan --models --no-exe`.
2. Clean `voice/raw` into `voice/dataset`: 48 kHz mono, 70 Hz high-pass, remove silences, optional dereverb, -20 LUFS, 8 s clips (drop any under 3 s).
3. Train: `rvc.sh preprocess`, `extract` (rmvpe, contentvec), `train` (48 kHz, HiFi-GAN, ~250 epochs, batch 4), `index`.
4. Convert a song: separate the vocal, `rvc.sh infer` with the trained model, remix with the instrumental, then `tools/master.py`.

The recording script is [`../docs/voice/recording-script.md`](../docs/voice/recording-script.md).
