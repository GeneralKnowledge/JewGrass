# Open-source music production ecosystem: research and compatibility audit

**Date:** 2026-11-13. **Scope:** research only. Nothing was installed, upgraded or
changed on the laptop while writing this.

**The question:** what is the best production ecosystem we can build around the
ACE-Step 1.5 install that already makes complete 4-minute songs on the GTX 1060
laptop, without breaking it, and while keeping the cloud option open?

**Short answer:**

1. **Implement first:** (a) section repair with ACE-Step's own **repaint** mode, which
   our turbo model supports and needs no new software; (b) **stem separation and vocal
   clean-up** with `python-audio-separator` in its own virtual environment; (c) a
   **CPU-only analysis and mastering chain** (beat_this, librosa, pyloudnorm, Pedalboard,
   Matchering). These three are independent, cheap to roll back, and each improves
   finished songs directly.
2. **Next:** consistent band voices with **RVC through Applio**, trained only on vocals
   we generated ourselves. Compare against zero-shot conversion (Seed-VC, SoulX-Singer-SVC)
   before committing hours to training.
3. **Leave alone:** all-in-one studios (Remiqora, StemForge, ComfyUI), anything
   non-commercial (Side-Step, YuE2 weights, MusicGen weights, RAVE, madmom models,
   MuScriptor, SheetSage2, Vevo2, Diff-MST), and anything pinned to CUDA 12.8 or 13
   unless we can override the pin.
4. **Revisit with more compute:** ACE-Step LoRA training for a "band sound", the XL
   model, the base model's lego/complete tasks at full quality, HeartMuLa, Stable Audio 3
   Medium.

How to read this report: **[V]** marks a fact checked against the repository, its
docs, its issues or the laptop itself on 2026-11-13. **[A]** marks an assumption or
inference that still needs a test. Repository metadata (licence, last push, stars,
archived flag) comes from the GitHub API on 2026-11-13.

---

## Contents

1. [The existing system](#1-the-existing-system)
2. [Hardware compatibility rules for this laptop](#2-hardware-compatibility-rules-for-this-laptop)
3. [Projects investigated](#3-projects-investigated)
4. [The current shortlist, re-verified](#4-the-current-shortlist-re-verified)
5. [Detailed compatibility table](#5-detailed-compatibility-table)
6. [Scores and tiers](#6-scores-and-tiers)
7. [Best projects per production category](#7-best-projects-per-production-category)
8. [Rejected projects and why](#8-rejected-projects-and-why)
9. [Recommended architecture](#9-recommended-architecture)
10. [Prioritised roadmap](#10-prioritised-roadmap)
11. [The first three experiments](#11-the-first-three-experiments)
12. [Outstanding uncertainties](#12-outstanding-uncertainties)
13. [Final recommendation](#13-final-recommendation)

---

## 1. The existing system

### 1.1 Repository

The [JewGrass repository](https://github.com/GeneralKnowledge/JewGrass) holds
planning documents only, no code **[V]**:

| File | Content |
|---|---|
| `README.md` | Index |
| `docs/stage-2-local-option.md` | Local plan: ACE-Step 1.5 install, Pascal notes, batches, band members, RVC idea |
| `docs/stage-1-suno-plan.md` | Earlier Suno plan |
| `docs/experiment-log.md` | Scoring sheet (still the Suno table) |
| `docs/lyrics.md`, `docs/drafts/` | Lyrics |

The scripts that actually run (`gen.sh`, `gen.py`) live only on the laptop in `~/band`
and are not in version control **[V]**. That is the first gap the roadmap closes
(Stage 0).

### 1.2 The machine

Collected read-only from the laptop on 2026-11-13 (full dump in `~/band/env-report.txt`
on the laptop) **[V]**:

| Property | Value |
|---|---|
| Laptop | Dell G3 17 |
| CPU | Intel i7-8750H, 6 cores / 12 threads |
| GPU | GTX 1060 Max-Q, 6144 MiB, compute capability **6.1 (Pascal)** |
| Driver | 550.163.01 (supports CUDA runtime up to **12.4**) |
| RAM / swap | 7.6 GB / 19 GB |
| Disk free | ~497 GB |
| OS | Ubuntu 24.04.3 Server, kernel 6.8, headless, lid closed |
| System CUDA toolkit | `nvcc` 12.0 at `/usr/bin/nvcc` |
| Tools present | `ffmpeg` 6.1.1, `uv`, `git`, miniconda3 (envs incl. `AveDeus`, `mfa`), `~/.venv`, `~/.venvs`. `sox` is **not** installed. |

**Correction to the brief:** the brief says "CUDA 12.4 / PyTorch 2.6.0". The working
ACE-Step environment actually runs **PyTorch 2.10.0+cu126** (cuDNN 9.10.2) on the
550 driver **[V]**. CUDA 12.4 is the driver's ceiling, and the cu126 wheel works on it.
PyTorch 2.6.0 may exist in one of the other environments; it is not what ACE-Step uses.

### 1.3 The ACE-Step installation

| Property | Value |
|---|---|
| Location | `~/band/ACE-Step-1.5`, commit `ca1e85f`, origin `github.com/ace-step/ACE-Step-1.5` **[V]** |
| Environment | its own `.venv`, run with `uv run --no-sync` (so `uv` never reinstalls the cu128 torch) **[V]** |
| Key packages | Python 3.12.3, torch/torchaudio 2.10.0+cu126, transformers 4.57.6, diffusers 0.37.1, accelerate 1.12.0, torchao 0.16.0, peft 0.18.1, numpy 2.5.4, gradio 6.2.0; no xformers **[V]** |
| Model | 2B **turbo** DiT, 5 Hz language model **off** **[V]** |
| Entry point | `~/band/gen.sh <caption> <lyrics> <seconds> <seed> <name> <bpm>` → `~/band/gen.py` **[V]** |
| Local Pascal fixes in `gen.py` | DiT weights fp16 on GPU but **computed per layer in fp32** (`BAND_FP32_COMPUTE=1`; pure fp16 gives all-NaN latents); DiT loaded tensor by tensor straight to CUDA (`BAND_DIT_DIRECT=1`); never offloaded (`BAND_OFFLOAD_DIT=0`); SDPA attention (`BAND_ATTN=sdpa`; eager attention was quadratic in memory) **[V]** |
| Performance | 240 s song in one pass, ~5–6 min end to end (load ~2.5 min, text encoder ~80 s, diffusion ~45 s, tiled VAE decode ~60 s); VRAM flat at ~5.9 GB (card full); RAM < 4.7 GB; GPU ≤ 66 °C **[V]** |
| Outputs | `~/band/outputs/*.wav`, flat folder, e.g. `your-landlord-len240.wav`, `cold-hame-full-2222.wav` **[V]** |

**Architecture today:** one Python process per song. It loads the model, generates,
decodes, writes one WAV and exits. No analysis, no stems, no mixing or mastering, and
no metadata file next to the audio beyond the filename **[V]**. Captions and lyrics
are text files in `~/band` **[V]**.

**Upstream status that matters to us:** the official fixes for Pascal NaNs are still
open pull requests upstream ([#1377](https://github.com/ace-step/ACE-Step-1.5/pull/1377),
[#1352](https://github.com/ace-step/ACE-Step-1.5/pull/1352),
[#1312](https://github.com/ace-step/ACE-Step-1.5/pull/1312),
[#1353](https://github.com/ace-step/ACE-Step-1.5/pull/1353); related closed issues
[#1274](https://github.com/ace-step/ACE-Step-1.5/issues/1274) and
[#1055](https://github.com/ace-step/ACE-Step-1.5/issues/1055)) **[V]**. Our local patch
is therefore still needed. **Do not `git pull` ACE-Step** without re-testing; pin
commit `ca1e85f`.

### 1.4 What ACE-Step already offers that we are not using

This is the most important finding: much of the "improve, edit, regenerate" wish list
is already inside ACE-Step 1.5
([README](https://github.com/ace-step/ACE-Step-1.5#readme),
[Tutorial](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/Tutorial.md),
[Inference API](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md)) **[V]**:

| Task (`task_type`) | turbo (ours) | base | What it does |
|---|:-:|:-:|---|
| `repaint` | ✅ | ✅ | Regenerate only `repainting_start`–`repainting_end` seconds, using the rest of the song as context. Also extends songs (segments of 3–90 s can be chained). |
| `cover` | ✅ | ✅ | Keep a song's melody/structure (`src_audio`, `audio_cover_strength` 0–1) and change caption or lyrics |
| `reference_audio` | ✅ | ✅ | Global timbre/mix/style guidance from a reference file |
| `extract` | ❌ | ✅ | Pull one track (e.g. vocals) out of a mix |
| `lego` | ❌ | ✅ | Add a new track (drums, guitar...) on top of existing audio in a time window |
| `complete` | ❌ | ✅ | Generate a full backing for a single track (e.g. a cappella vocal) |
| Audio understanding | via LM | via LM | BPM, key, time signature, caption from audio. Needs the 5 Hz LM, which we have switched off |

The **base** model is the same 2B size as turbo but uses 50 steps and CFG **[V]**.
Whether it fits in 6 GB with our fp32-compute patch is unknown **[A]** (CFG roughly
doubles the activations per step). It is a Tier 2 experiment.

LoRA training needs **16 GB minimum, 20 GB recommended** per the official
[LoRA tutorial](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/LoRA_Training_Tutorial.md)
**[V]** (the README's "12 GB" figure is older and contradicts the tutorial). That is
cloud-only for us.

---

## 2. Hardware compatibility rules for this laptop

These rules decide most verdicts below. They are why "supports CUDA" is not the same
as "runs on our GTX 1060".

| # | Rule | Evidence |
|---|---|---|
| R1 | **PyTorch must come from the `cu126` (or `cu118`) wheel index.** The cu128 build of torch 2.10 failed on this card with "no kernel image"; cu126 works. | **[V]** laptop smoke test, `docs/stage-2-local-option.md` |
| R2 | **CUDA 13 (`cu130`) builds are impossible** here: CUDA 13 drops Pascal and needs a newer driver than 550. | **[V]** driver 550 = CUDA 12.4 ceiling; **[A]** CUDA 13 Pascal removal per NVIDIA release notes (not re-checked today) |
| R3 | **No bf16, and fp16 *compute* is unsafe** (NaNs in ACE-Step; Pascal GP106 fp16 throughput is also very low). Run every model in **fp32**, or fp16 storage with fp32 compute. Turn off `--fp16`, `autocast`, `half` flags. | **[V]** laptop; ACE-Step issues #1274/#1055 |
| R4 | **No Flash-Attention 2, no vLLM/Triton fast paths, recent xformers and NATTEN builds are doubtful.** FA2 needs Ampere (sm80+); ACE-Step itself forces the PyTorch LM backend below sm70. | **[V]** [ACE-Step GPU_COMPATIBILITY](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/GPU_COMPATIBILITY.md); **[V]** Stable Audio 3 Medium README requires FA2 |
| R5 | **GGML/C++ ports need an explicit architecture flag.** acestep.cpp's default CUDA list starts at 7.5 (Turing), so a default build has no sm_61 code. Build with `-DCMAKE_CUDA_ARCHITECTURES=61`. | **[V]** [acestep.cpp CMakeLists.txt](https://github.com/ServeurpersoCom/acestep.cpp/blob/master/CMakeLists.txt) lines 42–49 |
| R6 | **The GPU is one exclusive resource.** ACE-Step uses ~5.9 of 6 GB, so nothing else can share the card while it runs. Every GPU tool must run as its own process, one at a time. | **[V]** laptop |
| R7 | **7.6 GB RAM.** Avoid tools that load several models at once or keep everything resident (all-in-one studios, web UIs that preload). Loading from disk is slow (ACE-Step load is ~2.5 min), so batch work per model. | **[V]** laptop |
| R8 | **Silent CPU fallback is the main trap.** A plain `pip install torch` on Linux gets the default (cu128) wheel or, with some resolvers, a CPU wheel. Always check `torch.cuda.is_available()` **and** `torch.cuda.get_arch_list()` contains `sm_61` after installing. | **[V]** Remiqora README notes `uv add demucs` "would silently resolve a CPU-only torch wheel" |
| R9 | **Never install into ACE-Step's `.venv`.** Every new tool gets its own environment. | Brief constraint |

**Licensing rule used throughout:** a GPL or AGPL *tool* used to process our audio does
not make the audio GPL; it only matters if we redistribute the software. **Model
weights** under non-commercial licences (CC BY-NC, research-only) are different: using
them to make release audio is the risk, so they are excluded for release work.

---

## 3. Projects investigated

About 95 repositories were checked. Licence = GitHub's detected licence (checked
by hand where it says NOASSERTION). "Last push" = last commit on any branch.

### A. Generation and editing

| Project | Licence | Last push | Stars | Notes |
|---|---|---|---|---|
| [ace-step/ACE-Step-1.5](https://github.com/ace-step/ACE-Step-1.5) | MIT (code + weights) | 2026-11-08 | 14.7k | Our engine. Repaint/cover/extract/lego/complete built in. |
| [ServeurpersoCom/acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp) | MIT | 2026-11-13 | 462 | C++/GGML port, Q4–Q8 GGUF, all six task types, `--clamp-fp16`, VAE tiling, HTTP server and CLI pipe. |
| [0xShug0/audio.cpp](https://github.com/0xShug0/audio.cpp) | Apache-2.0 | 2026-11-13 | 4.2k | One GGML runtime for 140+ audio models incl. ACE-Step, RVC, Seed-VC, BS/Mel-RoFormer, HTDemucs, Apollo. Per-model weight licences listed in [docs/model_licenses.md](https://github.com/0xShug0/audio.cpp/blob/main/docs/model_licenses.md). |
| [ace-step/ACE-Step-ComfyUI](https://github.com/ace-step/ACE-Step-ComfyUI) | MIT | 2026-03-01 | 87 | Official ComfyUI nodes |
| [ace-step/awesome-ace-step](https://github.com/ace-step/awesome-ace-step) | CC0 | 2026-05-28 | 166 | Community index |
| [inikolax/remiqora](https://github.com/inikolax/remiqora) | MIT | 2026-11-12 | 222 | All-in-one studio (see §4) |
| [koda-dernet/Side-Step](https://github.com/koda-dernet/Side-Step) | **CC BY-NC-SA 4.0** | 2026-09-01 | 164 | LoRA trainer (see §4) |
| [fspecii/ace-step-ui](https://github.com/fspecii/ace-step-ui) | **none** | 2026-06-27 | 5.2k | UI, no licence |
| [roblaughter/ace-step-studio](https://github.com/roblaughter/ace-step-studio) | **none** | 2026-02-07 | 160 | UI, no licence |
| [multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE) | Apache-2.0 code; **YuE2 weights CC BY-NC 4.0** | 2026-11-01 | 13.5k | Needs ~24 GB |
| [HeartMuLa/heartlib](https://github.com/HeartMuLa/heartlib) | Apache-2.0 (code + weights) | 2026-10-08 | 3.8k | 3B, bf16 default |
| [Stability-AI/stable-audio-3](https://github.com/Stability-AI/stable-audio-3) | MIT code; weights **Stability AI Community License** | 2026-11-02 | 877 | Small (433M) runs on CPU; Medium needs FA2 |
| [Stability-AI/stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools) | MIT | 2026-11-01 | 3.9k | Older Stable Audio Open tooling |
| [facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft) | MIT code; **MusicGen weights CC BY-NC** | 2026-03-03 | 23.9k | No vocals |
| [ASLP-lab/DiffRhythm](https://github.com/ASLP-lab/DiffRhythm) | Apache-2.0 | 2025-11-27 | 2.4k | Alternative full-song model |
| [magenta/magenta](https://github.com/magenta/magenta) | Apache-2.0 | 2026-01-06 | 19.8k | **Archived** |

### B. Vocals: conversion, synthesis, enhancement, pitch, timing

| Project | Licence | Last push | Stars | Notes |
|---|---|---|---|---|
| [IAHispano/Applio](https://github.com/IAHispano/Applio) | MIT (+ Terms of Use) | 2026-11-04 | 4.0k | Maintained RVC distribution, CLI `core.py` |
| [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) | MIT | 2026-08-04 | 39.2k | Original RVC |
| [Plachtaa/seed-vc](https://github.com/Plachtaa/seed-vc) | **GPL-3.0** | 2025-04-20 | 3.9k | **Archived** |
| [Soul-AILab/SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer) | Apache-2.0 | 2026-05-29 | 1.1k | Zero-shot singing synthesis + SVC (English supported) |
| [yxlllc/DDSP-SVC](https://github.com/yxlllc/DDSP-SVC) | MIT | 2026-11-03 | 2.7k | Light SVC, needs training |
| [openvpi/DiffSinger](https://github.com/openvpi/DiffSinger) | Apache-2.0 (LICENSE file) | active | – | Score-based singing synthesis; needs a voicebank |
| [open-mmlab/Amphion](https://github.com/open-mmlab/Amphion) | MIT code | 2026-03-25 | 10.3k | Vevo2 weights are **CC BY-NC-ND** |
| [svc-develop-team/so-vits-svc](https://github.com/svc-develop-team/so-vits-svc) | AGPL-3.0 | 2023-11-11 | 28.0k | **Archived** |
| [prophesier/diff-svc](https://github.com/prophesier/diff-svc) | AGPL-3.0 | 2026-06-06 | 2.7k | Superseded |
| [w-okada/voice-changer](https://github.com/w-okada/voice-changer) | MIT | 2026-09-26 | 21.4k | Real-time, desktop-oriented |
| [ASLP-lab/MeanVC](https://github.com/ASLP-lab/MeanVC) | Apache-2.0 | 2026-07-31 | 310 | Speech VC, not singing |
| [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) | MIT | 2026-11-11 | 64.9k | Speech TTS; for spoken skits only |
| [maxrmorrison/torchcrepe](https://github.com/maxrmorrison/torchcrepe) | MIT | 2025-05-16 | 525 | Pitch tracking |
| [yxlllc/RMVPE](https://github.com/yxlllc/RMVPE) / [Dream-High/RMVPE](https://github.com/Dream-High/RMVPE) | – / Apache-2.0 | 2024 | 336 | Vocal pitch tracking used by RVC |
| [JeremyCCHsu/Python-Wrapper-for-World-Vocoder](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder) (pyworld) | MIT | 2026-08-20 | 798 | Analysis/resynthesis for pitch correction |
| [breakfastquay/rubberband](https://github.com/breakfastquay/rubberband) + [bmcfee/pyrubberband](https://github.com/bmcfee/pyrubberband) | GPL-2.0 (+ commercial) / ISC | 2025 / 2024 | 812 / 218 | Formant-preserving pitch shift, time stretch |
| [Signalsmith-Audio/signalsmith-stretch](https://github.com/Signalsmith-Audio/signalsmith-stretch) | MIT | 2026-10-22 | 580 | Permissive pitch/time library (C++) |
| [YannickJadoul/Parselmouth](https://github.com/YannickJadoul/Parselmouth) | GPL-3.0 | 2026-10-13 | 1.3k | Praat PSOLA (StemForge uses it for auto-tune) |
| [maxrmorrison/psola](https://github.com/maxrmorrison/psola) | GPL-3.0 | 2023-08-16 | 93 | Small, stale |
| [MontrealCorpusTools/Montreal-Forced-Aligner](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) | MIT | 2026-11-12 | 1.9k | Word/phone timing. **A conda env named `mfa` already exists on the laptop** **[V]** |
| [m-bain/whisperX](https://github.com/m-bain/whisperX) | BSD-2 | 2026-11-12 | 25.2k | Lyrics transcription with word timestamps |
| [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) | MIT/Apache-2.0 | 2026-11-06 | 5.0k | Speech denoise |
| [resemble-ai/resemble-enhance](https://github.com/resemble-ai/resemble-enhance) | MIT | 2024-12-03 | 2.5k | Speech enhance; needs deepspeed |
| [haoheliu/voicefixer](https://github.com/haoheliu/voicefixer) | MIT | 2025-02-17 | 1.4k | Speech restoration |
| [NVIDIA/BigVGAN](https://github.com/NVIDIA/BigVGAN) | MIT | 2024-09-05 | 1.2k | Vocoder used inside SVC models |

### C. Separation and restoration

| Project | Licence | Last push | Stars | Notes |
|---|---|---|---|---|
| [nomadkaraoke/python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) | MIT | 2026-10-29 | 1.5k | CLI + library; RoFormer, MDX, VR, Demucs models; de-reverb, denoise, karaoke, drum-sep |
| [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) | MIT | 2026-11-12 | 1.6k | Training + inference for RoFormer/MDX23C/HTDemucs; torch ≥ 2.6 |
| [lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer) | MIT | 2026-11-04 | 1.0k | Architecture only, no trained weights |
| [adefossez/demucs](https://github.com/adefossez/demucs) | MIT (LICENSE file) | active | – | **Now the officially maintained Demucs** (README) **[V]** |
| [facebookresearch/demucs](https://github.com/facebookresearch/demucs) | MIT | 2024-04-24 | 10.4k | **Archived** |
| [Anjok07/ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui) | MIT | 2026-10-06 | 27.1k | Desktop GUI (needs a display) |
| [tsondo/StemForge](https://github.com/tsondo/StemForge) | Apache-2.0 | 2026-09-20 | 42 | Web workstation (see §4) |
| [deezer/spleeter](https://github.com/deezer/spleeter) | MIT | 2026-10-28 | 28.5k | Superseded quality |
| [sigsep/open-unmix-pytorch](https://github.com/sigsep/open-unmix-pytorch) | MIT | 2024-06-17 | 1.5k | Superseded quality |
| [haoheliu/versatile_audio_super_resolution](https://github.com/haoheliu/versatile_audio_super_resolution) (AudioSR) | MIT | 2025-08-27 | 2.0k | Bandwidth extension, heavy |
| [JusperLee/Apollo](https://github.com/JusperLee/Apollo) | **CC BY-SA 4.0** | 2026-08-24 | 437 | Lossy-to-lossless music restoration |

### D. Analysis

| Project | Licence | Last push | Stars | Notes |
|---|---|---|---|---|
| [CPJKU/beat_this](https://github.com/CPJKU/beat_this) | MIT | 2026-05-28 | 483 | Beats + downbeats, CLI, GPU or CPU |
| [librosa/librosa](https://github.com/librosa/librosa) | ISC | 2026-11-12 | 8.9k | Onsets, tempo, chroma, key estimation building blocks |
| [CPJKU/madmom](https://github.com/CPJKU/madmom) | BSD code; **models CC BY-NC-SA** | 2026-03-20 | 1.7k | Classic beat/key/chord models |
| [MTG/essentia](https://github.com/MTG/essentia) | AGPL-3.0 | 2026-11-05 | 3.8k | Key, tempo, many descriptors |
| [mir-aidj/all-in-one](https://github.com/mir-aidj/all-in-one) | MIT | 2024-05-09 | 888 | Beats, downbeats, section labels in one model; needs NATTEN + Demucs |
| [ASLP-lab/SongFormer](https://github.com/ASLP-lab/SongFormer) | CC BY 4.0 | 2026-05-14 | 190 | Section boundaries + labels (verse/chorus...) |
| [urinieto/msaf](https://github.com/urinieto/msaf) | MIT | 2026-11-13 | 561 | Classic structure-analysis algorithms (CPU) |
| [spotify/basic-pitch](https://github.com/spotify/basic-pitch) | Apache-2.0 | 2025-11-13 | 6.0k | Audio-to-MIDI, polyphonic |
| [magenta/mt3](https://github.com/magenta/mt3) | Apache-2.0 | 2026-10-27 | 1.8k | Multi-instrument transcription (JAX) |
| [mimbres/YourMT3](https://github.com/mimbres/YourMT3) | GPL-3.0 | 2024-11-29 | 255 | Multi-instrument transcription |
| [Music-and-Culture-Technology-Lab/omnizart](https://github.com/Music-and-Culture-Technology-Lab/omnizart) | MIT | 2026-05-31 | 2.0k | Older TF-based transcription |
| [jayg996/BTC-ISMIR19](https://github.com/jayg996/BTC-ISMIR19) | MIT | 2020-05-23 | 219 | Chord recognition, unmaintained |
| [ptnghia-j/ChordMiniApp](https://github.com/ptnghia-j/ChordMiniApp) | MIT | 2026-07-30 | 434 | Web app around chord/beat models |
| [LAION-AI/CLAP](https://github.com/LAION-AI/CLAP) | CC0 | 2025-05-15 | 2.3k | Audio-text embeddings for similarity |
| [marl/openl3](https://github.com/marl/openl3) | MIT | 2023-06-17 | 605 | Audio embeddings (TF) |

### E. Production, mixing, mastering

| Project | Licence | Last push | Stars | Notes |
|---|---|---|---|---|
| [spotify/pedalboard](https://github.com/spotify/pedalboard) | GPL-3.0 | 2026-11-09 | 6.5k | Effects, VST3 hosting, file I/O; Python 3.10–3.15 |
| [sergree/matchering](https://github.com/sergree/matchering) | GPL-3.0 | 2026-10-07 | 2.7k | Reference mastering (RMS, frequency response, peak, stereo width) |
| [csteinmetz1/pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) | MIT | 2026-01-04 | 792 | ITU-R BS.1770-4 loudness |
| [DBraun/DawDreamer](https://github.com/DBraun/DawDreamer) | GPL-3.0 | 2026-11-01 | 1.3k | Python DAW: VST hosting, automation, Faust |
| [csteinmetz1/dasp-pytorch](https://github.com/csteinmetz1/dasp-pytorch) | Apache-2.0 | 2023-12-04 | 303 | Differentiable EQ/compressor (research) |
| [csteinmetz1/automix-toolkit](https://github.com/csteinmetz1/automix-toolkit) | Apache-2.0 | 2024-08-28 | 119 | Automatic-mixing research models |
| [sai-soum/Diff-MST](https://github.com/sai-soum/Diff-MST) | **CC BY-NC-SA 4.0** | 2025-07-07 | 66 | Mix style transfer |
| [Ardour/ardour](https://github.com/Ardour/ardour) | GPL-2.0 | 2026-11-13 | 5.4k | Full DAW (needs a display; use on another computer) |
| [lsp-plugins/lsp-plugins](https://github.com/lsp-plugins/lsp-plugins) | LGPL-3.0 | 2026-11-05 | 909 | Linux EQ/comp/limiter plugins (VST3/LV2) |
| [airwindows/airwindows](https://github.com/airwindows/airwindows) | MIT | 2026-11-08 | 1.3k | Hundreds of small effects incl. tape/saturation |

### F. Creative and symbolic

| Project | Licence | Last push | Stars | Notes |
|---|---|---|---|---|
| [cuthbertLab/music21](https://github.com/cuthbertLab/music21) | BSD-3 | 2026-11-11 | 2.6k | Music theory: keys, scales, harmonisation, voice leading |
| [craffel/pretty-midi](https://github.com/craffel/pretty-midi) | MIT | 2026-11-08 | 1.1k | MIDI manipulation |
| [mido/mido](https://github.com/mido/mido) | MIT | 2026-11-09 | 1.7k | Low-level MIDI |
| [FluidSynth/fluidsynth](https://github.com/FluidSynth/fluidsynth) | LGPL-2.1 | 2026-11-11 | 2.5k | SoundFont rendering, headless CLI |
| [surge-synthesizer/surge](https://github.com/surge-synthesizer/surge) | GPL-3.0 | 2026-11-13 | 4.1k | Synth; Linux VST3, can be hosted by Pedalboard **[A]** |
| [SkyTNT/midi-model](https://github.com/SkyTNT/midi-model) | Apache-2.0 | 2024-12-31 | 386 | Small MIDI generation model |
| [jthickstun/anticipation](https://github.com/jthickstun/anticipation) | Apache-2.0 | 2026-07-19 | 205 | MIDI infilling ("anticipatory" model) |
| [EleutherAI/aria](https://github.com/EleutherAI/aria) | Apache-2.0 | 2026-05-12 | 123 | Symbolic (piano) model |
| [AMAAI-Lab/Text2midi](https://github.com/AMAAI-Lab/Text2midi) | MIT | 2025-02-28 | 185 | Text to MIDI |
| [magenta/ddsp](https://github.com/magenta/ddsp) | Apache-2.0 | 2026-10-19 | 3.4k | Timbre transfer (TF) |
| [acids-ircam/RAVE](https://github.com/acids-ircam/RAVE) | **CC BY-NC 4.0** | 2026-11-09 | 1.8k | Neural timbre transfer |

---

## 4. The current shortlist, re-verified

| # | Repository | Exists? | Maintained? | Does what it says? | Licence | Runs on our laptop? | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | [inikolax/remiqora](https://github.com/inikolax/remiqora) | ✅ | ✅ pushed 2026-11-12 | ✅ ACE-Step 1.5 + YuE2 + Stable Audio 3 + Demucs + MuScriptor + DAW + LoRA, one web UI | MIT (YuE2/MuScriptor/SheetSage2 **weights are non-commercial**) | ❌ in practice. Windows/macOS installers; tested on an RTX 4080 16 GB; Demucs routed to the **cu128** index (no Pascal, R1); installs its *own* ACE-Step; ~18–80 GB download; orchestrator assumes one big GPU. **[V]** README | **Reject** as a tool. Borrow ideas: shared SQLite track store, mutual-exclusion of models on one GPU, "variants on a lane next to the source". |
| 2 | [tsondo/stemforge](https://github.com/tsondo/StemForge) | ✅ (`tsondo/StemForge`) | ✅ 2026-09-20, 42 stars, single author | ✅ Demucs, BS-RoFormer, BasicPitch, UVR clean-up, CREPE+Praat auto-tune, RVC (vendored Applio), ACE-Step, Stable Audio Open, mixer | Apache-2.0 | ❌ Pinned to **torch 2.13 / cu130** **[V]** pyproject → impossible on Pascal and driver 550 (R2). Recommends 16 GB VRAM. | **Reject**. Its *recipes* are useful: the auto-tune design (CREPE + PSOLA with key snapping) and the UVR de-reverb presets. |
| 3 | [nomadkaraoke/python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) | ✅ | ✅ v0.48.0, pushed 2026-10-29 | ✅ CLI and library, ~100 models incl. BS/Mel-RoFormer vocals, de-reverb (8+ variants), denoise, karaoke lead/backing, crowd, drum-sep, Demucs | MIT (code). **Model weights mostly community-trained with no stated licence** **[V]** | ✅ likely: torch ≥ 2.3 on Linux **[V]**, so we can pin torch 2.10.0+cu126; explicit fp32 mode (fp16 is opt-in) **[V]**; ONNX models can stay on CPU. Not yet run on Pascal **[A]**. | **Tier 1.** |
| 4 | [IAHispano/Applio](https://github.com/IAHispano/Applio) | ✅ | ✅ pushed 2026-11-04 | ✅ RVC training + inference, CLI (`core.py`) and Gradio UI | MIT + Terms of Use (commercial allowed with rights to the voice) **[V]** | ⚠️ `requirements-engine.txt` pins **torch 2.11.0** and `run-install.sh` uses the **cu128** index **[V]** → must override with a Pascal-capable wheel (R1). Whether 2.11 has a cu126 wheel is unchecked **[A]**; fallback is torch 2.10.0+cu126 (known good here). | **Tier 2** (the main band-voice route). |
| 5 | ComfyUI-ACE-Step/ComfyUI-ACE-Step | ❌ **Does not exist** (repo not found) **[V]** | – | – | – | – | **Reject.** Alternatives: [ace-step/ACE-Step-ComfyUI](https://github.com/ace-step/ACE-Step-ComfyUI) (official, MIT, last push 2026-03-01), [billwuhao/ComfyUI_ACE-Step](https://github.com/billwuhao/ComfyUI_ACE-Step) (ACE-Step v1 only). ComfyUI adds a node-graph server and its own torch install on top of a 7.6 GB headless box: no benefit over our CLI. |
| 6 | [spotify/pedalboard](https://github.com/spotify/pedalboard) | ✅ | ✅ 2026-11-09 | ✅ EQ, compressor, limiter, reverb, delay, distortion, chorus, phaser, pitch shift, VST3 hosting, fast audio I/O | **GPL-3.0** (fine for processing our audio) | ✅ CPU only, manylinux wheels, Python 3.10–3.15 **[V]** | **Tier 1.** |
| 7 | [sergree/matchering](https://github.com/sergree/matchering) | ✅ | ✅ 2026-10-07 | ✅ Matches RMS, frequency response, peak and stereo width to a reference **[V]** | **GPL-3.0** | ✅ CPU only, Python ≥ 3.8 **[V]** | **Tier 1** (for album consistency, using our own best master as the reference). |
| 8 | [spotify/basic-pitch](https://github.com/spotify/basic-pitch) | ✅ | ⚠️ slow: last push 2025-11-13 | ✅ polyphonic audio-to-MIDI with pitch bends | Apache-2.0 | ⚠️ On Python ≥ 3.11 Linux it requires **TensorFlow < 2.15.1** **[V]**, which has no Python 3.12 wheels **[A]**. Use a Python 3.11 venv (uv can fetch it) and run on CPU. | **Tier 2.** |
| 9 | [Plachtaa/seed-vc](https://github.com/Plachtaa/seed-vc) | ✅ | ❌ **Archived** (read-only since 2025-04) **[V]** | ✅ zero-shot voice and singing conversion, optional fine-tuning from very little data | **GPL-3.0** code; weights GPL-3.0 per audio.cpp's table **[V]** | ⚠️ torch 2.4.0 pin (old default wheel has sm_61 **[A]**); `--fp16` defaults to True and **must be set False** (R3) | **Tier 2**, for quick identity prototyping only; no fixes will come. |
| 10 | [koda-dernet/Side-Step](https://github.com/koda-dernet/Side-Step) | ✅ | ✅ 2026-09-01 | ✅ advanced LoRA/LoKR/DoRA training for ACE-Step 1.5 | **CC BY-NC-SA 4.0: non-commercial** **[V]** LICENSE | ❌ "Minimal" profile is 8–10 GB VRAM **[V]** | **Reject** for release work. If we ever train in the cloud, use the official MIT trainer. |

**Correction to the earlier shortlist memory:** it called Side-Step's minimum "~8 GB";
the README's own minimal profile is 8–10 GB and "trains down to ~10 GB" **[V]**. It also
did not know Seed-VC is archived, that StemForge is cu130-only, or that item 5 does not
exist.

---

## 5. Detailed compatibility table

Legend for **Class**: **1** confirmed compatible (on this machine), **2** likely
compatible, needs testing, **3** possible with modifications, **4** not practical on our
hardware, **5** needs external hardware or cloud.

"Conflicts with ACE-Step?" is answered as if installed in a **separate** environment
(R9), which is how we would do it. Every row also assumes "No" to Docker.

| Project | Class | GPU / Pascal | VRAM (realistic) | RAM 7.6 GB | CUDA / PyTorch | CPU mode | Precision options | Deps vs ACE-Step | Headless | Maintenance | Licence (commercial?) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ACE-Step 1.5 turbo text2music | **1** | ✅ with local fp32-compute patch **[V]** | 5.9 GB **[V]** | ✅ < 4.7 GB **[V]** | 2.10.0+cu126 **[V]** | slow | fp16 storage + fp32 compute | itself | ✅ | ✅ | MIT ✅ |
| ACE-Step 1.5 turbo **repaint / cover / reference** | **2** | same model **[V]** | ≥ 5.9 GB plus VAE *encode* of source **[A]** | ✅ **[A]** | same | slow | same | none (same env, new script) | ✅ | ✅ | MIT ✅ |
| ACE-Step 1.5 **base** (extract/lego/complete) | **3** | needs our patch applied to base **[A]** | 2B weights like turbo, CFG doubles activations **[A]** | ✅ **[A]** | same | slow | same | extra 4.8 GB download | ✅ | ✅ | MIT ✅ |
| ACE-Step XL / LM 1.7B+ / LoRA training | **5** | – | ≥ 12 GB (XL), ≥ 16 GB (LoRA) **[V]** | – | – | – | – | – | ✅ | ✅ | MIT ✅ |
| acestep.cpp | **3** | ggml supports Pascal **[A]**, but default build skips sm_61; build with `-DCMAKE_CUDA_ARCHITECTURES=61` (R5) **[V]** | Q8 DiT 2.4 GB, VAE 0.3 GB, text enc 0.75 GB **[V]**; leaves room for base/LM **[A]** | ✅ **[A]** | CUDA toolkit to build; system nvcc 12.0 may be too old for the bundled ggml **[A]** | ✅ | Q4–Q8, `--clamp-fp16` **[V]** | none (C++ binary) | ✅ server + CLI **[V]** | ✅ daily | MIT ✅ |
| audio.cpp | **3** | same ggml caveat; `--cuda-arch 61` **[V]** | per model **[A]** | ✅ **[A]** | build from source | ✅ | F16/Q8/Q4 per model **[V]** | none | ✅ CLI + server **[V]** | ✅ very active | Apache-2.0 ✅; **weights vary** (see §8) |
| python-audio-separator | **2** | torch path ✅ with cu126 wheel **[A]** | segment-size dependent; RoFormer 4-min song expected to fit in fp32 **[A]** | ✅ **[A]** | torch ≥ 2.3 **[V]** → pin 2.10.0+cu126 | ✅ (slow) | fp32 default; autocast/native fp16 opt-in **[V]** | separate venv | ✅ CLI **[V]** | ✅ | MIT ✅; **model weights mostly unlicensed** ⚠️ |
| Demucs (adefossez) | **2** | torch ✅ **[A]** | `--segment` reduces VRAM **[V]** | ✅ | torch ≥ 2.1 **[V]** | ✅ (`-j`) | fp32 | separate venv (or via audio-separator) | ✅ | ✅ (official fork) | MIT ✅ |
| ZFTurbo MSST | **2** | torch ✅ **[A]** | model dependent | ✅ | torch ≥ 2.6 **[V]** | ✅ | `--use_tta`, fp32 | separate venv; `wxpython` in requirements (GUI dep, skip) **[V]** | ✅ `inference.py` | ✅ | MIT ✅ |
| Applio (RVC) | **3** | needs cu126 wheel override (R1) **[V]** | inference ~2–4 GB, training batch 4–8 on 6 GB **[A]** | ✅ inference; training tight **[A]** | pins torch 2.11.0 + cu128 index **[V]** | ✅ inference | fp32 on Pascal **[A]** | separate venv | ✅ `core.py` CLI | ✅ | MIT ✅ (+ Terms of Use) |
| RVC WebUI (original) | **3** | same as Applio | same | same | older pins | ✅ | – | separate | ⚠️ web UI | ⚠️ slower | MIT ✅ |
| Seed-VC | **3** | torch 2.4.0 default wheel **[A]**; `--fp16 False` required (R3) | ~3–6 GB for SVC model **[A]** | ✅ **[A]** | torch 2.4.0 **[V]** | ✅ slow | fp16 flag | separate | ✅ `inference.py` **[V]** | ❌ archived | GPL-3.0 (tool OK) |
| SoulX-Singer-SVC | **2** | torch 2.2.0 **[V]** | unknown **[A]** | unknown | torch 2.2.0 **[V]**, Python 3.10 conda **[V]** | unknown | unknown | separate | ✅ scripts + web UI | ⚠️ 2026-05 | Apache-2.0 code ✅; weight licence unverified ⚠️ |
| DDSP-SVC | **2** | torch ✅ **[A]** | small **[A]** | ✅ | unpinned | ✅ | – | separate | ✅ | ✅ | MIT ✅ |
| Pedalboard | **1–2** | CPU | 0 | ✅ | none | ✅ | – | separate | ✅ | ✅ | GPL-3.0 (tool OK) |
| Matchering | **2** | CPU | 0 | ✅ for 4-min songs **[A]** | none | ✅ | – | separate | ✅ | ✅ | GPL-3.0 (tool OK) |
| pyloudnorm | **2** | CPU | 0 | ✅ | none | ✅ | – | separate | ✅ | ✅ | MIT ✅ |
| ffmpeg `loudnorm`/`ebur128` | **1** | CPU | 0 | ✅ | – | ✅ | – | already installed **[V]** | ✅ | ✅ | LGPL/GPL tool |
| beat_this | **2** | torch ≥ 2 **[V]**, falls back to CPU automatically **[V]** | small | ✅ | any | ✅ | – | separate | ✅ CLI **[V]** | ✅ | MIT ✅ (weights in repo releases) |
| librosa | **2** | CPU | 0 | ✅ | – | ✅ | – | separate | ✅ | ✅ | ISC ✅ |
| all-in-one | **3** | needs **NATTEN** on Linux **[V]**; recent NATTEN versions' Pascal support unknown **[A]** | – | ✅ | – | ✅ (`--device cpu`) **[V]** | – | separate | ✅ | ⚠️ last push 2024-05 | MIT ✅ (pulls madmom, whose models are NC ⚠️) |
| SongFormer | **2** | torch, Python 3.10 conda **[V]** | unknown (benchmarked on L40) | unknown | – | unknown | – | separate | ✅ | ✅ | CC BY 4.0 ✅ (attribution); its SSL backbone weights' licences unverified ⚠️ |
| basic-pitch | **2** | CPU (TF) | 0 | ✅ | TF < 2.15.1 → Python 3.11 venv **[V]** | ✅ | ONNX option **[V]** | separate | ✅ CLI | ⚠️ | Apache-2.0 ✅ |
| Essentia | **2** | CPU | 0 | ✅ | – | ✅ | – | separate | ✅ | ✅ | AGPL-3.0 (tool OK) |
| music21 / pretty_midi / mido | **1–2** | CPU | 0 | ✅ | – | ✅ | – | separate | ✅ | ✅ | BSD / MIT ✅ |
| FluidSynth | **2** | CPU | 0 | ✅ | – | ✅ | – | `apt` package | ✅ `fluidsynth -ni` | ✅ | LGPL ✅ (check each SoundFont's licence) |
| Montreal Forced Aligner | **2** | CPU | 0 | ✅ | – | ✅ | – | **existing `mfa` conda env** **[V]** | ✅ | ✅ | MIT ✅ |
| whisperX | **3** | faster-whisper/CTranslate2 GPU builds target newer archs **[A]**; CPU int8 works **[A]** | – | ✅ CPU | – | ✅ | int8 | separate | ✅ | ✅ | BSD-2 ✅ |
| torchcrepe / RMVPE / pyworld / Rubber Band | **2** | torch or CPU | small | ✅ | – | ✅ | – | separate | ✅ | ✅ / ⚠️ | MIT / Apache / MIT / GPL ✅ |
| Stable Audio 3 Small | **2** | CPU model **[V]** | 0 | 433M params, likely OK **[A]** | torch 2.7.1, cu126 default **[V]** | ✅ | – | separate | ✅ | ✅ | Stability Community License ⚠️ (free under a revenue cap; read before release) |
| Stable Audio 3 Medium | **4** | **requires Flash-Attention 2** (Ampere+) **[V]** | 5–6.5 GB on H200 **[V]** | – | – | – | – | – | – | ✅ | same ⚠️ |
| HeartMuLa 3B | **5** | bf16 default (R3) **[V]** | fp32 3B ≈ 12 GB **[A]** | – | – | – | – | – | ✅ | ✅ | Apache-2.0 ✅ |
| YuE / YuE2 | **5** + licence | – | ~24 GB | – | – | – | – | – | – | ✅ | **weights CC BY-NC** ❌ |
| DeepFilterNet / resemble-enhance / voicefixer | **2–3** | – | small | ✅ | resemble-enhance needs deepspeed **[V]** | ✅ | – | separate | ✅ | mixed | MIT/Apache ✅, but **trained on speech** (risk to singing) **[A]** |
| AudioSR / Apollo | **4–5** | – | heavy diffusion **[A]** | – | – | slow | – | – | ✅ | ⚠️ | MIT / **CC BY-SA** |

---

## 6. Scores and tiers

Scores are 1–10 on: **Q** music-quality improvement, **C** practical compatibility,
**E** ease of integration, **H** hardware efficiency, **M** maintenance/reliability,
**L** long-term usefulness to JewGrass. **Hardware is a gate:** class 4–5 projects
cannot be Tier 1 or 2 regardless of total. Totals are out of 60.

| Project | Q | C | E | H | M | L | Total | Tier |
|---|---|---|---|---|---|---|---|---|
| ACE-Step repaint/cover/reference (existing install) | 9 | 8 | 8 | 7 | 8 | 10 | **50** | **1** |
| python-audio-separator (RoFormer + de-reverb + karaoke) | 8 | 8 | 9 | 7 | 8 | 9 | **49** | **1** |
| Pedalboard | 7 | 10 | 9 | 10 | 9 | 9 | **54** | **1** |
| pyloudnorm + ffmpeg loudnorm | 5 | 10 | 10 | 10 | 8 | 8 | **51** | **1** |
| Matchering | 7 | 9 | 9 | 10 | 7 | 8 | **50** | **1** |
| beat_this | 4 | 9 | 9 | 9 | 8 | 8 | **47** | **1** |
| librosa | 4 | 10 | 9 | 10 | 9 | 8 | **50** | **1** |
| Applio (RVC) | 8 | 6 | 7 | 6 | 8 | 10 | **45** | **2** |
| Seed-VC (zero-shot prototype) | 7 | 6 | 7 | 6 | 3 | 6 | **35** | **2** |
| SoulX-Singer-SVC | 7 | 5 | 6 | 5 | 6 | 7 | **36** | **2** |
| DDSP-SVC | 6 | 7 | 6 | 8 | 7 | 6 | **40** | **2** |
| acestep.cpp | 6 | 5 | 5 | 9 | 8 | 8 | **41** | **2** |
| ACE-Step base (lego/complete/extract) | 8 | 5 | 7 | 5 | 8 | 9 | **42** | **2** |
| audio.cpp | 6 | 5 | 5 | 8 | 8 | 7 | **39** | **2** |
| Demucs (htdemucs_6s, via audio-separator) | 6 | 8 | 9 | 7 | 7 | 7 | **44** | **2** |
| ZFTurbo MSST | 7 | 7 | 6 | 7 | 9 | 7 | **43** | **2** |
| basic-pitch | 5 | 7 | 7 | 9 | 5 | 8 | **41** | **2** |
| SongFormer | 5 | 6 | 6 | 6 | 6 | 8 | **37** | **2** |
| all-in-one | 5 | 4 | 6 | 6 | 3 | 7 | **31** | **2–3** |
| Montreal Forced Aligner (existing env) | 5 | 8 | 6 | 10 | 8 | 6 | **43** | **2** |
| Pitch correction (torchcrepe/RMVPE + pyworld or Rubber Band) | 6 | 8 | 5 | 9 | 7 | 7 | **42** | **2** |
| music21 + pretty_midi + FluidSynth (+ Surge via Pedalboard) | 5 | 9 | 6 | 10 | 8 | 8 | **46** | **2** |
| Stable Audio 3 Small (CPU textures) | 4 | 6 | 6 | 7 | 8 | 5 | **36** | **2** (licence check first) |
| Essentia | 4 | 7 | 6 | 9 | 8 | 6 | **40** | **2** (optional) |
| DawDreamer | 5 | 8 | 5 | 9 | 7 | 6 | **40** | **2** (alt. to Pedalboard) |
| ACE-Step LoRA (official trainer, cloud) | 9 | 2 | 6 | 2 | 8 | 10 | 37 | **3** |
| ACE-Step XL sft (cloud) | 9 | 2 | 8 | 2 | 8 | 8 | 37 | **3** |
| HeartMuLa 3B (cloud) | 8 | 2 | 6 | 2 | 7 | 6 | 31 | **3** |
| Stable Audio 3 Medium | 6 | 1 | 6 | 3 | 8 | 5 | 29 | **3** |
| AudioSR / Apollo | 5 | 3 | 5 | 2 | 5 | 4 | 24 | **3** (Apollo licence ⚠️) |
| YourMT3 / MT3 | 5 | 4 | 4 | 4 | 4 | 6 | 27 | **3** |
| Remiqora, StemForge, ComfyUI, Side-Step, YuE2, MusicGen, RAVE, Diff-MST, madmom models, so-vits-svc, spleeter, open-unmix | – | – | – | – | – | – | – | **Reject** (§8) |

**Tier 1 – immediately useful on the current laptop:** ACE-Step repaint/cover/reference,
python-audio-separator, Pedalboard, pyloudnorm/ffmpeg, Matchering, beat_this, librosa.

**Tier 2 – valuable, needs experiments or compromises:** Applio/RVC, zero-shot SVC
(Seed-VC, SoulX-Singer-SVC), DDSP-SVC, acestep.cpp, ACE-Step base, audio.cpp, Demucs 6-stem,
MSST, basic-pitch, SongFormer/all-in-one, MFA, pitch-correction kit, symbolic toolkit
(music21, pretty_midi, FluidSynth, Surge), Stable Audio 3 Small, Essentia, DawDreamer.

**Tier 3 – revisit with more compute:** ACE-Step LoRA training, XL sft, HeartMuLa, Stable
Audio 3 Medium, AudioSR/Apollo, YourMT3/MT3.

---

## 7. Best projects per production category

| Category | Best choice | Runner-up | Why |
|---|---|---|---|
| **Section regeneration, continuation, inpainting** | ACE-Step `repaint` on our turbo model | acestep.cpp (same tasks, quantised) | Already installed, MIT, same model that made the song, so timbre matches. |
| **Style transfer / re-arrangement of a good song** | ACE-Step `cover` with `audio_cover_strength` | ACE-Step `reference_audio` | Keeps melody/structure, changes production words. |
| **Adding or replacing an instrument part** | Separate → re-generate with ACE-Step base `lego` (Tier 2) | Render MIDI part via FluidSynth/Surge and mix | `lego` is base-only; the MIDI route is fully controllable and CPU-only. |
| **Music-to-MIDI** | basic-pitch (Apache-2.0) on separated stems | – | Best permissive option. MuScriptor and SheetSage2 weights are non-commercial. |
| **Band vocal identities** | RVC via Applio, trained on our own ACE-Step vocals | Seed-VC / SoulX-Singer-SVC zero-shot for quick trials | RVC needs ~10 min of clean audio and is the most proven on small GPUs. |
| **Vocal isolation** | audio-separator: BS-RoFormer or Mel-RoFormer vocals | Demucs htdemucs_ft | RoFormers are the current quality leaders in the UVR community model list. |
| **Lead vs backing vocals** | audio-separator karaoke RoFormers | – | Lets us treat doubles/harmonies separately. |
| **De-reverb / denoise of vocals** | audio-separator de-reverb and denoise RoFormers | DeepFilterNet (speech only, use carefully) | Music-trained models, same tool as separation. |
| **Pitch correction** | torchcrepe or RMVPE pitch track → snap to key (from librosa/music21) → pyworld resynthesis or Rubber Band shifts | Parselmouth PSOLA (StemForge's approach) | No mature open-source CLI auto-tune exists; this is a small script. |
| **Vocal timing correction** | MFA (already on the laptop) or beat grid from beat_this → Rubber Band time-stretch per phrase | whisperX word timestamps | Align phrases to the downbeat grid. |
| **Harmonisation** | music21 to choose intervals in key → Rubber Band formant-preserving shift of the lead → RVC through the member's voice | ACE-Step repaint with harmony words in the caption | Combines theory with conversion so harmonies keep the member's identity. |
| **Instrument separation** | Demucs `htdemucs_6s` (guitar, piano) and MDX23C DrumSep through audio-separator | MSST models | Banjo/fiddle have no dedicated public models **[A]**; they land in "other". |
| **Restoration / repair** | De-reverb, bleed-suppressor RoFormers; repaint for damaged bars | AudioSR (Tier 3) | Generative repair beats spectral repair for AI artefacts. |
| **Spectral editing** | Manual, on another computer (Audacity/Ardour) | librosa STFT scripts | No good headless spectral editor; rarely needed if repaint works. |
| **BPM and beats** | beat_this | librosa | MIT, accurate, has CPU fallback. |
| **Key / chords** | librosa chroma + key profile, or Essentia `KeyExtractor` | BTC (MIT, stale) | ACE-Step's own audio understanding needs the LM, which is off. |
| **Structure** | SongFormer (CC BY 4.0) | all-in-one (MIT, needs NATTEN) / msaf (CPU) | Also: ACE-Step lyrics already carry section tags, and we know the generation's BPM and length, so a beat grid plus our own lyrics often gives the structure for free. |
| **Similarity / consistency checks** | LAION CLAP embeddings (CC0) | openl3 | Useful later to check a new song "sounds like the band". |
| **Effects, EQ, compression, saturation, stereo** | Pedalboard (+ Airwindows/LSP VST3s if needed) | DawDreamer | Python, fast, scriptable presets. |
| **Loudness** | pyloudnorm + ffmpeg `loudnorm` (two-pass) | – | Standards-based, trivial. |
| **Mastering / reference match** | Matchering against our own approved master | Pedalboard chain | Same reference for every song = consistent EP. |
| **Multitrack DAW** | Ardour or Reaper on a desktop computer, fed by our stems | DawDreamer for scripted renders | The laptop is headless. |
| **Automated mixing** | Not recommended yet | automix-toolkit (research), Diff-MST (NC) | Research quality; rule-based Pedalboard presets are more predictable. |
| **Creative / motifs** | basic-pitch → pretty_midi motif library → music21 variations → FluidSynth/Surge render or ACE-Step `reference_audio` | anticipation (MIDI infilling) | Turns our best hooks into reusable, ownable material. |

---

## 8. Rejected projects and why

| Project | Reason |
|---|---|
| Remiqora | All-in-one GUI; reinstalls ACE-Step; cu128 Demucs pin; tested on 16 GB; bundles non-commercial weights (YuE2, MuScriptor, SheetSage2). Redundant with what we have. |
| StemForge | torch 2.13/cu130 pin: cannot run on Pascal with driver 550. GUI-centric. |
| ComfyUI-ACE-Step/ComfyUI-ACE-Step | Does not exist. ComfyUI in general: extra server and torch stack, no headless benefit for a single-model pipeline. |
| Side-Step | CC BY-NC-SA (non-commercial); 8–10 GB VRAM minimum. |
| fspecii/ace-step-ui, roblaughter/ace-step-studio | No licence: legally "all rights reserved". |
| YuE / YuE2 | Weights CC BY-NC; ~24 GB VRAM. |
| MusicGen (audiocraft) | Weights CC BY-NC; no vocals. |
| RAVE | CC BY-NC 4.0. |
| Diff-MST | CC BY-NC-SA 4.0. |
| madmom (for models) | Model files CC BY-NC-SA; beat_this replaces it. (Note: all-in-one and beat_this's optional `--dbn` pull madmom in.) |
| MuScriptor, SheetSage2, Vevo2, MuLaCover | Non-commercial weight licences (per audio.cpp's licence table and upstream cards). |
| so-vits-svc, diff-svc | Archived or stale; AGPL; superseded by RVC/Applio. |
| Spleeter, Open-Unmix | Clearly lower quality than RoFormer/Demucs models; no benefit. |
| Magenta | Archived. |
| resemble-enhance, voicefixer | Speech-trained; resemble-enhance needs deepspeed; risk of damaging sung vocals. DeepFilterNet kept only as a last-resort noise tool. |
| UVR GUI (Anjok07) | Needs a desktop; audio-separator gives the same models headless. |
| Stable Audio 3 Medium | Requires Flash-Attention 2 (Ampere+). |

---

## 9. Recommended architecture

### 9.1 Principles

1. **ACE-Step stays frozen.** Commit `ca1e85f`, its `.venv`, `gen.sh` and `gen.py` are not
   edited. New ACE-Step tasks get **new scripts** that run in the same venv with
   `uv run --no-sync` and add no packages.
2. **One tool, one environment, one command.** Each tool family lives in
   `~/band/tools/<name>/.venv`, created with `uv`, with torch pinned from the cu126 index.
   (`uv`'s cache hard-links identical wheels, so duplicate torch installs cost little
   extra disk; 497 GB is free anyway.)
3. **Files are the interface.** Every step reads WAV/JSON/MIDI files and writes new ones.
   No step overwrites its input. No database, no web server, no orchestrator.
4. **One GPU job at a time.** Every GPU command is wrapped in
   `flock ~/band/.gpu.lock <command>`. The process exits when done, which frees VRAM
   cleanly (more reliable than unloading models inside a long-running process).
5. **Selective, not mandatory.** A song only goes through the stages it needs.
6. **Every output gets a sidecar.** A small JSON next to each file records tool, version,
   model, settings and input file, so any result can be reproduced or rolled back.

### 9.2 Layout

```
~/band/
  ACE-Step-1.5/            # frozen, untouched
  gen.sh, gen.py           # frozen, untouched
  bin/                     # small wrapper scripts, one per job (in git)
    band-gen               # wraps gen.sh, writes song folder + params.json
    band-repaint           # ACE-Step repaint/cover (same venv, new script)
    band-separate          # audio-separator
    band-analyse           # beat_this + librosa + pyloudnorm -> analysis.json
    band-master            # pedalboard chain and/or matchering
    band-voice             # Applio/RVC conversion (later)
  tools/
    sep/.venv              # audio-separator (+ demucs models), torch 2.10.0+cu126
    fx/.venv               # pedalboard, pyloudnorm, matchering, librosa, soundfile, beat_this (CPU torch)
    vc/                    # Applio checkout + .venv (later)
    midi/.venv             # basic-pitch (Python 3.11), pretty_midi, music21 (later)
  models/                  # downloaded model files, one subfolder per tool
  songs/
    your-landlord/
      song.json            # title, owner member, status, chosen take
      00-gen/              # original take(s) + params.json (caption, lyrics, seed, bpm, secs, model, commit)
      10-analysis/         # analysis.json (bpm, beats, downbeats, key, sections, LUFS)
      20-stems/<model>/    # vocals.wav, instrumental.wav, ...
      30-vocals/           # dereverb, lead/backing, tuned, converted-<member>
      40-edits/            # repaint_<start>-<end>_<seed>.wav
      50-mix/              # mix.wav + chain.json
      60-master/           # master.wav + report.json
      NOTES.md             # what was kept and why
  library/
    voices/<member>/       # RVC model + index + dataset manifest (source takes listed)
    presets/               # pedalboard chains as JSON (e.g. "porch-tape", "glitch-bus")
    references/            # approved masters used as Matchering references
    motifs/                # MIDI + audio hooks with provenance
    arrangements/          # section templates: caption blocks + lyric tag skeletons
```

`bin/` scripts and the `library/*` manifests (not audio) belong in the JewGrass repo,
so the pipeline is version-controlled. Audio stays on the laptop.

### 9.3 Workflow mapped to the brief's eight stages

| Stage | Tool | GPU? | Typical trigger |
|---|---|---|---|
| 1 Generate | `gen.sh` (unchanged) | yes, ~6 min | Every song |
| 2 Analyse | beat_this + librosa + pyloudnorm | CPU | Every keeper |
| 3 Separate | audio-separator | yes (or slow CPU) | Only when a part needs work |
| 4 Vocals and identities | de-reverb/karaoke RoFormers → pitch/timing scripts → RVC member model | yes | When a vocal is weak or a member must be enforced |
| 5 Instruments | Demucs 6-stem / DrumSep; Pedalboard per stem; MIDI re-render | mixed | When one instrument is weak |
| 6 Edit / regenerate | ACE-Step repaint/cover (turbo now; base lego later) | yes | Weak section, bad ending, cut-off outro |
| 7 Mix and master | Pedalboard presets, Matchering vs. library reference, loudness check | CPU | Every release candidate |
| 8 Store | `songs/<slug>` + `library/` | – | Always |

**Why not one big Python environment?** ACE-Step pins torch 2.10, transformers 4.57,
numpy 2.5; Applio pins torch 2.11, transformers 5.x, numpy 2.4; basic-pitch needs
TensorFlow on Python 3.11 **[V]**. A shared environment would force upgrades to
ACE-Step's stack, which the brief forbids.

### 9.4 Building the long-term assets

- **Three or more vocal identities:** generate each member's voice with ACE-Step
  (captions from `stage-2-local-option.md` Batches 2–3), keep 3–5 best takes per member,
  isolate and de-reverb with audio-separator, collect 10–20 min per member, train one RVC
  model per member in Applio, and pass every future take for that member through it.
  Keep the dataset manifest with each model so it is provably built from our own
  material. Later (cloud), a LoRA on finished songs reinforces the band sound itself.
- **Motif library:** basic-pitch on the best banjo/fiddle stems → trim to 1–4 bar hooks
  with pretty_midi → store MIDI plus the source audio clip → reuse via FluidSynth/Surge
  renders, via ACE-Step `reference_audio`, or (base model) `lego`.
- **Production presets:** Pedalboard chains saved as JSON; one per band texture.
- **Arrangement templates:** lyric tag skeletons and caption blocks that worked, stored
  with the seeds that proved them.
- **Musical signature:** the same Matchering reference and the same mastering chain
  across the EP; CLAP similarity scores to flag songs drifting from the band sound.

---

## 10. Prioritised roadmap

Each stage only starts when the previous one is judged useful. Every experiment is
independent and reversible.

| Stage | Goal | Experiments | Installs? |
|---|---|---|---|
| **0. Housekeeping** | Know what we have | Put `gen.sh`/`gen.py` into git (copies, not edits); create `songs/` folders for Your Landlord and Cold Hame with `params.json`; measure loudness with the existing ffmpeg (`ffmpeg -i x.wav -af ebur128 -f null -`); listen for cut-off outros | None |
| **1. Quick wins** | Improve finished songs | **E1** stems and vocal clean-up; **E2** analysis + mastering; **E3** repaint a weak section | E1: one venv; E2: one CPU venv; E3: none |
| **2. Identity** | Consistent band members | E4 zero-shot SVC trial (Seed-VC `--fp16 False` and SoulX-Singer-SVC) on one member; E5 Applio RVC training for the Frontman; E6 pitch/timing correction script | Separate venvs |
| **3. Control** | Arrange and reuse | E7 basic-pitch motif library; E8 structure detection (SongFormer vs. our lyric tags + beat grid); E9 ACE-Step base model trial for `lego`/`complete` in 6 GB; E10 acestep.cpp build with sm_61 to compare quality/VRAM | Separate venvs / C++ build |
| **4. Cloud (optional)** | Things the 1060 cannot do | E11 ACE-Step LoRA on a rented 24 GB GPU using the official MIT trainer; E12 XL sft generation of the EP's best songs; E13 HeartMuLa comparison | Rented machine only |

---

## 11. The first three experiments

All three can run in any order. None touches `~/band/ACE-Step-1.5`, its `.venv`,
`gen.sh` or `gen.py`. **Do not run any install command below until approved.**

Common safety check before and after each experiment (proves ACE-Step is untouched):

```bash
cd ~/band/ACE-Step-1.5
git rev-parse --short HEAD                                # must print ca1e85f
git status --porcelain | sha256sum > /tmp/ace-git.before  # (.after afterwards; compare)
uv pip freeze --python .venv/bin/python | sha256sum > /tmp/ace-pip.before
# after the experiment: repeat into *.after files, then
diff /tmp/ace-git.before /tmp/ace-git.after && diff /tmp/ace-pip.before /tmp/ace-pip.after && echo UNCHANGED
# and a 30 s smoke generation:
~/band/gen.sh "$(cat ~/band/<caption-file>)" "$(cat ~/band/<lyrics-file>)" 30 1111 smoke-after 140
```

### E1. Stem separation and vocal clean-up (python-audio-separator)

| | |
|---|---|
| **Testing** | Can we split a finished ACE-Step song into clean vocals and instrumental on the 1060, then de-reverb the vocal and split lead from backing? |
| **Why** | Every later improvement (vocal identity, tuning, per-instrument mixing, motif extraction) needs clean stems. |
| **Dependencies** | New venv: `torch==2.10.0` + `torchaudio==2.10.0` from the cu126 index, `audio-separator[cpu]` (CPU onnxruntime for ONNX models; RoFormer models run on torch/GPU). |
| **Hardware** | GPU in fp32 (no `--use_autocast`, no `--use_native_fp16`). VRAM unknown for a 4-min song **[A]**; reduce `--mdxc_segment_size` if it runs out. RAM: expected < 4 GB **[A]**. Disk: models ~0.2–0.7 GB each **[A]**. |
| **Test input** | `~/band/outputs/cold-hame-full-2222.wav` (and a 60 s excerpt first: `ffmpeg -i cold-hame-full-2222.wav -t 60 -c copy ~/band/songs/cold-hame/00-gen/excerpt60.wav`). |
| **Expected output** | `20-stems/bs-roformer/{vocals,instrumental}.wav`; `30-vocals/dereverb.wav`; `30-vocals/{lead,backing}.wav`; timing and peak VRAM noted. |
| **Success** | (1) `torch.cuda.get_arch_list()` includes `sm_61` and the log shows CUDA in use; (2) no NaN/silent output; (3) by ear, vocals have little instrument bleed and instrumental has no audible lead vocal; (4) vocals + instrumental summed vs. original leaves only a quiet residual (null test); (5) 4-min song separates in a usable time (note it; no target set because we have no benchmark); (6) ACE-Step safety check prints UNCHANGED. |
| **Failure** | "no kernel image" (wrong torch wheel); CUDA OOM even at small segment size; NaN output; resolver replaces torch with a cu128/CPU build; separation quality worse than the original song for our purposes. |
| **Rollback** | `rm -rf ~/band/tools/sep ~/band/models/separator` and delete the `20-stems`/`30-vocals` folders. Nothing else was touched. |

Installation (only if approved):

```bash
mkdir -p ~/band/tools/sep ~/band/models/separator && cd ~/band/tools/sep
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch==2.10.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv/bin/python "audio-separator[cpu]"
# verify torch was NOT replaced:
.venv/bin/python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_arch_list())"
# expect: 2.10.0+cu126 True [... 'sm_61' ...]
```

Run (examples; model names from the package's model list **[V]**):

```bash
S=~/band/songs/cold-hame
flock ~/band/.gpu.lock ~/band/tools/sep/.venv/bin/audio-separator "$S/00-gen/excerpt60.wav" \
  -m model_bs_roformer_ep_317_sdr_12.9755.ckpt \
  --model_file_dir ~/band/models/separator --output_dir "$S/20-stems/bs-roformer" --output_format WAV
# then the full song, then de-reverb and karaoke on the vocal stem, e.g.
#   -m dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt
#   -m mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt
~/band/tools/sep/.venv/bin/audio-separator --list_models --list_filter vocals | head -30
```

### E2. Analysis and mastering chain (CPU only)

| | |
|---|---|
| **Testing** | Do a loudness/tonal-balance pass and reference mastering audibly improve our raw ACE-Step WAVs, and can analysis produce a reliable BPM/beat/key sheet? |
| **Why** | Mastering is the cheapest audible gain, and consistent loudness/tone across songs is what makes an EP sound like one band. Analysis output feeds repaint windows, tuning and motif work. |
| **Dependencies** | New venv `fx`: `pedalboard`, `pyloudnorm`, `matchering`, `librosa`, `soundfile`, `beat_this` with **CPU** torch (keeps the GPU free and avoids any CUDA question). |
| **Hardware** | CPU only, no VRAM. RAM well under 2 GB per song **[A]**. |
| **Test input** | `your-landlord-len240.wav` and `cold-hame-full-2222.wav`. |
| **Expected output** | `10-analysis/analysis.json` (BPM, beat and downbeat times, key estimate, integrated LUFS, true peak, loudness range); `60-master/pedalboard.wav`, `60-master/matchering.wav` (Cold Hame mastered to the approved Your Landlord master, or vice versa); `report.json` with before/after loudness. |
| **Success** | (1) beat_this BPM within ±1 of the BPM we generated at (140) or a clean half/double; downbeats line up with bar starts by ear; (2) masters hit the chosen target (e.g. −14 LUFS integrated ±0.5 LU, true peak ≤ −1 dBTP) with no audible clipping; (3) in a blind A/B (loudness-matched so "louder" doesn't win), the master is preferred in at least 2 of 3 trials per song; (4) the two songs sound closer in tone after Matchering. |
| **Failure** | Mastered versions sound smaller, harsher or pumped; Matchering errors on our files; analysis BPM wildly off on glitch/stutter sections (then treat analysis as advisory). |
| **Rollback** | `rm -rf ~/band/tools/fx` and the generated `10-analysis`/`60-master` files. |

Installation (only if approved):

```bash
mkdir -p ~/band/tools/fx && cd ~/band/tools/fx
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch==2.10.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python pedalboard pyloudnorm matchering librosa soundfile \
  beat-this
```

Starting chain to try (a preset, not a final answer): high-pass 30 Hz → gentle
compressor (ratio 2:1, slow attack) → a little saturation on the 808 bus if stems exist →
limiter → loudness normalise with pyloudnorm. Compare against Matchering with our
own approved master as reference.

### E3. Repaint a weak section with the existing ACE-Step install

| | |
|---|---|
| **Testing** | Can ACE-Step turbo regenerate one 20–30 s section of a finished 4-min song on the 1060 without changing the rest? |
| **Why** | This is the core "keep the excellent material, fix the weak part" capability, and it needs no new software. It also fixes cut-off outros (repaint/extend the last seconds). |
| **Dependencies** | None. A **new** script `~/band/repaint.py` that reuses the loading approach in `gen.py` (fp32 compute, direct load, no offload, SDPA) and sets `task_type="repaint"`, `src_audio`, `repainting_start`, `repainting_end` per the [Inference API](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md). First step is reading `gen.py` to see how it calls ACE-Step at commit `ca1e85f` (the API doc is for the latest commit; parameter names may differ **[A]**). |
| **Hardware** | Same as generation plus a VAE **encode** of the source song. With VRAM already at 5.9/6 GB this is the main risk **[A]**. Fallback: cut a 60–90 s excerpt around the section with ffmpeg, repaint that, and splice it back with a short crossfade at bar lines from E2's beat grid. |
| **Test input** | `your-landlord-len240.wav`; pick the weakest section by ear (e.g. the second chorus) and give exact start/end seconds; same caption, lyrics and BPM; 4 seeds. |
| **Expected output** | `40-edits/repaint_<start>-<end>_<seed>.wav` × 4, with sidecar JSON. |
| **Success** | (1) runs without OOM in about the time of a normal generation; (2) outside the window, audio is unchanged or near-identical (null test vs. original outside the window); (3) the joins are not audible; (4) at least 1 of 4 seeds is judged better than the original section; (5) ACE-Step safety check prints UNCHANGED. |
| **Failure** | OOM during encode or decode; changes leak outside the window; timbre or key of the new section doesn't match; obvious seams. |
| **Rollback** | Delete `~/band/repaint.py` and `40-edits/`. ACE-Step itself was never modified (safety check proves it). |

---

## 12. Outstanding uncertainties

| # | Uncertainty | How to resolve |
|---|---|---|
| U1 | Exact `gen.py` internals: whether it builds `GenerationParams` and can be reused for repaint/cover at commit `ca1e85f` | Read `~/band/gen.py` (read-only) at the start of E3 |
| U2 | VRAM headroom for repaint/cover (VAE encode of a 4-min source on top of 5.9 GB) | E3; excerpt-and-splice fallback |
| U3 | Whether the ACE-Step **base** model fits in 6 GB with CFG and our fp32-compute patch | Stage 3 experiment E9 |
| U4 | VRAM and speed of RoFormer separation on the 1060 in fp32 | E1 |
| U5 | Whether onnxruntime-gpu wheels still include sm_61 kernels | Avoided by using `audio-separator[cpu]`; test only if ONNX models are too slow |
| U6 | Whether torch 2.11 (Applio's pin) has a cu126 wheel, and whether Applio runs on torch 2.10 instead | Check the wheel index before E5; test inference before training |
| U7 | RVC training time and quality on the 1060 with 10–20 min of AI-generated vocals | E5 |
| U8 | SoulX-Singer-SVC VRAM needs and weight licence (Hugging Face card not reachable from the research environment) | Read the model card before E4 |
| U9 | Licences of community separation weights (most UVR/RoFormer checkpoints state none) | Treat stems as working material; the released audio is our generation processed by a tool, but ask before relying on a specific model commercially |
| U10 | acestep.cpp and audio.cpp CUDA builds on sm_61 with the system nvcc 12.0 | Stage 3 build test (E10); may need a newer CUDA toolkit installed locally (a toolkit install, not a driver change) |
| U11 | NATTEN support on Pascal (for all-in-one) | Only if SongFormer and the simple beat-grid approach both fall short |
| U12 | Stability AI Community License terms for Stable Audio 3 Small outputs | Read the licence before any release use |
| U13 | Applio Terms of Use require rights to the voice; our voices are ACE-Step-generated (MIT model, outputs ours) | Keep dataset manifests per member; never train on real artists |
| U14 | basic-pitch install on Python 3.11 with TF 2.15 on this CPU (AVX2 present on i7-8750H **[A]**) | E7 |
| U15 | No public banjo/fiddle-specific separation model found | Accept "other" stem, or regenerate the part with repaint/lego |
| U16 | Benchmarks: none were invented. All speed/VRAM numbers above are either from the laptop or from the projects' own docs. | Record real numbers in each experiment's sidecar JSON |

---

## 13. Final recommendation

**Implement first (Stage 0 + three experiments):**

1. Put the working `gen.sh`/`gen.py` under version control and start the `songs/` folder
   convention. No installs.
2. **E3 repaint**: the biggest quality lever, zero new dependencies, and it fixes cut-off
   outros.
3. **E1 audio-separator** in its own cu126 venv: unlocks every vocal and instrument
   improvement.
4. **E2 CPU analysis + mastering**: immediate, safe, audible gains and EP consistency.

**Then:** band voices with Applio/RVC trained on our own generated takes, after a cheap
zero-shot trial (Seed-VC with fp16 off, SoulX-Singer-SVC) to decide whether training is
worth it.

**Leave alone:** Remiqora, StemForge, ComfyUI, Side-Step, anything with non-commercial
weights (YuE2, MusicGen, RAVE, madmom models, MuScriptor, SheetSage2, Vevo2, Diff-MST),
unlicensed UIs, speech enhancers on sung vocals, and upstream ACE-Step updates until the
Pascal fixes land and are re-tested.

**Revisit with more compute (rented 24 GB GPU or a newer card):** ACE-Step LoRA training
with the official MIT trainer to lock in the band sound, XL sft for final renders, the
base model's lego/complete at full quality, HeartMuLa as a second opinion, and
Stable Audio 3 Medium for textures. A Turing-or-newer card (sm_75+) would also remove
almost every compatibility caveat in §2.
