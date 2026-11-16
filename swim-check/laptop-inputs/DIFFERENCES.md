# Swim 5555: laptop render vs Kaggle swim-check turbo 5555

Laptop render: 2026-11-14 00:33 UTC, `~/band/gen.sh swim-caption.txt swim-lyrics.txt 240 5555 swim-at-your-own-risk-5555 140`
(log: joblogs/swim5555.log, params: params-5555.json). Kaggle: v9 (2026-11-16), kaggle/render.py, baseline-run/swim-check.yaml @ 1ce54bb.

## Same on both
- ACE-Step-1.5 commit ca1e85f (laptop tree clean, no local diffs; Kaggle setup.sh pins ACE_COMMIT=ca1e85f)
- torch 2.10.0+cu126
- model acestep-v15-turbo, 8 steps, no CFG (guidance 7.0 -> 1.0 override), shift 3.0, no LM (thinking/use_cot_* all False)
- fp16 weights with fp32 compute (BAND_FP32_COMPUTE=1), attention sdpa, DiT kept on GPU
- batch_size 1, use_random_seed False, seeds [5555]; duration 240 s, bpm 140
- caption: identical text
- keyscale / timesignature: not set on the laptop (log: "keyscale: N/A"); empty strings on Kaggle (job key='' timesig='') -> same
- DCW correction active (same commit default)

## Different
1. **Lyrics: Kaggle drops a line.** kaggle/plan.py `clean_lyrics()` removes every line wholly in parentheses.
   Swim's verse 2 ends with `(Sustainability report, baby!)`, so Kaggle's lyrics have 1 line fewer than the laptop's.
   The lyric text is part of the conditioning, so even with identical noise the output would differ.
2. **Initial noise is drawn on the GPU.** `prepare_noise()` (turbo, models/turbo/modeling_acestep_v15_turbo.py ~1814) uses
   `torch.Generator(device="cuda").manual_seed(seed)` + `torch.randn(..., device="cuda")`. PyTorch's CUDA Philox randn
   maps counters to threads from a launch grid sized by the GPU's SM count, so the same seed can give different noise on a
   GTX 1060 Max-Q (10 SMs) vs a T4 (40 SMs). Laptop fingerprint for seed 5555, shape (1,6000,64), fp32:
   see noise-fingerprint-laptop.txt (sha256 272d6e3a8d125402). Running noise_fingerprint.py on a T4 confirms or rules it out.
3. **GPU arithmetic:** Pascal sm_61 vs Turing sm_75 — different kernels (sdpa backend, cuBLAS/TF32-free fp32 GEMMs), so tiny
   numerical differences, amplified over 8 steps. Small compared to 1 and 2.
4. **Checkpoint files:** laptop acestep-v15-turbo downloaded 2026-11-12/13 (model.safetensors sha256 3f6e0797fad420a3...,
   config.json 74745ff704ea4916..., silence_latent.pt a778e9dd942f5e8b...). Kaggle downloads via ensure_dit_model at run
   time with no pinned revision; if the HF repo changed since, weights could differ. Compare these hashes on Kaggle.
5. Kaggle turbo runs with the text encoder / VAE offload pattern of render.py (OFFLOAD only for XL); the laptop also offloads
   VAE/text encoder between stages. No numerical effect expected.

## Suggested next test (cheap)
On Kaggle: print noise_fingerprint.py for seed 5555 and the three checkpoint hashes, and render swim-check 5555 with the
laptop's exact lyrics (no clean_lyrics). If the noise sha differs, a seed can never be reproduced across these GPUs unless
the noise is drawn on the CPU (a render.py/gen.py change: generator on "cpu", then .to(device)).
