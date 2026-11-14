"""Repaint a time range of an existing song with ACE-Step 1.5 (2B turbo), same GTX 1060 settings as gen.py.

Usage (from ~/band/ACE-Step-1.5):
  uv run --no-sync python ../repaint.py SRC_WAV CAPTION_FILE LYRICS_FILE BPM START END OUT_DIR SEED[,SEED...]
Writes OUT_DIR/repaint_<start>-<end>_<seed>.wav and .json. The model loads once for all seeds.
BAND_* env vars work as in gen.py (the loading/fp32 patches below are copied from gen.py, which stays unchanged).
BAND_TE_CPU=1 (default): keep the Qwen3 text encoder on the CPU. Moving it to the GPU next to the resident DiT
runs out of memory on a 6 GB card when the source audio has just been encoded.
BAND_ENC_CHUNK_S=5 (default): VAE encode chunk length in seconds for the source audio. ACE-Step uses 15 s on
GPUs <= 8 GB, which runs out of memory next to the resident DiT on a 6 GB card. Overlap is 1 s.
"""
import os
import shutil
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

FP32_COMPUTE = os.environ.get("BAND_FP32_COMPUTE", "1") == "1"
DIT_DIRECT = os.environ.get("BAND_DIT_DIRECT", "1") == "1"
OFFLOAD_DIT = os.environ.get("BAND_OFFLOAD_DIT", "0") == "1"
ATTN = os.environ.get("BAND_ATTN", "sdpa" if FP32_COMPUTE else "eager")


# --- fp32 compute for modules flagged with _band_f32 (weights stay fp16) -------------
def _up(x):
    return x.float() if torch.is_tensor(x) and x.dtype in (torch.float16, torch.bfloat16) else x


def _f(t):
    return None if t is None else t.float()


_linear_fwd = nn.Linear.forward
_conv_fwd = nn.Conv1d._conv_forward
_convt_fwd = nn.ConvTranspose1d.forward
_emb_fwd = nn.Embedding.forward


def _linear(self, x):
    if getattr(self, "_band_f32", False):
        return F.linear(_up(x), self.weight.float(), _f(self.bias))
    return _linear_fwd(self, x)


def _conv(self, x, weight, bias):
    if getattr(self, "_band_f32", False):
        return _conv_fwd(self, _up(x), weight.float(), _f(bias))
    return _conv_fwd(self, x, weight, bias)


def _convt(self, x, output_size=None):
    if getattr(self, "_band_f32", False):
        op = self._output_padding(x, output_size, self.stride, self.padding, self.kernel_size, 1, self.dilation)
        return F.conv_transpose1d(_up(x), self.weight.float(), _f(self.bias), self.stride,
                                  self.padding, op, self.groups, self.dilation)
    return _convt_fwd(self, x, output_size)


def _emb(self, x):
    out = _emb_fwd(self, x)
    return _up(out) if getattr(self, "_band_f32", False) else out


nn.Linear.forward = _linear
nn.Conv1d._conv_forward = _conv
nn.ConvTranspose1d.forward = _convt
nn.Embedding.forward = _emb

# --- load the DiT straight onto the GPU ------------------------------------------------
if DIT_DIRECT:
    import transformers

    _from_pretrained = transformers.AutoModel.from_pretrained.__func__

    def _direct(cls, path, *args, **kwargs):
        if "acestep-v15" in str(path) and "device_map" not in kwargs:
            kwargs["device_map"] = {"": "cuda"}
            kwargs["low_cpu_mem_usage"] = True
        return _from_pretrained(cls, path, *args, **kwargs)

    transformers.AutoModel.from_pretrained = classmethod(_direct)

from acestep.handler import AceStepHandler  # noqa: E402
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402

src_wav, caption_file, lyrics_file, bpm, rs, re_, out_dir, seeds = sys.argv[1:9]
rs, re_, bpm = float(rs), float(re_), int(bpm)
seeds = [int(s) for s in seeds.split(",")]
os.makedirs(out_dir, exist_ok=True)
tmp_dir = os.path.join(out_dir, ".tmp")
os.makedirs(tmp_dir, exist_ok=True)
print(f"BAND settings: FP32_COMPUTE={FP32_COMPUTE} DIT_DIRECT={DIT_DIRECT} OFFLOAD_DIT={OFFLOAD_DIT} ATTN={ATTN}", flush=True)

t0 = time.time()
dit = AceStepHandler()
status, ok = dit.initialize_service(
    project_root=os.getcwd(), config_path="acestep-v15-turbo", device="cuda",
    offload_to_cpu=True, offload_dit_to_cpu=OFFLOAD_DIT, quantization=None, compile_model=False,
)
print(status)
if not ok:
    sys.exit("model init failed")
if FP32_COMPUTE:
    for m in dit.model.modules():
        m._band_f32 = True
for m in dit.model.modules():
    cfg = getattr(m, "config", None)
    if cfg is not None and hasattr(cfg, "_attn_implementation"):
        cfg._attn_implementation = ATTN
TE_CPU = os.environ.get("BAND_TE_CPU", "1") == "1"
if TE_CPU:
    from contextlib import contextmanager

    _ctx = dit._load_model_context

    @contextmanager
    def _ctx_keep_te_on_cpu(model_name, *a, **k):
        if model_name == "text_encoder":
            yield
            return
        with _ctx(model_name, *a, **k):
            yield

    def _to(obj, device, dtype=None):
        if torch.is_tensor(obj):
            obj = obj.to(device)
            return obj.to(dtype) if dtype is not None and obj.is_floating_point() else obj
        if isinstance(obj, (list, tuple)):
            return type(obj)(_to(o, device, dtype) for o in obj)
        if isinstance(obj, dict):
            return {key: _to(v, device, dtype) for key, v in obj.items()}
        return obj

    class _CPUTextEncoder:
        """Runs the text encoder on the CPU; inputs are moved there and outputs back to the GPU."""

        def __init__(self, m):
            self._m = m.to("cpu")

        def __call__(self, *a, **k):
            out = self._m(*_to(a, "cpu"), **_to(k, "cpu"))
            for key in list(out.keys()):
                out[key] = _to(out[key], dit.device, dit.dtype)
            return out

        def embed_tokens(self, ids):
            return self._m.embed_tokens(ids.to("cpu")).to(dit.device).to(dit.dtype)

        def __getattr__(self, name):
            return getattr(self._m, name)

    dit._load_model_context = _ctx_keep_te_on_cpu
    dit.text_encoder = _CPUTextEncoder(dit.text_encoder)
print(f"TE_CPU={TE_CPU}", flush=True)
ENC_CHUNK_S = float(os.environ.get("BAND_ENC_CHUNK_S", "5"))
_tiled_encode = dit.tiled_encode


def _small_chunk_encode(audio, chunk_size=None, overlap=None, offload_latent_to_cpu=True):
    return _tiled_encode(audio, chunk_size=chunk_size or int(48000 * ENC_CHUNK_S),
                         overlap=overlap or 48000, offload_latent_to_cpu=offload_latent_to_cpu)


dit.tiled_encode = _small_chunk_encode
t_load = time.time() - t0
print(f"ENC_CHUNK_S={ENC_CHUNK_S}", flush=True)
print(f"LOAD_SECONDS={t_load:.1f}", flush=True)

caption = open(caption_file).read().strip()
lyrics = open(lyrics_file).read().strip()
for seed in seeds:
    # Free whatever the previous seed left on the GPU (a second seed in one process otherwise runs out of memory).
    result = None
    gc.collect()
    torch.cuda.empty_cache()
    params = GenerationParams(
        task_type="repaint", src_audio=src_wav, repainting_start=rs, repainting_end=re_,
        chunk_mask_mode="explicit", caption=caption, lyrics=lyrics, bpm=bpm, seed=seed, shift=3.0,
        thinking=False, use_cot_metas=False, use_cot_caption=False, use_cot_language=False,
    )
    config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[seed], audio_format="wav")
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    result = generate_music(dit, None, params, config, save_dir=tmp_dir)
    t_gen = time.time() - t1
    peak = torch.cuda.max_memory_allocated() / 2**30
    if not result.success:
        print(f"SEED {seed} FAILED: {result.error} PEAK_VRAM_TORCH_GB={peak:.2f}", flush=True)
        continue
    name = f"repaint_{rs:g}-{re_:g}_{seed}"
    dst = os.path.join(out_dir, name + ".wav")
    shutil.move(result.audios[0]["path"], dst)
    json.dump({"task_type": "repaint", "src_audio": src_wav, "repainting_start": rs, "repainting_end": re_,
               "chunk_mask_mode": "explicit", "caption": caption, "lyrics_file": lyrics_file, "bpm": bpm,
               "seed": seed, "load_seconds": round(t_load, 1), "gen_seconds": round(t_gen, 1),
               "peak_vram_torch_gb": round(peak, 2),
               "band_settings": {"FP32_COMPUTE": FP32_COMPUTE, "DIT_DIRECT": DIT_DIRECT,
                                 "OFFLOAD_DIT": OFFLOAD_DIT, "ATTN": ATTN}},
              open(os.path.join(out_dir, name + ".json"), "w"), indent=2)
    print(f"SEED {seed} OK GEN_SECONDS={t_gen:.1f} PEAK_VRAM_TORCH_GB={peak:.2f} OUTPUT={dst}", flush=True)
