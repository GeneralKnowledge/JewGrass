"""Render a list of ACE-Step 1.5 takes on one GPU, loading the model once.

Usage (from the ACE-Step-1.5 checkout, with CUDA_VISIBLE_DEVICES set to one card):
  python /path/to/kaggle/render.py JOBS_JSON GPU_INDEX OUT_DIR
Renders the jobs whose "gpu" equals GPU_INDEX into OUT_DIR/<slug>/<slug>-<seed>.wav (+ .mp3 and .json).
Jobs whose .wav already exists are skipped, so a rerun picks up where the last one stopped.
BAND_DEADLINE (unix time): don't start a new take after this, so the run ends inside Kaggle's 12 hours.

Same settings as the laptop's gen.py (laptop/patches/gen.py.patch), so a seed sounds the way it does there:
fp32 compute with fp16 weights (fp16 compute gives NaN latents on pre-Ampere cards, and the T4 and P100 are
pre-Ampere), DiT loaded straight to the GPU, SDPA attention, turbo DiT, no language model. The only change is
that nothing is offloaded to the CPU: Kaggle's cards have 16 GB.
"""
import gc
import json
import os
import shutil
import subprocess
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

FP32_COMPUTE = os.environ.get("BAND_FP32_COMPUTE", "1") == "1"
ATTN = os.environ.get("BAND_ATTN", "sdpa" if FP32_COMPUTE else "eager")
OFFLOAD = os.environ.get("BAND_OFFLOAD", "0") == "1"
DEADLINE = float(os.environ.get("BAND_DEADLINE", "0") or 0)


# --- fp32 compute for modules flagged with _band_f32 (weights stay fp16); copied from the laptop's gen.py ---------
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

# --- load the DiT straight onto the GPU --------------------------------------------------------------------------
import transformers  # noqa: E402

_from_pretrained = transformers.AutoModel.from_pretrained.__func__


def _direct(cls, path, *args, **kwargs):
    if "acestep-v15" in str(path) and "device_map" not in kwargs:
        kwargs["device_map"] = {"": "cuda"}
        kwargs["low_cpu_mem_usage"] = True
    return _from_pretrained(cls, path, *args, **kwargs)


transformers.AutoModel.from_pretrained = classmethod(_direct)

from acestep.handler import AceStepHandler  # noqa: E402
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402


def log(msg):
    print(f"[gpu{gpu} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


jobs_file, gpu, out_root = sys.argv[1], int(sys.argv[2]), sys.argv[3]
jobs = [j for j in json.load(open(jobs_file)) if j["gpu"] == gpu]
todo = [j for j in jobs if not os.path.exists(os.path.join(out_root, j["slug"], f"{j['slug']}-{j['seed']}.wav"))]
log(f"{len(jobs)} takes for this card, {len(todo)} still to do. "
    f"FP32_COMPUTE={FP32_COMPUTE} ATTN={ATTN} OFFLOAD={OFFLOAD} card={torch.cuda.get_device_name(0)}")
if not todo:
    sys.exit(0)

t0 = time.time()
dit = AceStepHandler()
status, ok = dit.initialize_service(
    project_root=os.getcwd(), config_path="acestep-v15-turbo", device="cuda",
    offload_to_cpu=OFFLOAD, offload_dit_to_cpu=False, quantization=None, compile_model=False,
)
print(status, flush=True)
if not ok:
    sys.exit("model init failed")
if FP32_COMPUTE:
    for m in dit.model.modules():
        m._band_f32 = True
for m in dit.model.modules():
    cfg = getattr(m, "config", None)
    if cfg is not None and hasattr(cfg, "_attn_implementation"):
        cfg._attn_implementation = ATTN
log(f"model loaded in {time.time() - t0:.0f} s, {torch.cuda.memory_allocated() / 2**30:.1f} GB on the card")

tmp_dir = os.path.join(out_root, f".tmp-gpu{gpu}")
done = failed = 0
for j in todo:
    if DEADLINE and time.time() > DEADLINE:
        log("time budget reached, stopping before the next take (rerun to finish the rest)")
        break
    slug, seed = j["slug"], j["seed"]
    out_dir = os.path.join(out_root, slug)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    gc.collect()
    torch.cuda.empty_cache()
    params = GenerationParams(
        caption=j["caption"], lyrics=j["lyrics"], bpm=j["bpm"],
        keyscale=j["key"], timesignature=j["timesig"],
        duration=float(j["seconds"]), seed=seed, shift=3.0,
        thinking=False, use_cot_metas=False, use_cot_caption=False, use_cot_language=False,
    )
    config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[seed], audio_format="wav")
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    try:
        result = generate_music(dit, None, params, config, save_dir=tmp_dir)
    except Exception as e:  # one bad take shouldn't end the batch
        result = None
        err = repr(e)
    secs = time.time() - t1
    peak = torch.cuda.max_memory_allocated() / 2**30
    if result is None or not result.success:
        failed += 1
        log(f"FAILED {slug} seed {seed} after {secs:.0f} s: {err if result is None else result.error}")
        continue
    base = os.path.join(out_dir, f"{slug}-{seed}")
    shutil.move(result.audios[0]["path"], base + ".wav")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", base + ".wav", "-b:a", "320k", base + ".mp3"])
    meta = dict(j, render_seconds=round(secs, 1), peak_vram_gb=round(peak, 2), card=torch.cuda.get_device_name(0),
                settings={"fp32_compute": FP32_COMPUTE, "attn": ATTN, "dit": "acestep-v15-turbo", "shift": 3.0})
    json.dump(meta, open(base + ".json", "w"), indent=1)
    done += 1
    log(f"done {slug} seed {seed}: {j['seconds']:.0f} s of music in {secs:.0f} s, peak {peak:.1f} GB")
shutil.rmtree(tmp_dir, ignore_errors=True)
log(f"finished: {done} rendered, {failed} failed, {len(todo) - done - failed} left for next time")
