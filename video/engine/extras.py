"""Stage 5 extras: thumbnail and vertical social cut.

  python -m engine.extras SONG_DIR [--out OUT] [--jobs N] [--thumb-only]

storyboard.yaml may set:
  thumbnail: {shot: s16, at: 0.5, title: "DEED TO THE GLEN", laird: true}
  vertical:  {sections: [chorus-3], max_s: 45}

The vertical cut is re-composed, not cropped: the same storyboard is rendered
on a 1080x1920 canvas, so plates, props, signs and lyrics are laid out for
portrait (sizes scale with the 16:9-equivalent unit; text wraps narrower).
"""
import argparse
import json
import os
import subprocess
from multiprocessing import Pool

import numpy as np
from PIL import Image

from . import palette as P
from . import typography as T
from . import props as PR
from . import storyboard as SB
from .compositor import Compositor
from .render import find_master

_C = None


def _load(song_dir):
    a = json.load(open(os.path.join(song_dir, "analysis.json")))
    plan = SB.load(os.path.join(song_dir, "storyboard.yaml"), os.path.join(song_dir, "timing.yaml"), a)
    return plan, a


def thumbnail(song_dir, out):
    plan, a = _load(song_dir)
    cfg = plan["meta"].get("thumbnail", {})
    W, H = 1280, 720
    C = Compositor(plan, a, W, H)
    C.text_on = False
    sh = next((s for s in plan["shots"] if s["id"] == cfg.get("shot")), plan["shots"][len(plan["shots"]) // 2])
    i = int((sh["t0"] + cfg.get("at", 0.5) * (sh["t1"] - sh["t0"])) * a["fps"])
    img = C.frame(i).astype(np.float32) / 255
    # darken the left for the title
    ramp = np.clip(1 - np.linspace(0, 1, W) * 1.6, 0, 1)[None, :, None]
    img *= 1 - 0.45 * ramp
    if cfg.get("laird", True):
        spr = PR.laird_sprite(int(H * 0.86))
        h, w = spr.shape[:2]
        C._paste_rgb(img, spr, int(W * 0.80 - w / 2), H - h + 10, 1.0)
    title = cfg.get("title", plan["meta"]["song"].replace("-", " ").upper())
    words = title.split()
    size = int(H * 0.17)
    lay, (lw, lh) = T.layout_words(words, "stencil", size, int(W * 0.55))
    pad = size // 6
    for (word, x, y) in lay:
        sh_spr, _ = T.word_sprite(word, "stencil", size, P.PEAT)
        spr, _ = T.word_sprite(word, "stencil", size, P.BONE)
        C._paste_rgb(img, sh_spr, int(W * 0.05) + x - pad + 6, int(H * 0.12) + y - pad + 6, 0.8)
        C._paste_rgb(img, spr, int(W * 0.05) + x - pad, int(H * 0.12) + y - pad, 1.0)
    band, _ = T.word_sprite("JEWGRASS", "stencil", int(H * 0.06), P.WARNING_RED)
    C._paste_rgb(img, band, int(W * 0.05) - int(H * 0.01), int(H * 0.12) + lh + int(H * 0.02), 1.0)
    path = os.path.join(out, "thumbnail.jpg")
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path, quality=90)
    print("wrote", path, os.path.getsize(path) // 1024, "KB")


def _init(song_dir):
    global _C
    os.nice(10)
    plan, a = _load(song_dir)
    _C = Compositor(plan, a, 1080, 1920)


def _chunk(args):
    i0, i1, path, fps = args
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1080x1920",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p",
           "-threads", "1", "-x264-params", "keyint=60:min-keyint=60:scenecut=0", path]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(i0, i1):
        pr.stdin.write(_C.frame(i).tobytes())
    pr.stdin.close()
    pr.wait()
    return path


def vertical(song_dir, out, jobs):
    plan, a = _load(song_dir)
    cfg = plan["meta"].get("vertical", {})
    names = cfg.get("sections") or [s["name"] for s in plan["sections"] if s["name"].startswith("chorus")][-1:]
    shots = [s for s in plan["shots"] if s.get("at") in names]
    t0, t1 = min(s["t0"] for s in shots), max(s["t1"] for s in shots)
    t1 = min(t1, t0 + cfg.get("max_s", 45))
    fps = a["fps"]
    f0, f1 = int(t0 * fps), int(t1 * fps)
    work = os.path.join(out, "work-vertical")
    os.makedirs(work, exist_ok=True)
    step = 60 * max(1, (f1 - f0) // (60 * jobs * 2))
    chunks = [(i, min(i + step, f1), os.path.join(work, f"v-{i:06d}.mp4"), fps) for i in range(f0, f1, step)]
    with Pool(jobs, initializer=_init, initargs=(song_dir,)) as pool:
        segs = pool.map(_chunk, chunks, chunksize=1)
    lst = os.path.join(work, "list.txt")
    open(lst, "w").writelines(f"file '{os.path.abspath(s)}'\n" for s in segs)
    path = os.path.join(out, "vertical-cut.mp4")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                    "-ss", f"{f0 / fps:.6f}", "-t", f"{(f1 - f0) / fps:.6f}", "-i", find_master(song_dir),
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                    "-af", f"afade=t=in:d=0.3,afade=t=out:st={(f1 - f0) / fps - 1.0:.3f}:d=1.0",
                    "-shortest", "-movflags", "+faststart", path], check=True)
    for s in segs:
        os.remove(s)
    print("wrote", path, f"({t0:.1f}-{t1:.1f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song_dir")
    ap.add_argument("--out")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--thumb-only", action="store_true")
    a = ap.parse_args()
    out = a.out or os.path.join(a.song_dir, "out")
    os.makedirs(out, exist_ok=True)
    thumbnail(a.song_dir, out)
    if not a.thumb_only:
        vertical(a.song_dir, out, a.jobs)


if __name__ == "__main__":
    main()
