"""Render a song's video: chunks in parallel -> H.264 segments -> final MP4.

  python -m engine.render SONG_DIR [--out OUT_DIR] [--size 1920x1080]
         [--jobs 4] [--frames A:B] [--stills]

SONG_DIR holds storyboard.yaml, timing.yaml, analysis.json and master audio
(master.wav / master.flac). Outputs (stage 5):
  video-<W>p.mp4    H.264 High, yuv420p, CRF 18, AAC 320k, +faststart
  contact-sheet.jpg one frame from the middle of each shot
  frames.md5        md5 of every rendered frame (before encoding), for determinism checks

Workers are low priority (nice); on the laptop run the whole thing through
~/band/heavy.sh as well.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image

from .compositor import Compositor
from . import storyboard as SB

_C = None


def _init(song_dir, W, H):
    global _C
    os.nice(10)
    a = json.load(open(os.path.join(song_dir, "analysis.json")))
    plan = SB.load(os.path.join(song_dir, "storyboard.yaml"), os.path.join(song_dir, "timing.yaml"), a)
    _C = Compositor(plan, a, W, H)


def _render_chunk(args):
    i0, i1, path, fps, W, H = args
    # Each chunk is encoded straight to final quality (one x264 thread per worker,
    # so the encode is deterministic too); chunks are joined by stream copy.
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-profile:v", "high",
           "-pix_fmt", "yuv420p", "-threads", "1", "-x264-params", "keyint=60:min-keyint=60:scenecut=0",
           path]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    sums = []
    for i in range(i0, i1):
        fr = _C.frame(i)
        sums.append(f"{i} {hashlib.md5(fr.tobytes()).hexdigest()}")
        pr.stdin.write(fr.tobytes())
    pr.stdin.close()
    if pr.wait() != 0:
        raise RuntimeError(f"ffmpeg failed on chunk {i0}-{i1}")
    return path, sums


def find_master(song_dir):
    for n in ("master.wav", "master.flac", "master.mp3", "master.opus"):
        p = os.path.join(song_dir, n)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("no master.wav/.flac/.mp3 in " + song_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song_dir")
    ap.add_argument("--out")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--frames", help="A:B, render only this frame range (preview)")
    ap.add_argument("--stills", action="store_true", help="contact sheet only")
    a = ap.parse_args()
    W, H = map(int, a.size.split("x"))
    out = a.out or os.path.join(a.song_dir, "out")
    work = os.path.join(out, "work")
    os.makedirs(work, exist_ok=True)

    analysis = json.load(open(os.path.join(a.song_dir, "analysis.json")))
    plan = SB.load(os.path.join(a.song_dir, "storyboard.yaml"), os.path.join(a.song_dir, "timing.yaml"), analysis)
    fps, n = analysis["fps"], analysis["frames"]
    print(f"storyboard OK: {len(plan['shots'])} shots, {n} frames at {fps} fps, {W}x{H}")

    # Build once here too: catches asset errors before the workers start, and makes the contact sheet.
    t0 = time.time()
    _init(a.song_dir, W, H)
    print(f"assets ready in {time.time() - t0:.1f}s")

    # Contact sheet: middle frame of each shot.
    thumbs = []
    for sh in plan["shots"]:
        i = int((sh["t0"] + sh["t1"]) / 2 * fps)
        thumbs.append(Image.fromarray(_C.frame(min(i, n - 1))).resize((480, int(480 * H / W))))
    cols = 4
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 480, rows * thumbs[0].height), (20, 20, 20))
    for k, im in enumerate(thumbs):
        sheet.paste(im, ((k % cols) * 480, (k // cols) * im.height))
    sheet.save(os.path.join(out, "contact-sheet.jpg"), quality=88)
    if a.stills:
        return

    f0, f1 = (0, n)
    if a.frames:
        f0, f1 = (int(x) for x in a.frames.split(":"))
    step = max(60, ((f1 - f0) // (a.jobs * 3) + 59) // 60 * 60)  # whole GOPs
    chunks = [(i, min(i + step, f1), os.path.join(work, f"seg-{i:06d}.mp4"), fps, W, H)
              for i in range(f0, f1, step)]
    t0 = time.time()
    with Pool(a.jobs, initializer=_init, initargs=(a.song_dir, W, H)) as pool:
        res = pool.map(_render_chunk, chunks, chunksize=1)
    segs = [r[0] for r in res]
    with open(os.path.join(out, "frames.md5"), "w") as fh:
        fh.writelines(line + "\n" for r in res for line in r[1])
    dt = time.time() - t0
    print(f"rendered {f1 - f0} frames in {dt:.1f}s ({dt / (f1 - f0) * a.jobs:.3f} s/frame/worker)")

    lst = os.path.join(work, "segments.txt")
    with open(lst, "w") as fh:
        fh.writelines(f"file '{os.path.abspath(s)}'\n" for s in segs)
    master = find_master(a.song_dir)
    final = os.path.join(out, f"video-{H}p.mp4")
    ss = ["-ss", f"{f0 / fps:.6f}", "-t", f"{(f1 - f0) / fps:.6f}"]
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                    *ss, "-i", master, "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-ar", "48000",
                    "-shortest", "-movflags", "+faststart", final], check=True)
    for s in segs:
        os.remove(s)
    print("wrote", final)


if __name__ == "__main__":
    main()
