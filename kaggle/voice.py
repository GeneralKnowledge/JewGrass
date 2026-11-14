"""Train (or keep training) the RVC voice model with Applio on Kaggle, and convert the Swim probe at every checkpoint.

Usage: python kaggle/voice.py   (settings come from environment variables, set by the notebook)
  VOICE_MODEL        model name, e.g. reuben (also the folder name under Applio/logs)
  VOICE_EPOCHS       train up to this epoch in total (a resumed model carries on from where it stopped)
  VOICE_SAVE_EVERY   save a checkpoint and convert the probe every N epochs (25)
  VOICE_BATCH        batch size (8 fits a 16 GB card easily)
  VOICE_PITCH        semitones to shift the probe when converting (0)
  VOICE_SR           sample rate, 48000 (must match the model if resuming)
  APPLIO_DIR, APPLIO_PY   the Applio checkout and its venv python (setup.sh makes them)
  OUT_DIR            where results go (/kaggle/working/out)
  BAND_DEADLINE      unix time: stop training here so the run ends inside Kaggle's 12 hours

Inputs are found under /kaggle/input (attach them with "Add Input"):
  - a dataset folder of cleaned clips (a folder named "dataset", or any folder with 5+ .wav/.flac files)
  - optional probe-vocal.wav / probe-inst.wav (the Swim line the laptop uses); otherwise the probe vocal
    is downloaded from the repo's voice-probe branch and only vocal-only clips are made
  - optional earlier state to resume: a folder or <model>-state.tar containing G_*.pth, D_*.pth
    (this notebook's own earlier output, or the laptop's Applio/logs/<model> folder)

Outputs in OUT_DIR/voice:
  probes/e025-your-voice.mp3, probes/e025-mix.mp3, ...   one pair per checkpoint
  model/<model>_<N>e_<steps>s.pth (+ .index at the end)  the voice model to use on the laptop
  state/<model>-state.tar                                attach this next time to resume training
"""
import glob
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request

MODEL = os.environ.get("VOICE_MODEL", "reuben")
EPOCHS = int(os.environ.get("VOICE_EPOCHS", "250"))
SAVE_EVERY = int(os.environ.get("VOICE_SAVE_EVERY", "25"))
BATCH = int(os.environ.get("VOICE_BATCH", "8"))
PITCH = int(os.environ.get("VOICE_PITCH", "0"))
SR = os.environ.get("VOICE_SR", "48000")
APPLIO = os.environ.get("APPLIO_DIR", "/kaggle/temp/Applio")
PY = os.environ.get("APPLIO_PY", "/kaggle/temp/venv-applio/bin/python")
OUT = os.path.join(os.environ.get("OUT_DIR", "/kaggle/working/out"), "voice")
DEADLINE = float(os.environ.get("BAND_DEADLINE", "0") or 0)
INPUT = os.environ.get("KAGGLE_INPUT", "/kaggle/input")
os.environ["MPLBACKEND"] = "Agg"  # Kaggle's inline backend breaks matplotlib in subprocesses
os.environ.setdefault("APPLIO_CONFIG_FILE", "/kaggle/temp/applio-config.json")  # written by setup.sh (fp16/fp32)
PROBE_URL = ("https://raw.githubusercontent.com/GeneralKnowledge/JewGrass/claude/voice-probe/"
             "voice-probe/00-original-ai-vocal.mp3")
AUDIO = (".wav", ".flac", ".mp3", ".ogg", ".m4a")

LOGS = os.path.join(APPLIO, "logs", MODEL)
GPUS = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if g != ""] or ["0"]
TRAIN_GPU, PROBE_GPU = GPUS[0], GPUS[-1]   # with two T4s, probes convert on the second card while training runs


def log(msg):
    print(f"[voice {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def applio(*args, gpu=TRAIN_GPU, check=True):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    cmd = [PY, "core.py", *map(str, args)]
    log("$ " + " ".join(cmd[1:]))
    r = subprocess.run(cmd, cwd=APPLIO, env=env)
    if check and r.returncode:
        sys.exit(f"Applio {args[0]} failed (exit {r.returncode})")
    return r.returncode


def find_dataset():
    named = [d for d in glob.glob(f"{INPUT}/**/dataset", recursive=True) if os.path.isdir(d)]
    cands = named or sorted({os.path.dirname(f) for f in glob.glob(f"{INPUT}/**/*", recursive=True)
                             if f.lower().endswith(AUDIO) and "probe" not in os.path.basename(f).lower()})
    for d in cands:
        n = len([f for f in os.listdir(d) if f.lower().endswith(AUDIO)])
        if n >= 5:
            return d, n
    return None, 0


def find_file(name):
    hits = glob.glob(f"{INPUT}/**/{name}", recursive=True)
    return hits[0] if hits else None


def restore_state():
    """Copy an earlier training state into Applio/logs/<model>. Returns the epoch it reached, or 0."""
    if glob.glob(f"{LOGS}/G_*.pth"):
        return 1  # already here (rerun inside the same session)
    tars = glob.glob(f"{INPUT}/**/{MODEL}-state.tar", recursive=True)
    dirs = sorted({os.path.dirname(p) for p in glob.glob(f"{INPUT}/**/G_*.pth", recursive=True)})
    dirs = [d for d in dirs if os.path.basename(d) == MODEL]  # never resume a different model
    if tars:
        log(f"resuming from {tars[0]}")
        os.makedirs(os.path.dirname(LOGS), exist_ok=True)
        with tarfile.open(tars[0]) as t:
            t.extractall(os.path.dirname(LOGS), filter="data")
    elif dirs:
        log(f"resuming from folder {dirs[0]}")
        shutil.copytree(dirs[0], LOGS, dirs_exist_ok=True)
    else:
        return 0
    weights = glob.glob(f"{LOGS}/{MODEL}_*e_*s.pth")
    return max([int(re.search(r"_(\d+)e_", w).group(1)) for w in weights] or [1])


def probe_inputs():
    os.makedirs(f"{OUT}/probes", exist_ok=True)
    vocal = find_file("probe-vocal.wav") or find_file("swim-probe.wav")
    inst = find_file("probe-inst.wav") or find_file("swim-probe-inst.wav")
    if not vocal:
        vocal = "/kaggle/temp/probe-vocal.mp3"
        if not os.path.exists(vocal):
            try:
                urllib.request.urlretrieve(PROBE_URL, vocal)
            except Exception as e:  # probes are a nice-to-have; training goes on without them
                log(f"no probe vocal ({e}); skipping probe clips")
                return None, None
    log(f"probe vocal: {vocal}; backing track: {inst or 'none (vocal-only clips)'}")
    return vocal, inst


def convert_probe(weight, vocal, inst):
    epoch = int(re.search(r"_(\d+)e_", weight).group(1))
    tag = f"e{epoch:03d}"
    if os.path.exists(f"{OUT}/probes/{tag}-your-voice.mp3"):
        return
    raw = f"/kaggle/temp/{tag}-raw.wav"
    rc = applio("infer", "--input-path", vocal, "--output-path", raw, "--pth-path", weight, "--index-path", "",
                "--pitch", PITCH, "--f0-method", "rmvpe", "--protect", "0.33", "--export-format", "WAV",
                gpu=PROBE_GPU, check=False)
    if rc or not os.path.exists(raw):
        log(f"probe conversion failed for {tag}")
        return
    # Loudness-match every vocal-only clip so they can be compared fairly.
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", raw, "-af", "loudnorm=I=-20:TP=-1.5",
                    "-ar", "48000", "-b:a", "320k", f"{OUT}/probes/{tag}-your-voice.mp3"])
    if inst:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", inst, "-i", raw, "-filter_complex",
                        "[0:a][1:a]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95",
                        "-ar", "48000", "-b:a", "320k", f"{OUT}/probes/{tag}-mix.mp3"])
    os.remove(raw)
    log(f"probe clip ready: probes/{tag}-your-voice.mp3")


def probe_watcher(stop, vocal, inst):
    while True:
        for w in sorted(glob.glob(f"{LOGS}/{MODEL}_*e_*s.pth"), key=os.path.getmtime):
            if time.time() - os.path.getmtime(w) > 5:  # let the file finish writing
                convert_probe(w, vocal, inst)
        if stop.is_set():
            return
        stop.wait(30)


def package():
    os.makedirs(f"{OUT}/model", exist_ok=True)
    os.makedirs(f"{OUT}/state", exist_ok=True)
    for f in glob.glob(f"{LOGS}/{MODEL}_*e_*s.pth") + glob.glob(f"{LOGS}/*.index"):
        shutil.copy2(f, f"{OUT}/model/")
    # The state to resume from: latest G/D, the preprocessed and extracted data, config. No per-epoch weights.
    tar = f"{OUT}/state/{MODEL}-state.tar"
    with tarfile.open(tar, "w") as t:
        t.add(LOGS, arcname=MODEL, filter=lambda ti: None if re.search(rf"/{MODEL}_\d+e_\d+s\.pth$", ti.name) else ti)
    log(f"saved {tar} ({os.path.getsize(tar) / 2**20:.0f} MB) and {len(os.listdir(f'{OUT}/model'))} model files")


def main():
    os.makedirs(OUT, exist_ok=True)
    start_epoch = restore_state()
    if not start_epoch:
        ds, n = find_dataset()
        if not ds:
            sys.exit("No voice dataset found under /kaggle/input. Attach the jewgrass-voice dataset (see kaggle/README.md).")
        log(f"new model {MODEL!r} from {n} clips in {ds}")
        applio("preprocess", "--model-name", MODEL, "--dataset-path", ds, "--sample-rate", SR,
               "--cpu-cores", os.cpu_count() or 4, "--cut-preprocess", "Automatic")
        applio("extract", "--model-name", MODEL, "--f0-method", "rmvpe", "--sample-rate", SR,
               "--embedder-model", "contentvec", "--cpu-cores", os.cpu_count() or 4, "--gpu", "0")
    else:
        log(f"resuming {MODEL!r}, last saved weights at epoch {start_epoch}")

    vocal, inst = probe_inputs()
    stop = threading.Event()
    watcher = None
    if vocal:
        watcher = threading.Thread(target=probe_watcher, args=(stop, vocal, inst), daemon=True)
        watcher.start()

    cmd = [PY, "core.py", "train", "--model-name", MODEL, "--save-every-epoch", SAVE_EVERY, "--save-only-latest",
           "--total-epoch", EPOCHS, "--sample-rate", SR, "--batch-size", BATCH, "--gpu", "0", "--vocoder", "HiFi-GAN"]
    log("$ " + " ".join(map(str, cmd[1:])))
    p = subprocess.Popen(list(map(str, cmd)), cwd=APPLIO, env=dict(os.environ, CUDA_VISIBLE_DEVICES=TRAIN_GPU))
    while p.poll() is None:
        if DEADLINE and time.time() > DEADLINE:
            log("time budget reached: stopping training (it resumes from the last saved checkpoint next run)")
            p.terminate()
            try:
                p.wait(120)
            except subprocess.TimeoutExpired:
                p.kill()
            break
        time.sleep(20)
    log(f"training ended (exit {p.returncode})")
    stop.set()
    if watcher:
        watcher.join()
        for w in sorted(glob.glob(f"{LOGS}/{MODEL}_*e_*s.pth")):  # catch the last checkpoint too
            convert_probe(w, vocal, inst)
    if not glob.glob(f"{LOGS}/*.index") and glob.glob(f"{LOGS}/{MODEL}_*e_*s.pth"):
        applio("index", "--model-name", MODEL, check=False)  # Applio only builds it when training completes
    package()


if __name__ == "__main__":
    main()
