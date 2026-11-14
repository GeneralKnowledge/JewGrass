"""Write kaggle/jewgrass-kaggle.ipynb from the cells below. Edit here, then run: python kaggle/build_notebook.py"""
import json
import os

MD, CODE = "markdown", "code"
cells = [
    (MD, """# JewGrass heavy jobs (Kaggle GPU)

Renders ACE-Step takes for the album and trains the RVC voice model, on Kaggle's GPUs instead of the laptop.

**To run:** set the options in the next cell, then **Save Version → Save & Run All (Commit)**. It runs in the
background (you can close the tab) and the results appear under **Output** when it finishes. Full instructions:
`kaggle/README.md` in the JewGrass repo.

Needs: **Accelerator = GPU T4 x2** and **Internet = on** (Settings panel on the right; internet needs a
phone-verified Kaggle account). For voice training, attach the `jewgrass-voice` dataset with **Add Input**."""),
    (CODE, """# --- settings -------------------------------------------------------------------------------------------------
JOB = "render"            # "render" (album takes), "voice" (train the RVC voice model), or "both"
                          # (both: with two cards, voice trains on card 0 while takes render on card 1)
BRANCH = "claude/project-thread-l3ovu9"   # which branch of the repo to read tracks.yaml and these scripts from

# Render
TRACKS_YAML = "docs/album/tracks.yaml"   # track list: a path in the repo, or an https:// link to one
TRACKS = "queue"          # "queue" = every track marked generate, strongest first; or a list: ["brass-plate", "gavs-song"]
TAKES = 2                 # takes per track (different seeds)
FIRST_SEED = 8001         # takes use FIRST_SEED, FIRST_SEED+1, ...
SEEDS = {}                # exact seeds for a track instead, e.g. {"gavs-song": [7202, 7203, 7204]}
DIT_MODELS = ["turbo"]    # music model(s), every take is made with each: "turbo" (the laptop's 2B), "xl-turbo"
                          # (4B, about twice the time per take), "sft"/"xl-sft" (50 steps, roughly 10x turbo's time)
RENDER_FIRST = []         # batches rendered before the one above, same keys in lower case, e.g.
                          # [{"tracks_yaml": "https://...", "tracks": "queue", "seeds": {...}, "models": ["turbo", "xl-turbo"]}]
RENDER_MAX_HOURS = 6      # stop starting new takes after this long

# Voice
VOICE_MODEL = "reuben"    # model name; reuse it to carry on training an earlier run
VOICE_EPOCHS = 250        # train up to this epoch in total
VOICE_SAVE_EVERY = 25     # save + convert the Swim probe every N epochs
VOICE_PITCH = 0           # semitones to shift the probe vocal when converting
VOICE_MAX_HOURS = 6       # stop training after this long (it resumes next run)

KAGGLE_USER = ""          # your Kaggle username: lets the notebook save the voice training state for next time

SETTINGS_JSON = {}        # the laptop's kjob.py run fills this in; it overrides the values above
globals().update(SETTINGS_JSON)"""),
    (CODE, """# --- check the machine ------------------------------------------------------------------------------------------
import json, os, subprocess, sys, threading, time, urllib.request
T0 = time.time()
# Kaggle's Jupyter sets MPLBACKEND to its inline backend, which matplotlib rejects outside the notebook process;
# ACE-Step (via lightning/torchmetrics) and Applio import matplotlib, so every worker gets a plain backend.
os.environ["MPLBACKEND"] = "Agg"
HARD_STOP = T0 + 11.5 * 3600          # Kaggle stops GPU sessions at 12 h; leave time to save the output
OUT = "/kaggle/working/out"
os.makedirs(OUT, exist_ok=True)
gpus = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                      capture_output=True, text=True).stdout.strip().splitlines()
assert gpus, "No GPU. Settings → Accelerator → GPU T4 x2, then run again."
print("GPUs:", gpus)
try:
    urllib.request.urlopen("https://github.com", timeout=10)
except Exception as e:
    raise SystemExit("No internet. Settings → Internet on (needs a phone-verified Kaggle account). " + repr(e))
N_GPU = len(gpus)
print(f"JOB={JOB}  TRACKS={TRACKS}  TAKES={TAKES}  VOICE_MODEL={VOICE_MODEL}  VOICE_EPOCHS={VOICE_EPOCHS}")"""),
    (CODE, """# --- get the band's scripts and tracks.yaml from GitHub (small: skips the media files) -----------------------------
REPO = "/kaggle/temp/JewGrass"
if not os.path.isdir(REPO):
    subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse", "--branch", BRANCH,
                    "https://github.com/GeneralKnowledge/JewGrass", REPO], check=True)
    subprocess.run(["git", "-C", REPO, "sparse-checkout", "set", "kaggle", "docs/album"], check=True)
print(subprocess.run(["git", "-C", REPO, "log", "-1", "--format=%h %s"], capture_output=True, text=True).stdout)


def run_all(cmds, envs, cwd=None):
    \"\"\"Run commands side by side, printing their output as it comes. Returns the exit codes.\"\"\"
    def pump(p, tag):
        for line in p.stdout:
            print(f"{tag} {line}", end="", flush=True)
    procs = []
    for i, (cmd, env) in enumerate(zip(cmds, envs)):
        p = subprocess.Popen(cmd, cwd=cwd, env=dict(os.environ, PYTHONUNBUFFERED="1", MPLBACKEND="Agg", **env),
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        threading.Thread(target=pump, args=(p, f"[{i}]"), daemon=True).start()
        procs.append(p)
    return [p.wait() for p in procs]"""),
    (CODE, """# --- install (about 5-10 minutes) -------------------------------------------------------------------------------
parts = (["ace"] if JOB in ("render", "both") else []) + (["applio"] if JOB in ("voice", "both") else [])
subprocess.run(["bash", f"{REPO}/kaggle/setup.sh", *parts], check=True)
print(f"installed in {(time.time() - T0) / 60:.0f} min")"""),
    (CODE, """# --- the work: voice training and/or renders ---------------------------------------------------------------------
sys.path.insert(0, f"{REPO}/kaggle")
import yaml, plan
ace = "/kaggle/temp/ACE-Step-1.5"
CKPT = {"turbo": "acestep-v15-turbo", "xl-turbo": "acestep-v15-xl-turbo", "sft": "acestep-v15-sft", "xl-sft": "acestep-v15-xl-sft"}


def load_tracks(src):
    if src.startswith("https://"):
        return yaml.safe_load(urllib.request.urlopen(src, timeout=60).read())
    if os.path.exists(f"{REPO}/{src}"):
        return yaml.safe_load(open(f"{REPO}/{src}"))
    url = f"https://raw.githubusercontent.com/GeneralKnowledge/JewGrass/{BRANCH}/{src}"  # outside the sparse checkout
    return yaml.safe_load(urllib.request.urlopen(url, timeout=60).read())


voice_on = JOB in ("voice", "both")
render_on = JOB in ("render", "both")
voice_gpus = [0] if (voice_on and render_on and N_GPU > 1) else list(range(N_GPU))
render_gpus = [g for g in range(N_GPU) if g not in voice_gpus] if (voice_on and render_on and N_GPU > 1) else list(range(N_GPU))
cmds, envs, names = [], [], []

if render_on:
    batches = list(RENDER_FIRST) + [{"tracks_yaml": TRACKS_YAML, "tracks": TRACKS, "takes": TAKES,
                                     "first_seed": FIRST_SEED, "seeds": SEEDS, "models": DIT_MODELS}]
    jobs = []
    for bi, b in enumerate(batches):
        bj, notes = plan.plan(load_tracks(b.get("tracks_yaml", TRACKS_YAML)), b.get("tracks", "queue"), b.get("takes", TAKES),
                              b.get("first_seed", FIRST_SEED), b.get("seeds", {}), models=b.get("models", DIT_MODELS))
        for n in notes:
            print("NOTE", n)
        for j in bj:
            j["batch"] = bi
        print(f"batch {bi} ({b.get('tracks_yaml', TRACKS_YAML)}): {len(bj)} takes")
        jobs += bj
    plan.assign_gpus(jobs, render_gpus)
    json.dump(jobs, open("/kaggle/temp/jobs.json", "w"), indent=1)
    models = list(dict.fromkeys(j["model"] for j in jobs))
    for m in models:  # download each model once, before the workers start
        subprocess.run([f"{ace}/.venv/bin/python", "-c", f"from acestep.model_downloader import ensure_dit_model as e; print(e({CKPT[m]!r}))"],
                       cwd=ace, check=True)
    print(f"{len(jobs)} takes, {sum(j['seconds'] for j in jobs) / 60:.0f} min of music, on card(s) {render_gpus}")
    deadline = min(time.time() + RENDER_MAX_HOURS * 3600, HARD_STOP)
    for g in render_gpus:  # one worker per card; a card with several models runs them one after another
        # batch by batch (so RENDER_FIRST really comes first), and within a batch one model load per model
        mine = list(dict.fromkeys((j["batch"], j["model"]) for j in jobs if j["gpu"] == g))
        chain = " ; ".join(f"{ace}/.venv/bin/python {REPO}/kaggle/render.py /kaggle/temp/jobs.json {g} {OUT}/renders {m} {b}"
                           for b, m in mine)
        cmds.append(["bash", "-c", f"cd {ace} && {chain}"])
        envs.append({"CUDA_VISIBLE_DEVICES": str(g), "BAND_DEADLINE": str(deadline)})
        names.append(f"render card {g}")

if voice_on:
    deadline = min(time.time() + VOICE_MAX_HOURS * 3600, HARD_STOP)
    cmds.append([sys.executable, f"{REPO}/kaggle/voice.py"])
    envs.append({"VOICE_MODEL": VOICE_MODEL, "VOICE_EPOCHS": str(VOICE_EPOCHS), "VOICE_SAVE_EVERY": str(VOICE_SAVE_EVERY),
                 "VOICE_PITCH": str(VOICE_PITCH), "BAND_DEADLINE": str(deadline), "OUT_DIR": OUT,
                 "CUDA_VISIBLE_DEVICES": ",".join(map(str, voice_gpus))})
    names.append(f"voice card(s) {voice_gpus}")

if voice_on and render_on and N_GPU == 1:   # one card: voice first, then renders
    codes = run_all(cmds[-1:], envs[-1:]) + run_all(cmds[:-1], envs[:-1])
    codes = codes[1:] + codes[:1]
else:
    print("running side by side:", names)
    codes = run_all(cmds, envs)
for n, c in zip(names, codes):
    print(f"{n}: exit {c}")
print(f"work done after {(time.time() - T0) / 60:.0f} min in total")

if voice_on:
    # Keep the training state on Kaggle as the private dataset <user>/jewgrass-voice-state, so the next run
    # carries on from here without anything going through home internet. Needs the KAGGLE_API_TOKEN secret.
    tar = f"{OUT}/voice/state/{VOICE_MODEL}-state.tar"
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("KAGGLE_API_TOKEN")
    except Exception:
        token = None
    if os.path.exists(tar) and token and KAGGLE_USER:
        up = "/kaggle/temp/state-upload"
        os.makedirs(up, exist_ok=True)
        os.replace(tar, f"{up}/{VOICE_MODEL}-state.tar")
        ref = f"{KAGGLE_USER}/jewgrass-voice-state"
        json.dump({"title": "jewgrass-voice-state", "id": ref, "licenses": [{"name": "CC0-1.0"}]},
                  open(f"{up}/dataset-metadata.json", "w"))
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=True)
        kenv = dict(os.environ, KAGGLE_API_TOKEN=token)
        r = subprocess.run(["kaggle", "datasets", "version", "-p", up, "-m", f"{VOICE_MODEL} state", "-d"], env=kenv)
        if r.returncode:
            r = subprocess.run(["kaggle", "datasets", "create", "-p", up], env=kenv)
        print("training state saved to", ref if r.returncode == 0 else "NOWHERE (upload failed)")
        if r.returncode:
            os.replace(f"{up}/{VOICE_MODEL}-state.tar", tar)  # keep it in the output instead
    elif os.path.exists(tar):
        print("Training state is in the output (out/voice/state). To carry on next run without downloading it, "
              "add the KAGGLE_API_TOKEN secret and set KAGGLE_USER (see kaggle/README.md).")"""),
    (CODE, """# --- summary ----------------------------------------------------------------------------------------------------
lines = [f"JewGrass Kaggle run, {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(T0))}, "
         f"{(time.time() - T0) / 3600:.1f} GPU hours, {gpus[0]} x{N_GPU}", f"settings: JOB={JOB} TRACKS_YAML={TRACKS_YAML} TRACKS={TRACKS} "
         f"TAKES={TAKES} FIRST_SEED={FIRST_SEED} SEEDS={SEEDS} DIT_MODELS={DIT_MODELS} RENDER_FIRST={RENDER_FIRST} VOICE_MODEL={VOICE_MODEL} VOICE_EPOCHS={VOICE_EPOCHS}", ""]
for root, _, files in sorted(os.walk(OUT)):
    for f in sorted(files):
        p = os.path.join(root, f)
        lines.append(f"{os.path.relpath(p, OUT):<60} {os.path.getsize(p) / 2**20:8.1f} MB")
open(f"{OUT}/RUN.txt", "w").write("\\n".join(lines) + "\\n")
print("\\n".join(lines))"""),
]

nb = {
    "cells": [
        {"cell_type": t, "metadata": {}, "source": src.splitlines(keepends=True),
         **({"outputs": [], "execution_count": None} if t == CODE else {})}
        for t, src in cells
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jewgrass-kaggle.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out)
