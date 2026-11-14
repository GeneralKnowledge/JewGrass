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
JOB = "render"            # "render" (album takes), "voice" (train the RVC voice model), or "both" (render first)
BRANCH = "claude/project-thread-l3ovu9"   # which branch of the repo to read tracks.yaml and these scripts from

# Render
TRACKS = "queue"          # "queue" = every track marked generate, strongest first; or a list: ["brass-plate", "gavs-song"]
TAKES = 2                 # takes per track (different seeds)
FIRST_SEED = 8001         # takes use FIRST_SEED, FIRST_SEED+1, ...
SEEDS = {}                # exact seeds for a track instead, e.g. {"gavs-song": [7202, 7203, 7204]}
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
        p = subprocess.Popen(cmd, cwd=cwd, env=dict(os.environ, PYTHONUNBUFFERED="1", **env),
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        threading.Thread(target=pump, args=(p, f"[{i}]"), daemon=True).start()
        procs.append(p)
    return [p.wait() for p in procs]"""),
    (CODE, """# --- install (about 5-10 minutes) -------------------------------------------------------------------------------
parts = (["ace"] if JOB in ("render", "both") else []) + (["applio"] if JOB in ("voice", "both") else [])
subprocess.run(["bash", f"{REPO}/kaggle/setup.sh", *parts], check=True)
print(f"installed in {(time.time() - T0) / 60:.0f} min")"""),
    (CODE, """# --- render album takes, one worker per GPU ----------------------------------------------------------------------
if JOB in ("render", "both"):
    sys.path.insert(0, f"{REPO}/kaggle")
    import yaml, plan
    doc = yaml.safe_load(open(f"{REPO}/docs/album/tracks.yaml"))
    jobs, notes = plan.plan(doc, TRACKS, TAKES, FIRST_SEED, SEEDS, gpus=N_GPU)
    for n in notes:
        print("NOTE", n)
    json.dump(jobs, open("/kaggle/temp/jobs.json", "w"), indent=1)
    print(f"{len(jobs)} takes, {sum(j['seconds'] for j in jobs) / 60:.0f} min of music, over {N_GPU} GPU(s)")
    deadline = min(time.time() + RENDER_MAX_HOURS * 3600, HARD_STOP)
    ace = "/kaggle/temp/ACE-Step-1.5"
    codes = run_all([[f"{ace}/.venv/bin/python", f"{REPO}/kaggle/render.py", "/kaggle/temp/jobs.json", str(g),
                      f"{OUT}/renders"] for g in range(N_GPU)],
                    [{"CUDA_VISIBLE_DEVICES": str(g), "BAND_DEADLINE": str(deadline)} for g in range(N_GPU)], cwd=ace)
    print("render workers exited with", codes, f"after {(time.time() - T0) / 60:.0f} min in total")"""),
    (CODE, """# --- train the voice model (training on the first GPU, probe clips on the second) -------------------------------
if JOB in ("voice", "both"):
    deadline = min(time.time() + VOICE_MAX_HOURS * 3600, HARD_STOP)
    env = {"VOICE_MODEL": VOICE_MODEL, "VOICE_EPOCHS": str(VOICE_EPOCHS), "VOICE_SAVE_EVERY": str(VOICE_SAVE_EVERY),
           "VOICE_PITCH": str(VOICE_PITCH), "BAND_DEADLINE": str(deadline), "OUT_DIR": OUT,
           "CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in range(N_GPU))}
    codes = run_all([[sys.executable, f"{REPO}/kaggle/voice.py"]], [env])
    print("voice exited with", codes, f"after {(time.time() - T0) / 60:.0f} min in total")

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
         f"{(time.time() - T0) / 3600:.1f} GPU hours, {gpus[0]} x{N_GPU}", f"settings: JOB={JOB} TRACKS={TRACKS} "
         f"TAKES={TAKES} FIRST_SEED={FIRST_SEED} SEEDS={SEEDS} VOICE_MODEL={VOICE_MODEL} VOICE_EPOCHS={VOICE_EPOCHS}", ""]
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
