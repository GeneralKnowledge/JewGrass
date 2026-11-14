#!/usr/bin/env python3
"""Run JewGrass heavy jobs on Kaggle from the laptop, and bring the results home.

One-time setup (see kaggle/README.md): pip install --user kaggle; put the API token in ~/.kaggle/access_token;
write your Kaggle username in ~/band/kaggle/user.

  kjob.py upload-voice [--clips DIR] [--probe-vocal F] [--probe-inst F] [--laptop-state MODEL]
        Upload the cleaned voice clips (and the Swim probe) as the private dataset <user>/jewgrass-voice.
        --laptop-state MODEL also uploads the laptop's training so far, so Kaggle carries on from it.
  kjob.py run KEY=VALUE ...      e.g.  kjob.py run JOB=render TRACKS='["brass-plate","gavs-song"]' TAKES=3
                                       kjob.py run JOB=voice VOICE_EPOCHS=300
        Push the notebook with these settings and start it on Kaggle (GPU T4 x2, internet on).
      If a run is still going, it waits for it to finish and fetches its results first: one run at a time.
  kjob.py status                 Is the latest run still going?
  kjob.py log [VERSION]          The end of a run's log (why did it stop?)
  kjob.py fetch [--wait]         Download the latest run's output to ~/band/kaggle/runs/<time>/
                                 (--wait: check every 5 minutes until it finishes, then download).
  kjob.py quota                  GPU hours left this week.

Light on the laptop: it only uploads, polls and downloads. No GPU, no lock needed.
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

BAND = os.path.expanduser("~/band")
HOME = os.path.join(BAND, "kaggle")
HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK = os.path.join(HERE, "..", "jewgrass-kaggle.ipynb")
SLUG = "jewgrass-heavy-jobs"
AUDIO = (".wav", ".flac")


def user():
    f = os.path.join(HOME, "user")
    if not os.path.exists(f):
        sys.exit(f"Write your Kaggle username in {f} first (echo yourname > {f}).")
    return open(f).read().strip()


def kaggle(*args, check=True, capture=False):
    exe = shutil.which("kaggle") or os.path.expanduser("~/.local/bin/kaggle")
    r = subprocess.run([exe, *args], text=True, capture_output=capture)
    if check and r.returncode:
        sys.exit(f"kaggle {' '.join(args[:2])} failed:\n{(r.stdout or '') + (r.stderr or '')}")
    return r


def dataset_exists(ref):
    return kaggle("datasets", "status", ref, check=False, capture=True).returncode == 0


def push_dataset(folder, ref, title, note):
    json.dump({"title": title, "id": ref, "licenses": [{"name": "CC0-1.0"}]},
              open(os.path.join(folder, "dataset-metadata.json"), "w"))
    if dataset_exists(ref):
        kaggle("datasets", "version", "-p", folder, "-m", note, "--delete-old-versions")
    else:
        kaggle("datasets", "create", "-p", folder)  # private by default
    print(f"uploaded {ref}")


def upload_voice(a):
    u = user()
    clips = os.path.expanduser(a.clips)
    files = sorted(f for f in glob.glob(os.path.join(clips, "*")) if f.lower().endswith(AUDIO))
    if len(files) < 5:
        sys.exit(f"Only {len(files)} clips in {clips}; expected the cleaned dataset (8 s clips).")
    with tempfile.TemporaryDirectory(dir=HOME) as tmp:
        for f in files:
            shutil.copy2(f, tmp)
        # The Swim line the voice thread converts at each checkpoint (swim-probe*.wav somewhere under ~/band).
        for src, name, laptop_name in ((a.probe_vocal, "probe-vocal.wav", "swim-probe.wav"),
                                       (a.probe_inst, "probe-inst.wav", "swim-probe-inst.wav")):
            src = src or next(iter(sorted(glob.glob(os.path.join(BAND, "**", laptop_name), recursive=True))), None)
            if src and os.path.exists(src):
                shutil.copy2(src, os.path.join(tmp, name))
                print(f"probe: {src} -> {name}")
        print(f"{len(files)} clips from {clips}")
        push_dataset(tmp, f"{u}/jewgrass-voice", "jewgrass-voice", "voice clips")
    if a.laptop_state:
        upload_laptop_state(u, a.laptop_state)


def upload_laptop_state(u, model):
    logs = os.path.join(BAND, "tools/rvc/Applio/logs", model)
    gs = sorted(glob.glob(f"{logs}/G_*.pth"), key=os.path.getmtime)
    ds = sorted(glob.glob(f"{logs}/D_*.pth"), key=os.path.getmtime)
    if not gs or not ds:
        sys.exit(f"No G_/D_ checkpoints in {logs}")
    weights = sorted(glob.glob(f"{logs}/{model}_*e_*s.pth"), key=os.path.getmtime)
    # Only the newest G_/D_ and the newest small weight file (it tells Kaggle which epoch this is); the rest
    # of the folder (sliced audio, pitch, features, config) comes as it is, so Kaggle skips preprocessing.
    keep = {gs[-1], ds[-1]} | set(weights[-1:])
    drop = re.compile(r"^[GD]_\d+\.pth$|_\d+e_\d+s\.pth$")

    def only_latest(ti):
        path = os.path.join(os.path.dirname(logs), ti.name)
        return None if drop.search(os.path.basename(ti.name)) and path not in keep else ti

    with tempfile.TemporaryDirectory(dir=HOME) as tmp:
        tar = os.path.join(tmp, f"{model}-state.tar")
        with tarfile.open(tar, "w") as t:
            t.add(logs, arcname=model, filter=only_latest)
        print(f"laptop training state: {os.path.getsize(tar) / 2**20:.0f} MB (latest {os.path.basename(gs[-1])})")
        push_dataset(tmp, f"{u}/jewgrass-voice-state", "jewgrass-voice-state", f"laptop state {model}")


def parse_value(v):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def run(a):
    u = user()
    settings = {}
    for kv in a.settings:
        k, _, v = kv.partition("=")
        settings[k] = parse_value(v)
    settings.setdefault("JOB", "render")
    settings["KAGGLE_USER"] = u
    nb = json.load(open(NOTEBOOK))
    hits = 0
    for c in nb["cells"]:
        src = "".join(c["source"])
        if "SETTINGS_JSON = {}" in src:
            c["source"] = src.replace("SETTINGS_JSON = {}", "SETTINGS_JSON = " + json.dumps(settings)).splitlines(True)
            hits += 1
    assert hits == 1, "settings cell not found in the notebook"
    sources = []
    if settings["JOB"] in ("voice", "both"):
        for name in ("jewgrass-voice", "jewgrass-voice-state"):
            if dataset_exists(f"{u}/{name}"):
                sources.append(f"{u}/{name}")
        if f"{u}/jewgrass-voice" not in sources:
            sys.exit("Upload the voice clips first: kjob.py upload-voice")
    # One run at a time: Kaggle starts a pushed version even while an earlier one is still going, and both
    # then compete for the week's GPU hours. So wait for the current run, bring its results home, then push.
    if not a.now and busy():
        print("A run is already going on Kaggle; this one waits its turn (checking every 5 minutes).")
        while busy():
            time.sleep(300)
        fetch(argparse.Namespace(wait=False))
    push = os.path.join(HOME, "push")
    shutil.rmtree(push, ignore_errors=True)
    os.makedirs(push)
    json.dump(nb, open(os.path.join(push, f"{SLUG}.ipynb"), "w"), indent=1)
    json.dump({"id": f"{u}/{SLUG}", "title": SLUG, "code_file": f"{SLUG}.ipynb", "language": "python",
               "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
               "machine_shape": a.accelerator, "dataset_sources": sources, "competition_sources": [],
               "kernel_sources": [], "model_sources": []},
              open(os.path.join(push, "kernel-metadata.json"), "w"), indent=1)
    print("settings:", json.dumps(settings))
    print("inputs:", sources or "none")
    kaggle("kernels", "push", "-p", push, "--accelerator", a.accelerator)
    print(f"Started. Watch it at https://www.kaggle.com/code/{u}/{SLUG} or run: kjob.py fetch --wait")


def status(version=""):
    r = kaggle("kernels", "status", f"{user()}/{SLUG}" + (f"/{version}" if version else ""), check=False, capture=True)
    print((r.stdout + r.stderr).strip())
    return (r.stdout + r.stderr).lower()


def busy():
    """True while the latest version is queued or running (False if the notebook doesn't exist yet)."""
    return bool(re.search(r"running|queued|pending", status()))


def show_log(a):
    """Download a version's run log and print its end: the first place to look when a run fails."""
    ref = f"{user()}/{SLUG}" + (f"/{a.version}" if a.version else "")
    status(a.version)
    dest = os.path.join(HOME, "logs", a.version or "latest")
    os.makedirs(dest, exist_ok=True)
    kaggle("kernels", "output", ref, "-p", dest, "--file-pattern", r"\.log$")
    for f in glob.glob(f"{dest}/*.log"):
        text = open(f, errors="replace").read()
        try:  # Kaggle logs are a JSON list of {"stream_name", "time", "data"}
            text = "".join(e.get("data", "") for e in json.loads(text))
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
        lines = text.splitlines()
        print(f"--- {f}: last {min(a.lines, len(lines))} of {len(lines)} lines")
        print("\n".join(re.sub(r"KGAT_[0-9a-fA-F]+", "KGAT_***", line) for line in lines[-a.lines:]))


def fetch(a):
    while a.wait:
        s = status()
        if not re.search(r"running|queued|pending", s):
            break
        time.sleep(300)
    stamp = time.strftime("%Y%m%d-%H%M")
    dest = os.path.join(HOME, "runs", stamp)
    os.makedirs(dest, exist_ok=True)
    # Everything except the training state, which stays on Kaggle (it's large; the notebook keeps it there).
    kaggle("kernels", "output", f"{user()}/{SLUG}", "-p", dest, "--file-pattern", r"\A(?!.*state/)")
    n = sum(len(f) for _, _, f in os.walk(dest))
    print(f"{n} files in {dest}")
    run_txt = glob.glob(f"{dest}/**/RUN.txt", recursive=True)
    if run_txt:
        print(open(run_txt[0]).read())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("upload-voice")
    p.add_argument("--clips", default=os.path.join(BAND, "voice/dataset"))
    p.add_argument("--probe-vocal")
    p.add_argument("--probe-inst")
    p.add_argument("--laptop-state", metavar="MODEL", help="also upload the laptop's Applio/logs/MODEL to resume from")
    p = sub.add_parser("run")
    p.add_argument("settings", nargs="*")
    p.add_argument("--accelerator", default="NvidiaTeslaT4")
    p.add_argument("--now", action="store_true", help="push even if a run is still going (it will run alongside)")
    sub.add_parser("status")
    p = sub.add_parser("log", help="print the end of a run's log")
    p.add_argument("version", nargs="?", default="", help="version number (default: latest)")
    p.add_argument("--lines", type=int, default=80)
    p = sub.add_parser("fetch")
    p.add_argument("--wait", action="store_true")
    sub.add_parser("quota")
    a = ap.parse_args()
    os.makedirs(HOME, exist_ok=True)
    if a.cmd == "upload-voice":
        upload_voice(a)
    elif a.cmd == "run":
        run(a)
    elif a.cmd == "status":
        status()
    elif a.cmd == "log":
        show_log(a)
    elif a.cmd == "fetch":
        fetch(a)
    elif a.cmd == "quota":
        kaggle("quota")


if __name__ == "__main__":
    main()
