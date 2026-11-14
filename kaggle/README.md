# Heavy jobs on Kaggle

Kaggle gives about 30 free GPU hours a week (two NVIDIA T4 cards, 16 GB each, per session). The laptop's
GTX 1060 has 6 GB, overheats and drops off the network, so the big batch jobs go to Kaggle instead:

- **Album takes** with ACE-Step 1.5: every `generate` track in [`docs/album/tracks.yaml`](../docs/album/tracks.yaml),
  several seeds each, one take per card at a time.
- **Voice model** training with Applio (RVC), with a Swim probe clip at every checkpoint.

The laptop stays the place for quick tests, repaints, mastering, stems and video.

| File | What it is |
|---|---|
| `jewgrass-kaggle.ipynb` | The notebook. Settings in the first code cell. Built by `build_notebook.py` (edit that, not the .ipynb) |
| `setup.sh` | Installs ACE-Step 1.5 and/or Applio on the Kaggle machine, same versions as the laptop (PyTorch 2.10, CUDA 12.6) |
| `plan.py` | tracks.yaml + settings → list of takes, split across the cards. Drops caption phrases that cause hiss |
| `render.py` | Renders takes on one card with the laptop's `gen.py` settings, loading the model once |
| `voice.py` | Trains or resumes the voice model, converts the probe at every checkpoint, saves the state |
| `laptop/kjob.py` | Run all of this from the laptop: upload voice clips, start a run, fetch the results |

## One-time setup (about 10 minutes, on your phone or any browser)

1. Sign in at [kaggle.com](https://www.kaggle.com). Go to **Settings → Phone verification** and verify your
   phone number. Without it, notebooks can't use the internet, and this one needs it to install the tools.
2. Still in **Settings → API**, click **Generate New Token** and copy it. This lets the laptop start runs and
   download results.
3. On the laptop (or ask me to do it through Remote Control):
   ```
   pip install --user kaggle
   mkdir -p ~/.kaggle ~/band/kaggle
   echo 'PASTE_THE_TOKEN' > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
   echo 'your-kaggle-username' > ~/band/kaggle/user
   ```
4. For voice training only: the first voice run creates the notebook on Kaggle. Then, once, open it on
   kaggle.com (**Your Work → jewgrass-heavy-jobs → Edit**), go to **Add-ons → Secrets**, add a secret named
   `KAGGLE_API_TOKEN` with the same token, and tick it for this notebook. That lets each run save its training
   state on Kaggle, so the next run carries on without anything going through your home internet.

## Running jobs from the laptop

`kaggle/laptop/kjob.py` (it only uploads, waits and downloads, so it doesn't load the laptop):

```
# Album takes: every generate track, 2 takes each (about 1 to 1.5 GPU hours)
python3 kaggle/laptop/kjob.py run JOB=render

# Just some tracks, 3 takes each, or exact seeds
python3 kaggle/laptop/kjob.py run JOB=render TRACKS='["brass-plate","gavs-song"]' TAKES=3
python3 kaggle/laptop/kjob.py run JOB=render TRACKS='["gavs-song"]' SEEDS='{"gavs-song":[7202,7210,7211]}'

# Voice: upload the cleaned clips (and the Swim probe) once, then train
python3 kaggle/laptop/kjob.py upload-voice                      # from ~/band/voice/dataset
python3 kaggle/laptop/kjob.py upload-voice --laptop-state reuben  # also carry on from the laptop's training
python3 kaggle/laptop/kjob.py run JOB=voice VOICE_EPOCHS=250

# Results
python3 kaggle/laptop/kjob.py status
python3 kaggle/laptop/kjob.py fetch --wait     # waits for the run to finish, downloads to ~/band/kaggle/runs/<time>/
python3 kaggle/laptop/kjob.py quota            # GPU hours left this week
```

More options:

```
# Voice and takes at once: voice trains on card 0 while takes render on card 1
python3 kaggle/laptop/kjob.py run JOB=both

# A better music model: XL turbo (4B instead of 2B, about twice the time per take). Both, for a blind A/B:
python3 kaggle/laptop/kjob.py run JOB=render TRACKS='["gavs-song"]' DIT_MODELS='["turbo","xl-turbo"]'

# A pilot batch first, then the album (RENDER_FIRST takes the same settings in lower case)
python3 kaggle/laptop/kjob.py run JOB=both RENDER_FIRST='[{"tracks_yaml":"https://.../lowbridge/tracks.yaml","models":["turbo","xl-turbo"]}]'
```

Every setting in the notebook's first cell can be passed as `NAME=value`. `TRACKS_YAML` picks another track
list (a repo path or an https link), e.g. `TRACKS_YAML=lowbridge/tracks.yaml BRANCH=<its branch>`.

## Running from the browser instead (no laptop needed)

1. kaggle.com → **Create → New Notebook → File → Import Notebook**, and give it this file's link on GitHub:
   `https://github.com/GeneralKnowledge/JewGrass/blob/claude/project-thread-l3ovu9/kaggle/jewgrass-kaggle.ipynb`
2. Right-hand panel → **Session options**: **Accelerator = GPU T4 x2**, **Internet = On**.
3. Change the settings in the first code cell if you want (`JOB`, `TRACKS`, `TAKES`...).
4. For voice: **Add Input → Upload** a folder of the cleaned clips (name the dataset `jewgrass-voice`;
   optionally add `probe-vocal.wav` and `probe-inst.wav`).
5. **Save Version → Save & Run All (Commit) → Save**. It runs in the background; you can close the tab.
6. When it's done, open the notebook's **Output** tab and download `out/` (or individual mp3s to listen to on
   your phone).

## What comes back

```
out/RUN.txt                                   settings, GPU hours used, list of files
out/renders/<slug>/<slug>-<seed>.mp3          to listen to
out/renders/<slug>/<slug>-<seed>.wav          for mastering / stems on the laptop
out/renders/<slug>/<slug>-<seed>.json         caption, lyrics and settings used, render time
out/voice/probes/e025-your-voice.mp3          converted Swim line per checkpoint (loudness-matched)
out/voice/probes/e025-mix.mp3                 the same over the Swim backing (if probe-inst.wav was uploaded)
out/voice/model/reuben_250e_<steps>s.pth      the voice model, plus its .index, for converting songs on the laptop
```

## How the time is spent (estimates until the first run reports real numbers)

- Install: about 5 to 10 minutes per run (ACE-Step and its model, ~10 GB of downloads at datacentre speed).
- Renders: the laptop takes 5 to 6 minutes per 4-minute song. A T4 in the same fp32 mode should be roughly
  twice as fast, and there are two of them, so about 1.5 minutes per take overall. All 10 tracks × 2 takes is
  about 67 minutes of music: roughly an hour of session time in total.
- Voice: the laptop does an epoch in about 33 s on the current clips; a T4 in fp16 should be several times
  faster, so 250 epochs fits comfortably in one run.
- Kaggle ends a session at 12 hours. `RENDER_MAX_HOURS` and `VOICE_MAX_HOURS` (6 each by default) stop the
  work earlier. RUN.txt lists what got done, so a rerun can ask for just the missing tracks; voice resumes
  from the saved state.
- `RUN.txt` records the real hours, so the estimates above can be corrected after the first run.

## Notes

- With the default `DIT_MODELS=["turbo"]` the sound matches the laptop: same model (turbo DiT, no language model), same fp32 compute and attention
  settings, same `shift`. A seed should sound like the same seed on the laptop, though the different card can
  make tiny differences.
- Caption phrases that caused hiss ("crackling gramophone", "typewriter clacks", and similar) are dropped
  automatically; the notebook prints which ones it dropped.
- The notebook reads `tracks.yaml` and the scripts from the branch in its `BRANCH` setting
  (default: the repo's default branch).
- Kaggle's 30 hours are per week, counted while a GPU session is open. Commit runs close themselves when
  they finish, so no hours are wasted on an idle notebook.
