# Take server

Lets the phone browse, play, compare and pick takes without downloading mp3s one by one. It runs on the
laptop; the player is the **Listen** tab in Soundscape (ixOS, inside Device Lab: Apps, then Soundscape).

```
python3 tools/player/serve.py              # serves ~/band on port 8642
python3 -m unittest discover tools/player  # tests
```

Standard library Python only, so nothing is installed. Then in Soundscape, Listen, set the server to
`http://<laptop's Tailscale address>:8642`.

## What it does

- **Finds takes** under `~/band/kaggle/runs`, `~/band/songs` and `~/band/outputs` (`--dirs`), skipping stems,
  analysis folders and venvs. When a take has an mp3 and a wav, the phone gets the mp3.
- **Groups them by song**: the slug in the take's `.json` sidecar (Kaggle renders), otherwise the song folder
  (`songs/<slug>/60-master/x.wav` belongs to `<slug>`), otherwise the file name without its `-seed` ending.
  The sidecar's title, seed, model, caption and lyrics go to the phone.
- **Joins old slugs and shows the act** using `catalog.json` (next to `serve.py`; copy it with the script).
  The baseline run renamed Natural Capital's songs (`swim-at-your-own-risk` became `nc-swim-at-your-own-risk`),
  so without it the album-xl takes, the baseline takes and the laptop's original Swim would be three songs; with
  it they are one, and A/B works across them. Titles read `AFTER HOURS · One More Night` (`--plain-titles` to
  drop the act). When a song has takes from more than one run, each label says its run. Picks saved under an
  old slug still show. Rebuild it after adding an act or song:
  `python3 tools/player/build_catalog.py baseline-run/tracks.yaml baseline-run/sources/*.yaml` (needs PyYAML).
- **Saves picks** to `~/band/picks.json`: for each song, the chosen take and each take's stars (0 to 5) and
  note. `pick` is the take's path under `~/band` without the extension, for example
  `kaggle/runs/20261115-2100/renders/gavs-song/gavs-song-7202`. Read this file to find out what Reuben picked.
- **Measures loudness** once per file with ffmpeg's `ebur128` (one file at a time, `nice 19`, idle I/O,
  cached in `~/band/.player/loudness.json`), so the phone can compare takes at matched loudness. While a
  heavy job holds `~/band/.gpu.lock`, it only measures the takes of the song open on the phone (a second or
  two each). `--no-loudness` turns this off.
- **Streams audio** with byte ranges (iOS needs them), and only files it listed.

## API

| Request | Answer |
|---|---|
| `GET /api/library` | `{"songs": [{slug, title, act, actTitle, takes, updated, pick, pickLabel, rated}]}`, newest first. `?fresh` rescans now (otherwise every 20 s) |
| `GET /api/song?slug=S` | `{slug, title, act, actTitle, pick, takes: [{id, file, label, run, mtime, seed, model, seconds, bpm, key, caption, lyrics, lufs, stars, note}]}` |
| `GET /audio/<file>` | the audio, with `Range` support |
| `POST /api/pick` | body `{song, take, pick?: true/false, stars?: 0-5, note?: "..."}`; returns the song's saved entry |

## Running it all the time (optional)

Copy `serve.py` to `~/band/tools/player/` like the other laptop scripts, then:

```
cp laptop/systemd/take-server.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now take-server
```

It listens on every interface by default; `--bind <tailscale address>` keeps it off the home network. There
is no password, so only run it on a network you trust (Tailscale is).
