#!/usr/bin/env python3
"""Take server: lets the phone list, stream, compare and pick generated takes.

Runs on the band laptop. The player is Soundscape's Listen tab (ixOS Device Lab), which talks to this
server over Tailscale:

    python3 tools/player/serve.py                # serves ~/band on port 8642
    python3 tools/player/serve.py --band ~/band --port 8642 --bind 100.x.y.z

Standard library only. It reads audio from the folders under ~/band that hold takes (kaggle/runs, songs,
outputs by default), groups them by song, and saves your picks, stars and notes to ~/band/picks.json, which
anyone (or any Claude thread) can read to learn which take you chose.

Loudness matching for A/B: each file's integrated loudness is measured once with ffmpeg (nice 19, idle I/O,
one file at a time) and cached in ~/band/.player/loudness.json. While a heavy job holds ~/band/.gpu.lock,
only the takes of the song you have open are measured.

GET /api/catalog answers with everything the phone's library needs at once: artists (acts), albums and
tracks with their takes. Artwork is optional, as JPEG or PNG under ~/band/artwork: <act>/artist.jpg,
<act>/album.jpg, <act>/<song slug>.jpg, or <song slug>.jpg for a song with no act.
"""

import argparse
import fcntl
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, unquote, urlparse

AUDIO_EXTS = [".mp3", ".m4a", ".flac", ".wav", ".ogg", ".opus"]  # preference order when one take has several
ART_EXTS = [".jpg", ".jpeg", ".png"]
DEFAULT_DIRS = ["kaggle/runs", "songs", "outputs"]
DEFAULT_SKIP = ["10-analysis", "20-stems", "__pycache__", "tmp", "models", "model"]
mimetypes.add_type("audio/mp4", ".m4a")
mimetypes.add_type("audio/flac", ".flac")
mimetypes.add_type("audio/ogg", ".opus")


# ---------------------------------------------------------------- library

def read_json(path):
    with open(path) as f:
        return json.load(f)


def song_key(rel_path, meta):
    """Which song a take belongs to: the sidecar's slug, else the folder (skipping NN-stage folders)."""
    if meta.get("slug"):
        return str(meta["slug"])
    parts = rel_path.split("/")
    folders = parts[:-1]
    while folders and re.match(r"^\d\d-", folders[-1]):
        folders.pop()
    if not folders or folders[-1] in ("outputs", "renders", "runs"):
        # a loose file such as outputs/swim-len240-1111.wav: drop trailing -number / -lenN pieces
        stem = os.path.splitext(parts[-1])[0]
        return re.sub(r"(-(len)?\d+)+$", "", stem) or stem
    return folders[-1]


def take_label(rel_path, meta):
    bits = []
    if meta.get("seed") is not None:
        bits.append(f"seed {meta['seed']}")
    if meta.get("model") and meta.get("model") != "turbo":
        bits.append(str(meta["model"]))
    if not bits:
        bits.append(os.path.splitext(os.path.basename(rel_path))[0])
    stage = next((p for p in rel_path.split("/")[:-1] if re.match(r"^\d\d-", p)), None)
    if stage and stage != "00-gen":
        bits.append(stage[3:])
    return " · ".join(bits)


def run_label(rel_path):
    """kaggle/runs/<time>/... -> <time>, so the same seed fetched twice stays apart."""
    m = re.match(r"^kaggle/runs/([^/]+)/", rel_path)
    return m.group(1) if m else ""


def load_catalog(path):
    """catalog.json (build_catalog.py): {"acts": {act: name}, "songs": {slug: {title, act, aliases}}}, or {}."""
    try:
        data = read_json(path)
        return data if isinstance(data, dict) and isinstance(data.get("songs"), dict) else {}
    except (OSError, ValueError):
        return {}


def catalog_index(catalog):
    """{slug or alias: (canonical slug, entry)} from a loaded catalog."""
    index = {}
    for slug, entry in (catalog or {}).get("songs", {}).items():
        index[slug] = (slug, entry)
        for alias in entry.get("aliases") or []:
            index.setdefault(alias, (slug, entry))
    return index


def scan(band, dirs, skip, catalog=None):
    """Return {song: {"title", "takes": [...]}} for every audio file under band/<dirs>.

    With a catalog, takes rendered under an older slug join their song, and each song gets its act and title."""
    skip = set(skip)
    index = catalog_index(catalog)
    acts = (catalog or {}).get("acts", {})
    found = {}  # rel path without extension -> {ext: full path}
    for d in dirs:
        top = os.path.join(band, d)
        if not os.path.isdir(top):
            continue
        for root, subdirs, files in os.walk(top, followlinks=False):
            subdirs[:] = sorted(s for s in subdirs if not s.startswith(".") and s not in skip
                                and not s.endswith(".venv"))
            for f in files:
                stem, ext = os.path.splitext(f)
                if ext.lower() in AUDIO_EXTS:
                    full = os.path.join(root, f)
                    rel = os.path.relpath(os.path.join(root, stem), band).replace(os.sep, "/")
                    found.setdefault(rel, {})[ext.lower()] = full
    songs = {}
    for rel, by_ext in found.items():
        ext = next(e for e in AUDIO_EXTS if e in by_ext)
        full = by_ext[ext]
        meta = {}
        side = os.path.join(band, rel + ".json")
        if os.path.isfile(side):
            try:
                meta = read_json(side)
                if not isinstance(meta, dict):
                    meta = {}
            except (OSError, ValueError):
                meta = {}
        st = os.stat(full)
        key = song_key(rel + ext, meta)
        key, entry = index.get(key, (key, None))
        song = songs.setdefault(key, {"slug": key, "title": key, "takes": []})
        if entry:
            song["title"] = entry.get("title") or song["title"]
            song["aliases"] = list(entry.get("aliases") or [])
            if entry.get("act"):
                song["act"] = entry["act"]
                song["actTitle"] = acts.get(entry["act"], entry["act"])
        elif meta.get("title") and song["title"] == key:
            song["title"] = str(meta["title"])
        song["takes"].append({
            "id": rel,
            "file": rel + ext,
            "label": take_label(rel + ext, meta),
            "run": run_label(rel),
            "mtime": int(st.st_mtime),
            "size": st.st_size,
            "seed": meta.get("seed"),
            "model": meta.get("model"),
            "seconds": meta.get("seconds"),
            "bpm": meta.get("bpm"),
            "key": meta.get("key"),
            "caption": meta.get("caption"),
            "lyrics": meta.get("lyrics"),
        })
    for song in songs.values():
        song["takes"].sort(key=lambda t: (t["mtime"], t["id"]))
        song["updated"] = max(t["mtime"] for t in song["takes"])
        # takes from more than one Kaggle run (or a run and the laptop): add the run to tell them apart
        runs = {t["run"] for t in song["takes"]}
        labels = [t["label"] for t in song["takes"]]
        for t in song["takes"]:
            if t["run"] and (len(runs) > 1 or labels.count(t["label"]) > 1):
                t["label"] += f" · run {t['run']}"
    return songs


# ---------------------------------------------------------------- picks

class Picks:
    """~/band/picks.json: {"songs": {slug: {"pick": take id or null, "takes": {id: {"stars", "note"}}}}}."""

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()

    def load(self):
        try:
            data = read_json(self.path)
            if isinstance(data, dict) and isinstance(data.get("songs"), dict):
                return data
        except (OSError, ValueError):
            pass
        return {"songs": {}}

    def update(self, slug, take, pick=None, stars=None, note=None):
        """pick: True makes this take the song's pick, False clears it if it was. Returns the song's entry."""
        if not slug or not take:
            raise ValueError("song and take are required")
        with self.lock:
            data = self.load()
            song = data["songs"].setdefault(slug, {"pick": None, "takes": {}})
            entry = song["takes"].setdefault(take, {})
            if pick is True:
                song["pick"] = take
            elif pick is False and song.get("pick") == take:
                song["pick"] = None
            if stars is not None:
                stars = int(stars)
                if not 0 <= stars <= 5:
                    raise ValueError("stars must be 0 to 5")
                entry["stars"] = stars
            if note is not None:
                entry["note"] = str(note)[:2000]
            entry["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            song["updated"] = entry["updated"]
            data["note"] = "Written by tools/player/serve.py. pick = the chosen take's path under ~/band, without extension."
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=1, sort_keys=True)
            os.replace(tmp, self.path)
            return song


# ---------------------------------------------------------------- loudness

LUFS_RE = re.compile(r"I:\s+(-?\d+(?:\.\d+)?|-inf)\s+LUFS")


def parse_ebur128(stderr_text):
    """Integrated loudness from ffmpeg's ebur128 summary, or None."""
    found = LUFS_RE.findall(stderr_text)
    if not found or found[-1] == "-inf":
        return None
    return float(found[-1])


class Loudness:
    def __init__(self, band, cache_path, enabled=True):
        self.band = band
        self.cache_path = cache_path
        self.lock_path = os.path.join(band, ".gpu.lock")
        self.enabled = enabled and shutil.which("ffmpeg") is not None
        self.cache = {}
        try:
            self.cache = read_json(cache_path)
        except (OSError, ValueError):
            pass
        self.mutex = threading.Lock()
        self.wake = threading.Event()
        self.background = deque()
        self.urgent = deque()
        if self.enabled:
            threading.Thread(target=self._run, daemon=True).start()

    @staticmethod
    def _sig(full):
        st = os.stat(full)
        return f"{st.st_size}:{int(st.st_mtime)}"

    def get(self, rel_file):
        hit = self.cache.get(rel_file)
        full = os.path.join(self.band, rel_file)
        if hit and os.path.exists(full) and hit.get("sig") == self._sig(full):
            return hit.get("lufs")
        return None

    def want(self, rel_files, urgent=False):
        if not self.enabled:
            return
        with self.mutex:
            for r in rel_files:
                if self.get(r) is None and not (self.cache.get(r, {}).get("failed")):
                    (self.urgent if urgent else self.background).append(r)
        self.wake.set()

    def _heavy_job_running(self):
        try:
            with open(self.lock_path, "a") as f:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f, fcntl.LOCK_UN)
            return False
        except BlockingIOError:
            return True
        except OSError:
            return False

    def _next(self):
        with self.mutex:
            if self.urgent:
                return self.urgent.popleft()
            if self.background and not self._heavy_job_running():
                return self.background.popleft()
        return None

    def _run(self):
        while True:
            rel = self._next()
            if rel is None:
                self.wake.wait(60)
                self.wake.clear()
                continue
            if self.get(rel) is not None:
                continue
            full = os.path.join(self.band, rel)
            cmd = ["ffmpeg", "-nostats", "-hide_banner", "-threads", "1", "-i", full,
                   "-af", "ebur128=framelog=quiet", "-f", "null", "-"]
            prefix = ["nice", "-n", "19"] + (["ionice", "-c", "3"] if shutil.which("ionice") else [])
            try:
                out = subprocess.run(prefix + cmd, capture_output=True, text=True, timeout=300)
                lufs = parse_ebur128(out.stderr)
                entry = {"sig": self._sig(full), "lufs": lufs}
                if lufs is None:
                    entry["failed"] = True
            except (OSError, subprocess.TimeoutExpired):
                continue
            with self.mutex:
                self.cache[rel] = entry
                tmp = self.cache_path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(self.cache, f)
                os.replace(tmp, self.cache_path)


# ---------------------------------------------------------------- HTTP

def parse_range(header, size):
    """'bytes=a-b' -> (start, end) inclusive, or None for no/unsupported range. Raises ValueError if unsatisfiable."""
    if not header or not header.startswith("bytes=") or "," in header:
        return None
    a, _, b = header[6:].strip().partition("-")
    if a == "":
        if not b:
            return None
        n = int(b)
        if n == 0:
            raise ValueError
        return max(0, size - n), size - 1
    start = int(a)
    end = int(b) if b else size - 1
    if start >= size or start > end:
        raise ValueError
    return start, min(end, size - 1)


class App:
    def __init__(self, band, dirs, skip, measure=True, catalog=None, act_titles=True):
        self.band = os.path.abspath(os.path.expanduser(band))
        self.dirs = dirs
        self.skip = skip
        state = os.path.join(self.band, ".player")
        os.makedirs(state, exist_ok=True)
        self.picks = Picks(os.path.join(self.band, "picks.json"))
        self.loud = Loudness(self.band, os.path.join(state, "loudness.json"), measure)
        self.catalog_path = catalog
        self.act_titles = act_titles
        self._songs, self._at, self._catalog = None, 0, {}
        self._scan_lock = threading.Lock()

    def songs(self, fresh=False):
        with self._scan_lock:
            if fresh or self._songs is None or time.time() - self._at > 20:
                catalog = load_catalog(self.catalog_path) if self.catalog_path else {}
                self._catalog = catalog
                self._songs = scan(self.band, self.dirs, self.skip, catalog)
                self._at = time.time()
                self.loud.want([t["file"] for s in self._songs.values() for t in s["takes"]])
            return self._songs

    @staticmethod
    def song_picks(picks, slug, s):
        """The song's saved picks, including any saved under an old slug before the catalog joined them."""
        merged = {"pick": None, "takes": {}}
        for key in list(reversed(s.get("aliases", []))) + [slug]:
            p = picks.get(key) or {}
            merged["takes"].update(p.get("takes") or {})
            if p.get("pick"):
                merged["pick"] = p["pick"]
        return merged

    def library(self, fresh=False):
        picks = self.picks.load()["songs"]
        out = []
        for slug, s in self.songs(fresh).items():
            p = self.song_picks(picks, slug, s)
            pick = p.get("pick")
            out.append({"slug": slug, "title": self.title(s), "act": s.get("act"), "actTitle": s.get("actTitle"),
                        "takes": len(s["takes"]), "updated": s["updated"],
                        "pick": pick, "pickLabel": next((t["label"] for t in s["takes"] if t["id"] == pick), None),
                        "rated": sum(1 for t in s["takes"] if p.get("takes", {}).get(t["id"], {}).get("stars"))})
        out.sort(key=lambda x: -x["updated"])
        return {"songs": out, "band": self.band, "measuring": self.loud.enabled}

    def song(self, slug):
        s = self.songs().get(slug)
        if not s:
            return None
        self.loud.want([t["file"] for t in s["takes"]], urgent=True)
        p = self.song_picks(self.picks.load()["songs"], slug, s)
        takes = []
        for t in s["takes"]:
            mine = p.get("takes", {}).get(t["id"], {})
            takes.append(dict(t, lufs=self.loud.get(t["file"]), stars=mine.get("stars", 0), note=mine.get("note", "")))
        return {"slug": slug, "title": self.title(s), "act": s.get("act"), "actTitle": s.get("actTitle"),
                "pick": p.get("pick"), "takes": takes}

    def catalog(self, fresh=False):
        """Everything for the phone's library in one answer: artists (acts), albums, and tracks with their takes.

        An act is an artist and its songs one album, named after the act unless catalog.json's "albums" names it,
        numbered in catalog.json's order. Artwork comes from ~/band/artwork (see artwork())."""
        songs = self.songs(fresh)
        cat = self._catalog or {}
        acts = cat.get("acts", {})
        album_titles = cat.get("albums", {})
        order = {slug: i for i, slug in enumerate(cat.get("songs", {}))}
        art = self.artwork()
        picks = self.picks.load()["songs"]
        used = sorted({s["act"] for s in songs.values() if s.get("act")})
        artists = [{"id": a, "name": acts.get(a, a), "artwork": art.get(f"{a}/artist")} for a in used]
        albums = [{"id": a, "title": album_titles.get(a) or acts.get(a, a), "artist": a,
                   "artwork": art.get(f"{a}/album")} for a in used]
        tracks = []
        for slug, s in songs.items():
            act = s.get("act")
            p = self.song_picks(picks, slug, s)
            takes = []
            for t in s["takes"]:
                mine = p.get("takes", {}).get(t["id"], {})
                takes.append(dict(t, lufs=self.loud.get(t["file"]), stars=mine.get("stars", 0),
                                  note=mine.get("note", "")))
            tracks.append({"slug": slug, "title": s["title"], "artist": act, "album": act,
                           "artwork": art.get(f"{act}/{slug}" if act else slug),
                           "pick": p.get("pick"), "updated": s["updated"], "takes": takes,
                           "_order": order.get(slug, len(order))})
        # number each album's tracks in catalog order (then newest first for songs the catalog doesn't list)
        tracks.sort(key=lambda t: (t["album"] or "", t["_order"], -t["updated"], t["slug"]))
        counters = {}
        for t in tracks:
            del t["_order"]
            if t["album"]:
                counters[t["album"]] = counters.get(t["album"], 0) + 1
                t["number"] = counters[t["album"]]
            else:
                t["number"] = None
        return {"version": 1, "artists": artists, "albums": albums, "tracks": tracks,
                "measuring": self.loud.enabled}

    def artwork(self):
        """{"<act>/artist" | "<act>/album" | "<act>/<slug>" | "<slug>": "/art/<path>"} from ~/band/artwork."""
        top = os.path.join(self.band, "artwork")
        found = {}
        if not os.path.isdir(top):
            return found
        for root, subdirs, files in os.walk(top, followlinks=False):
            subdirs[:] = sorted(d for d in subdirs if not d.startswith("."))
            for f in sorted(files):
                stem, ext = os.path.splitext(f)
                if ext.lower() not in ART_EXTS or stem.startswith("."):
                    continue
                rel = os.path.relpath(os.path.join(root, f), top).replace(os.sep, "/")
                key = os.path.splitext(rel)[0]
                found.setdefault(key, "/art/" + quote(rel))
        return found

    def art_path(self, rel):
        """The file for /art/<rel>, only if artwork() lists it."""
        rel = unquote(rel)
        if "/art/" + quote(rel) in self.artwork().values():
            return os.path.join(self.band, "artwork", rel)
        return None

    def title(self, s):
        """'AFTER HOURS · One More Night', so the phone's flat song list still shows the act."""
        if self.act_titles and s.get("actTitle"):
            return f"{s['actTitle']} · {s['title']}"
        return s["title"]

    def audio_path(self, rel):
        """The file for /audio/<rel>, only if it is a take we listed (no wandering around the disk)."""
        rel = unquote(rel)
        for s in self.songs().values():
            for t in s["takes"]:
                if t["file"] == rel:
                    return os.path.join(self.band, rel)
        return None


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        server_version = "TakePlayer/1"

        def log_message(self, fmt, *args):
            if os.environ.get("PLAYER_LOG"):
                sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/api/library":
                self._json(app.library(fresh="fresh" in q))
            elif u.path == "/api/song":
                s = app.song(q.get("slug", [""])[0])
                self._json(s) if s else self._json({"error": "no such song"}, 404)
            elif u.path == "/api/catalog":
                self._json(app.catalog(fresh="fresh" in q))
            elif u.path.startswith("/audio/"):
                self._audio(u.path[len("/audio/"):])
            elif u.path.startswith("/art/"):
                self._art(u.path[len("/art/"):])
            else:
                self._json({"error": "not found"}, 404)

        def do_HEAD(self):
            u = urlparse(self.path)
            if u.path.startswith("/audio/"):
                self._audio(u.path[len("/audio/"):], head=True)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            u = urlparse(self.path)
            if u.path != "/api/pick":
                return self._json({"error": "not found"}, 404)
            try:
                n = int(self.headers.get("Content-Length") or 0)
                req = json.loads(self.rfile.read(min(n, 65536)) or b"{}")
                song = app.picks.update(req.get("song"), req.get("take"), pick=req.get("pick"),
                                        stars=req.get("stars"), note=req.get("note"))
                self._json({"ok": True, "song": song})
            except (ValueError, TypeError) as e:
                self._json({"error": str(e)}, 400)

        def _art(self, rel):
            full = app.art_path(rel)
            if not full or not os.path.isfile(full):
                return self._json({"error": "not found"}, 404)
            with open(full, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(full)[0] or "application/octet-stream")
            self.send_header("Cache-Control", "private, max-age=300")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _audio(self, rel, head=False):
            full = app.audio_path(rel)
            if not full or not os.path.isfile(full):
                return self._json({"error": "not found"}, 404)
            size = os.path.getsize(full)
            ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
            try:
                rng = parse_range(self.headers.get("Range"), size)
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            start, end = rng if rng else (0, size - 1)
            self.send_response(206 if rng else 200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "private, max-age=3600")
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            if head:
                return
            with open(full, "rb") as f:
                f.seek(start)
                left = end - start + 1
                try:
                    while left > 0:
                        chunk = f.read(min(65536, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # Safari often drops a range request part way; that's normal

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--band", default="~/band", help="the band folder (default ~/band)")
    ap.add_argument("--dirs", default=",".join(DEFAULT_DIRS),
                    help="comma-separated folders under --band to look for takes in (default %(default)s)")
    ap.add_argument("--skip", default=",".join(DEFAULT_SKIP), help="folder names to skip (default %(default)s)")
    ap.add_argument("--bind", default="0.0.0.0", help="address to listen on; use the Tailscale address to keep it off the LAN")
    ap.add_argument("--port", type=int, default=8642)
    ap.add_argument("--no-loudness", action="store_true", help="don't measure loudness (A/B is then not matched)")
    ap.add_argument("--catalog", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "catalog.json"),
                    help="songs, acts and old slugs (default: catalog.json next to this script; missing is fine)")
    ap.add_argument("--plain-titles", action="store_true", help="don't put the act in front of song titles")
    a = ap.parse_args()
    app = App(a.band, [d for d in a.dirs.split(",") if d], [s for s in a.skip.split(",") if s], not a.no_loudness,
              catalog=a.catalog, act_titles=not a.plain_titles)
    n = sum(len(s["takes"]) for s in app.songs().values())
    print(f"{n} takes in {len(app.songs())} songs under {app.band}")
    print(f"in Soundscape, Listen, set the server to http://{'<laptop tailscale address>' if a.bind == '0.0.0.0' else a.bind}:{a.port}")
    srv = ThreadingHTTPServer((a.bind, a.port), make_handler(app))
    srv.daemon_threads = True
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
