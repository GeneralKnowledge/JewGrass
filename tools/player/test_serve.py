"""Tests for the take server: python3 -m unittest discover tools/player"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import serve  # noqa: E402


def touch(path, data=b"\0" * 1000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


class LibraryTest(unittest.TestCase):
    def setUp(self):
        self.band = tempfile.mkdtemp()
        b = self.band
        r = f"{b}/kaggle/runs/20261115-2100/renders/gavs-song"
        touch(f"{r}/gavs-song-7202.mp3")
        touch(f"{r}/gavs-song-7202.wav")
        json.dump({"slug": "gavs-song", "title": "Gav's Song", "seed": 7202, "model": "turbo",
                   "caption": "fiddle", "lyrics": "[verse]\nhello"}, open(f"{r}/gavs-song-7202.json", "w"))
        touch(f"{r}/gavs-song-7202-xl-turbo.mp3")
        json.dump({"slug": "gavs-song", "seed": 7202, "model": "xl-turbo"}, open(f"{r}/gavs-song-7202-xl-turbo.json", "w"))
        touch(f"{b}/songs/deed-to-the-glen/60-master/pedalboard-v2.wav")
        touch(f"{b}/songs/deed-to-the-glen/20-stems/vocals.wav")
        touch(f"{b}/outputs/swim-len240-1111.wav")
        touch(f"{b}/tools/fx/.venv/lib/x.wav")

    def tearDown(self):
        shutil.rmtree(self.band)

    def test_groups_takes_by_song(self):
        songs = serve.scan(self.band, serve.DEFAULT_DIRS, serve.DEFAULT_SKIP)
        self.assertEqual(sorted(songs), ["deed-to-the-glen", "gavs-song", "swim"])
        gav = songs["gavs-song"]
        self.assertEqual(gav["title"], "Gav's Song")
        files = sorted(t["file"] for t in gav["takes"])
        # the mp3 wins over the wav of the same take; the xl-turbo take is separate
        self.assertEqual(files, ["kaggle/runs/20261115-2100/renders/gavs-song/gavs-song-7202-xl-turbo.mp3",
                                 "kaggle/runs/20261115-2100/renders/gavs-song/gavs-song-7202.mp3"])
        self.assertEqual(sorted(t["label"] for t in gav["takes"]), ["seed 7202", "seed 7202 · xl-turbo"])
        glen = songs["deed-to-the-glen"]["takes"]
        self.assertEqual([t["label"] for t in glen], ["pedalboard-v2 · master"])  # stems skipped

    def test_picks_round_trip(self):
        app = serve.App(self.band, serve.DEFAULT_DIRS, serve.DEFAULT_SKIP, measure=False)
        take = app.song("gavs-song")["takes"][0]["id"]
        app.picks.update("gavs-song", take, pick=True, stars=4, note="tight chorus")
        s = app.song("gavs-song")
        self.assertEqual(s["pick"], take)
        self.assertEqual((s["takes"][0]["stars"], s["takes"][0]["note"]), (4, "tight chorus"))
        lib = {x["slug"]: x for x in app.library()["songs"]}
        self.assertEqual(lib["gavs-song"]["pickLabel"], s["takes"][0]["label"])
        app.picks.update("gavs-song", take, pick=False)
        self.assertIsNone(app.song("gavs-song")["pick"])
        with self.assertRaises(ValueError):
            app.picks.update("gavs-song", take, stars=9)

    def test_audio_only_serves_listed_takes(self):
        app = serve.App(self.band, serve.DEFAULT_DIRS, serve.DEFAULT_SKIP, measure=False)
        self.assertIsNotNone(app.audio_path("outputs/swim-len240-1111.wav"))
        self.assertIsNone(app.audio_path("../etc/passwd"))
        self.assertIsNone(app.audio_path("songs/deed-to-the-glen/20-stems/vocals.wav"))

    def test_http(self):
        app = serve.App(self.band, serve.DEFAULT_DIRS, serve.DEFAULT_SKIP, measure=False)
        srv = ThreadingHTTPServer(("127.0.0.1", 0), serve.make_handler(app))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        try:
            lib = json.load(urllib.request.urlopen(base + "/api/library"))
            self.assertEqual(len(lib["songs"]), 3)
            req = urllib.request.Request(base + "/audio/outputs/swim-len240-1111.wav", headers={"Range": "bytes=10-19"})
            r = urllib.request.urlopen(req)
            self.assertEqual((r.status, r.headers["Content-Range"], len(r.read())), (206, "bytes 10-19/1000", 10))
            song = json.load(urllib.request.urlopen(base + "/api/song?slug=swim"))
            body = json.dumps({"song": "swim", "take": song["takes"][0]["id"], "pick": True}).encode()
            r = urllib.request.urlopen(urllib.request.Request(base + "/api/pick", data=body, method="POST"))
            self.assertEqual(json.load(r)["song"]["pick"], song["takes"][0]["id"])
            self.assertEqual(json.load(open(os.path.join(self.band, "picks.json")))["songs"]["swim"]["pick"],
                             "outputs/swim-len240-1111")
        finally:
            srv.shutdown()


class HelpersTest(unittest.TestCase):
    def test_range(self):
        self.assertIsNone(serve.parse_range(None, 100))
        self.assertEqual(serve.parse_range("bytes=0-", 100), (0, 99))
        self.assertEqual(serve.parse_range("bytes=0-1", 100), (0, 1))
        self.assertEqual(serve.parse_range("bytes=-10", 100), (90, 99))
        self.assertEqual(serve.parse_range("bytes=50-500", 100), (50, 99))
        with self.assertRaises(ValueError):
            serve.parse_range("bytes=100-", 100)

    def test_ebur128(self):
        text = "[Parsed_ebur128_0 @ 0x1] Summary:\n\n  Integrated loudness:\n    I:         -14.3 LUFS\n"
        self.assertEqual(serve.parse_ebur128(text), -14.3)
        self.assertIsNone(serve.parse_ebur128("    I:         -inf LUFS"))

    @unittest.skipUnless(shutil.which("ffmpeg"), "needs ffmpeg")
    def test_measures_a_real_file(self):
        d = tempfile.mkdtemp()
        try:
            touch(f"{d}/outputs/tone-1.wav", b"")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=440:d=3",
                            f"{d}/outputs/tone-1.wav"], check=True)
            loud = serve.Loudness(d, f"{d}/cache.json")
            loud.want(["outputs/tone-1.wav"], urgent=True)
            for _ in range(100):
                if loud.get("outputs/tone-1.wav") is not None:
                    break
                threading.Event().wait(0.1)
            self.assertAlmostEqual(loud.get("outputs/tone-1.wav"), -21.0, delta=2)
        finally:
            shutil.rmtree(d)


if __name__ == "__main__":
    unittest.main()
