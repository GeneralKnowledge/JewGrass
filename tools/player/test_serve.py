"""Tests for the take server: python3 -m unittest discover tools/player"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
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


class CatalogTest(unittest.TestCase):
    def setUp(self):
        self.band = tempfile.mkdtemp()
        b = self.band
        for run, slug, seed in [("20261114-2004", "swim-at-your-own-risk", 8101),
                                ("20261115-1635", "nc-swim-at-your-own-risk", 1501),
                                ("20261115-1635", "ah-one-more-night", 1501)]:
            r = f"{b}/kaggle/runs/{run}/renders/{slug}"
            touch(f"{r}/{slug}-{seed}-xl-turbo.mp3")
            json.dump({"slug": slug, "title": slug, "seed": seed, "model": "xl-turbo"},
                      open(f"{r}/{slug}-{seed}-xl-turbo.json", "w"))
        touch(f"{b}/outputs/swim-len240-5555.wav")
        self.catalog = f"{b}/catalog.json"
        json.dump({"acts": {"natural-capital": "Natural Capital", "after-hours": "AFTER HOURS"},
                   "songs": {"nc-swim-at-your-own-risk": {"title": "Swim at Your Own Risk", "act": "natural-capital",
                                                          "aliases": ["swim-at-your-own-risk", "swim"]},
                             "ah-one-more-night": {"title": "One More Night", "act": "after-hours", "aliases": []}}},
                  open(self.catalog, "w"))

    def tearDown(self):
        shutil.rmtree(self.band)

    def test_old_slugs_join_their_song(self):
        songs = serve.scan(self.band, serve.DEFAULT_DIRS, serve.DEFAULT_SKIP, serve.load_catalog(self.catalog))
        self.assertEqual(sorted(songs), ["ah-one-more-night", "nc-swim-at-your-own-risk"])
        swim = songs["nc-swim-at-your-own-risk"]
        self.assertEqual((swim["title"], swim["act"], swim["actTitle"]),
                         ("Swim at Your Own Risk", "natural-capital", "Natural Capital"))
        self.assertEqual(sorted(t["label"] for t in swim["takes"]),
                         ["seed 1501 · xl-turbo · run 20261115-1635", "seed 8101 · xl-turbo · run 20261114-2004",
                          "swim-len240-5555"])
        # one run only: no run in the label
        self.assertEqual([t["label"] for t in songs["ah-one-more-night"]["takes"]], ["seed 1501 · xl-turbo"])

    def test_act_in_titles_and_old_picks_kept(self):
        app = serve.App(self.band, serve.DEFAULT_DIRS, serve.DEFAULT_SKIP, measure=False, catalog=self.catalog)
        app.picks.update("swim", "outputs/swim-len240-5555", pick=True, stars=5)  # saved before the catalog
        lib = {x["slug"]: x for x in app.library()["songs"]}
        swim = lib["nc-swim-at-your-own-risk"]
        self.assertEqual((swim["title"], swim["actTitle"]), ("Natural Capital · Swim at Your Own Risk", "Natural Capital"))
        self.assertEqual((swim["pick"], swim["rated"]), ("outputs/swim-len240-5555", 1))
        plain = serve.App(self.band, serve.DEFAULT_DIRS, serve.DEFAULT_SKIP, measure=False, catalog=self.catalog,
                          act_titles=False)
        self.assertEqual(plain.song("ah-one-more-night")["title"], "One More Night")

    def test_catalog_has_artists_albums_tracks_and_artwork(self):
        b = self.band
        touch(f"{b}/artwork/after-hours/artist.jpg", b"\xff\xd8jpeg")
        touch(f"{b}/artwork/after-hours/ah-one-more-night.png", b"\x89PNG")
        touch(f"{b}/artwork/notes.txt", b"not art")
        app = serve.App(b, serve.DEFAULT_DIRS, serve.DEFAULT_SKIP, measure=False, catalog=self.catalog)
        app.picks.update("ah-one-more-night", "kaggle/runs/20261115-1635/renders/ah-one-more-night/"
                         "ah-one-more-night-1501-xl-turbo", pick=True, stars=4)
        cat = app.catalog()
        self.assertEqual(cat["version"], 1)
        self.assertEqual([(a["id"], a["name"], a["artwork"]) for a in cat["artists"]],
                         [("after-hours", "AFTER HOURS", "/art/after-hours/artist.jpg"),
                          ("natural-capital", "Natural Capital", None)])
        self.assertEqual([(a["id"], a["title"], a["artist"]) for a in cat["albums"]],
                         [("after-hours", "AFTER HOURS", "after-hours"),
                          ("natural-capital", "Natural Capital", "natural-capital")])
        tracks = {t["slug"]: t for t in cat["tracks"]}
        night = tracks["ah-one-more-night"]
        self.assertEqual((night["title"], night["artist"], night["album"], night["number"]),
                         ("One More Night", "after-hours", "after-hours", 1))
        self.assertEqual(night["artwork"], "/art/after-hours/ah-one-more-night.png")
        self.assertTrue(night["pick"].endswith("ah-one-more-night-1501-xl-turbo"))
        self.assertEqual(night["takes"][0]["stars"], 4)
        self.assertEqual(len(tracks["nc-swim-at-your-own-risk"]["takes"]), 3)
        # artwork is served only when listed
        self.assertTrue(app.art_path("after-hours/artist.jpg").endswith("artwork/after-hours/artist.jpg"))
        self.assertIsNone(app.art_path("notes.txt"))
        self.assertIsNone(app.art_path("../catalog.json"))

    def test_catalog_over_http(self):
        touch(f"{self.band}/artwork/after-hours/album.jpg", b"\xff\xd8jpeg")
        app = serve.App(self.band, serve.DEFAULT_DIRS, serve.DEFAULT_SKIP, measure=False, catalog=self.catalog)
        srv = ThreadingHTTPServer(("127.0.0.1", 0), serve.make_handler(app))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            base = f"http://127.0.0.1:{srv.server_address[1]}"
            cat = json.load(urllib.request.urlopen(base + "/api/catalog"))
            art = cat["albums"][0]["artwork"]
            self.assertEqual(art, "/art/after-hours/album.jpg")
            self.assertEqual(urllib.request.urlopen(base + art).read(), b"\xff\xd8jpeg")
            with self.assertRaises(urllib.error.HTTPError):
                urllib.request.urlopen(base + "/art/../catalog.json")
        finally:
            srv.shutdown()

    def test_no_catalog_is_fine(self):
        self.assertEqual(serve.load_catalog(f"{self.band}/missing.json"), {})
        songs = serve.scan(self.band, serve.DEFAULT_DIRS, serve.DEFAULT_SKIP, {})
        self.assertIn("swim", songs)

    def test_repo_catalog_is_valid(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "catalog.json")
        index = serve.catalog_index(serve.load_catalog(path))
        self.assertEqual(index["swim"][0], "nc-swim-at-your-own-risk")
        self.assertEqual(index["ah-one-more-night"][0], "ah-one-more-night")


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
