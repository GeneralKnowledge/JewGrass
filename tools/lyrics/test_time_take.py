import json
import os
import shutil
import tempfile
import unittest

import time_take as tt

LYRICS = """[verse]
Neon on the water,
City in my bones

[chorus]
[Big, full band]
One more night
"""


def seg(*words):
    return {"words": [{"word": f" {w}", "start": s, "end": s + 0.2, "probability": p} for w, s, p in words]}


class MatchTest(unittest.TestCase):
    def test_written_words_get_heard_times_and_sections(self):
        heard = [seg(("neon", 12.3, 0.9), ("on", 12.6, 0.9), ("the", 12.8, 0.9), ("water", 13.0, 0.95)),
                 seg(("city", 16.2, 0.9), ("and", 16.5, 0.3), ("my", 16.8, 0.9), ("bones", 17.0, 0.9)),
                 seg(("one", 31.4, 0.9), ("more", 31.7, 0.9), ("night", 32.0, 0.9))]
        lrc = tt.to_lrc(tt.match(LYRICS, heard)).splitlines()
        self.assertEqual(lrc, [
            "[00:12.30] [verse]",
            "[00:12.30] <00:12.30>Neon <00:12.60>on <00:12.80>the <00:13.00>water,",
            # "in" was heard as "and": its time is spread between "City" and "my"
            "[00:16.20] <00:16.20>City <00:16.50>in <00:16.80>my <00:17.00>bones",
            "[00:31.40] [chorus]",  # the performance note is dropped
            "[00:31.40] <00:31.40>One <00:31.70>more <00:32.00>night",
        ])

    def test_a_poorly_heard_line_keeps_only_its_line_time(self):
        heard = [seg(("neon", 12.3, 0.9), ("on", 12.6, 0.9), ("the", 12.8, 0.9), ("water", 13.0, 0.95)),
                 seg(("bones", 17.0, 0.9)),
                 seg(("one", 31.4, 0.9), ("more", 31.7, 0.9), ("night", 32.0, 0.9))]
        lines = tt.to_lrc(tt.match(LYRICS, heard)).splitlines()
        self.assertEqual(lines[2], "[00:17.00] City in my bones")

    def test_an_unheard_line_sits_between_its_neighbours(self):
        heard = [seg(("neon", 10.0, 0.9), ("on", 10.2, 0.9), ("the", 10.4, 0.9), ("water", 10.6, 0.9)),
                 seg(("one", 30.0, 0.9), ("more", 30.2, 0.9), ("night", 30.4, 0.9))]
        m = tt.match(LYRICS, heard)
        self.assertEqual([round(l["start"], 1) for l in m], [10.0, 20.0, 30.0])
        self.assertIn("[00:20.00] City in my bones", tt.to_lrc(m))


class FilesTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root)

    def make(self, rel, lyrics, lrc=False):
        base = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(base), exist_ok=True)
        open(base + ".mp3", "w").close()
        open(base + ".wav", "w").close()
        json.dump({"lyrics": lyrics}, open(base + ".json", "w"))
        if lrc:
            open(base + ".lrc", "w").close()
        return base

    def test_takes_to_time_skip_done_and_wordless_ones(self):
        a = self.make("renders/song/song-1", LYRICS)
        self.make("renders/song/song-2", LYRICS, lrc=True)
        self.make("renders/song/song-3", "")
        self.assertEqual(list(tt.takes(self.root)), [a + ".mp3"])

    def test_only_picked_takes(self):
        self.make("renders/song/song-1", LYRICS)
        b = self.make("renders/song/song-2", LYRICS)
        picks = os.path.join(self.root, "picks.json")
        json.dump({"songs": {"song": {"pick": "renders/song/song-2", "takes": {}}}}, open(picks, "w"))
        self.assertEqual(list(tt.takes(self.root, picks)), [b + ".mp3"])

    def test_writes_the_lrc_from_given_words(self):
        a = self.make("renders/song/song-1", LYRICS)
        words = os.path.join(self.root, "w.json")
        json.dump([seg(("one", 31.4, 0.9), ("more", 31.7, 0.9), ("night", 32.0, 0.9))], open(words, "w"))
        out = tt.time_take(a + ".mp3", "medium.en", words=words)
        self.assertEqual(out, a + ".lrc")
        self.assertIn("<00:31.40>One", open(out).read())


if __name__ == "__main__":
    unittest.main()
