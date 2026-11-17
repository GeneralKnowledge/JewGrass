# Album art

`covers.py` makes a square fractal cover for every act, in code (numpy + Pillow, no GPU or model), in the same
look as the lyric videos: glowing fractal filaments on deep ink with a soft bloom. Each act has its own fractal,
framing, palette and lettering. Everything is original.

| Act | Album | Artwork |
|---|---|---|
| Natural Capital | Natural Capital | the Deed to the Glen v3 pattern (8-fold Julia, house inks) |
| Book of Names | Book of Names | a 12-fold rose window in gold leaf and crimson, redacted line under it |
| AFTER HOURS | AFTER HOURS | a 4-fold Julia (four voices, one blend) in late-night neon |
| MR. SIDEQUEST | MR. SIDEQUEST | a bright Mandelbrot spiral in big game-map pixels, quest marker |
| NIGHTSHIFT FM | Phantom Frequency | violet and red Julia, scanlines, a tuning dial |
| IRON CANTICLE | Candles for a Dead Sun | Burning Ship spires in ember under a black sun |
| LAST CARRIAGE | Overnight North | a stretched Julia in cold steel with a seam of first light |
| LOWBRIDGE | Under the Bridge | grey Mandelbrot filaments with one sodium-orange band |

```
python3 tools/art/covers.py OUT_DIR [act ...]   # ~1 min for all eight on a CPU
```

writes `OUT_DIR/<act>/album.jpg` (1024 px, with lettering), `artist.jpg` (1024 px, no lettering) and
`album-2048.png` (master). The 1024 px covers are committed in `covers/`, laid out the way the take server reads
`~/band/artwork`, so on the laptop:

```
cp -r ~/band/JewGrass/tools/art/covers/. ~/band/artwork/   # adjust to where the repo is checked out
```

The take server picks them up on the next `/api/catalog?fresh`. Lettering uses the repo's fonts plus Inter and
JetBrains Mono where installed (falls back to Oswald). Make a new version by editing an act's function; keep old
covers (project files `album-art/v1/`, `v2/`, ...).
