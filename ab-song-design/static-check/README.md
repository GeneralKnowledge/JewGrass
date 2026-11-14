# Static check: Brass Plate, seed 7102, 30-90 s of each render (320k mp3)

| file | caption and lyrics | key | timesig |
|---|---|---|---|
| brass-7102-pre-design | old (album run) | - | - |
| brass-7102-design-v1 | design-v1 | C minor | 4 |
| brass-7102-static-plain | design-v1 | - | - |
| brass-7102-static-keyonly | design-v1 | C minor | - |
| brass-7102-static-timesigonly | design-v1 | - | 4 |

All rendered with the same laptop settings (fp32 compute, SDPA, no offload), seed 7102, no mastering.
Measured (full takes), no condition stands out:

| file | quiet-frame 6-16 kHz (dB) | flatness | hiss rel (dB) | HF floor (dB) | clicks/min | clipped |
|---|---|---|---|---|---|---|
| pre-design | -92.6 | 0.37 | -27.4 | -110.9 | 329 | 0 |
| design-v1 | -91.2 | 0.30 | -26.4 | -109.5 | 240 | 0 |
| static-plain | -92.8 | 0.31 | -25.7 | -108.1 | 307 | 0 |
| static-keyonly | -90.9 | 0.28 | -26.7 | -109.0 | 163 | 0 |
| static-timesigonly | -90.6 | 0.30 | -25.7 | -106.7 | 249 | 0 |
