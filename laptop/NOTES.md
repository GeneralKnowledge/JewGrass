# Band notes (standing preferences)

- Vary song lengths: don't default to 240 s. Mix shorter tracks (~120–150 s) with longer ones (~300 s, in one pass if VRAM allows, otherwise repaint-extend past the end).
- Add subtle bagpipes to captions where they fit, for example "subtle distant highland bagpipe drone ... bagpipes low in the mix".
- Heavy jobs: always through ~/band/heavy.sh and the GPU lock (`flock ~/band/.gpu.lock`), one repaint seed per process.
- The spoken intro usually comes out; spoken outros often don't (Glen, Swim). Plan an outro repaint of the last ~20 s with "spoken word outro" in the caption.
