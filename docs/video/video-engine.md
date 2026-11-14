# JewGrass Video Engine (design, v1)

A separate stage of the project that turns any finished song folder into an
art-directed music video. It's built from Reuben's plan of 14 Nov 2026:
**AI makes the artwork and the storyboard, deterministic code renders the
video.** It's independent of music generation: it only reads a finished song
folder and never touches ACE-Step or its environment.

First milestone: **one genuinely good video for _Deed to the Glen_**, not a
generic template.

---

## 1. Verified tool check (14 Nov 2026)

The five projects in the plan came from an outside AI summary, so each was
checked on GitHub directly: does it exist, is it maintained, what licence is it
under, and does it fit a headless GTX 1060 laptop.

| Project | Exists? | Activity | Licence | What it really is | Verdict |
|---|---|---|---|---|---|
| **muvid** (`reeleehq/muvid`) | Yes | Very active: 396 commits May–Nov 2026, v0.0.187, one maintainer | MIT | Python toolkit. The key-free parts (`visualize`, `lyricvid`, `montage`, `choreo`) are deterministic and ffmpeg-only. The "AI narrative" part needs paid **fal.ai** and **ElevenLabs** keys. | **Use for ideas, not as the backbone.** Its looks are visualiser and subtitle (ASS) based, it pins a chain of the author's own sibling packages that change weekly, and the AI part is cloud-paid. Worth borrowing: beat-cut montage logic, the "verify the video before shipping" checks, YouTube-safe export settings. |
| **Phrasync** (`jaydemks/phrasync`) | Yes | 27 commits Aug–Nov 2026, one developer, 10 stars | Apache-2.0 | A GUI lyric-video studio you drive in a browser (timeline, 2D/3D kinetic text, Faster-Whisper alignment, weather layers). | **Ideas only.** It's an interactive editor, not a pipeline, so it can't run unattended. Good reference for word-timing correction and kinetic text styles. |
| **PulseForge** (`TheAwaken1/PulseForge`) | Yes | 12 commits Feb–Oct 2026 | **No licence** (all rights reserved) | A Tauri/Pinokio desktop visualiser editor. | **Don't use.** With no licence we may not reuse any code. (Several unrelated repos share the name.) |
| **lyricreel** (`malilion/lyricreel`) | Yes | 6 commits Sep–Nov 2026, 1 star | Code MIT; its art, docs and demo are **CC BY-NC 4.0** (non-commercial) | Song + lyrics → lyric video, every frame drawn in code (headless Chrome + Canvas, ffmpeg), designed to be driven by an AI coding agent. | **Best architectural reference.** The approach is the one we want (agent plans scenes, code draws, review stills, deterministic). Don't copy its scenes or art: non-commercial, and its pop-art sticker look isn't ours. (A second, unrelated `lyricreel` also exists.) |
| **cymatic** (most likely `coreyepstein/cymatic`) | Yes, but the name is ambiguous | 35 commits, **all on one day** (2 Jun 2026), nothing since; 1 star | MIT | Browser WebGPU visualiser library with an "auto-director" that evolves intensity, motion, bloom and density across a song's arc. | **Ideas only, unmaintained.** Its nice look needs WebGPU, which a headless laptop won't give us. The auto-director idea (a few smooth "macro" signals for the whole song) is good, and we adopt it in §5. |

**Conclusion:** none of the five is a safe foundation. Two are one-person
projects a few weeks old, one has no licence, one is abandoned, and the most
mature (muvid) is visualiser-shaped and fast-moving. We build a **small engine of
our own** on long-established libraries and borrow their ideas.

What we build on instead (all checked: exist, maintained, permissive):

| Library | Licence | Used for |
|---|---|---|
| ffmpeg | LGPL/GPL build | encoding, muxing, loudness |
| numpy + Pillow | BSD / HPND | image maths, compositing |
| `skia-python` (333★, active Nov 2026) | BSD-3 | fast vector drawing and typography on CPU |
| librosa, beat_this (already on the laptop) | ISC / MIT | beats, downbeats, onsets |
| audio-separator + Demucs (`htdemucs`) (already on the laptop) | MIT | 4 stems: drums / bass / vocals / other |
| faster-whisper (26k★) | MIT | word timestamps, aligned to our own lyrics |
| diffusers (35k★) or stable-diffusion.cpp (8k★, active) | Apache-2.0 / MIT | image generation |
| Depth Anything V2 **Small** (repo Apache-2.0) | Apache-2.0 (Small only; Base/Large weights are non-commercial) | depth maps for 2.5D parallax |

## 2. What image generation is realistic on the GTX 1060

Facts that set the rules (from the ACE-Step work on this exact laptop): Pascal
cards do fp16 maths very slowly and gave NaNs with ACE-Step, so we **compute in
fp32**; torch must be the **cu126** build; one GPU job at a time under
`~/band/.gpu.lock`; 8 GB of system RAM is the tightest limit, not the 6 GB card.

| Model | Fits? | Expected speed (to benchmark) | Licence | Verdict |
|---|---|---|---|---|
| **SD 1.5 family, fp32**, 768×432 + hi-res pass | Yes: UNet ~3.4 GB fp32, loads within 8 GB RAM | roughly 20–40 s per image | CreativeML OpenRAIL-M (commercial use allowed with use restrictions); each fine-tune has its own licence | **Main workhorse.** Plus ControlNet (depth/lineart) and img2img to keep places and compositions consistent. |
| SDXL / SDXL-Turbo | Borderline: ~7 GB of fp16 weights against 8 GB RAM; only workable as a quantised GGUF through stable-diffusion.cpp built for `sm_61` | ~1–2 min per image | SDXL 1.0: OpenRAIL++. SDXL-Turbo: Stability Community Licence (check the terms at download time) | **Later experiment**, only if SD 1.5 can't reach the look. |
| FLUX, SD3.5, video models (AnimateDiff, SVD, Wan) | No | n/a | n/a | **Out.** Too big for 8 GB RAM, and video models invent frame-to-frame changes, which is exactly what the plan wants to avoid. |
| Depth Anything V2 Small | Yes, even on CPU | ~1–3 s per image | Apache-2.0 | For parallax layers. |
| Upscaler (Real-ESRGAN family, BSD-3) | Yes | seconds per image | BSD-3 (check each weight file) | To bring 768-wide art up to 1920+ with headroom for camera moves. |

A video needs about 15–25 base images. With 3–4 candidates each, that's 60–100
images, roughly **40–70 minutes of GPU time**, run as one batch when no music
job is running. Everything goes in its own venv (`~/band/tools/img`), so
ACE-Step's `.venv` is never touched.

**The honest limit:** SD 1.5 can't keep a *face* consistent from video to video.
So recurring **characters are not left to the image model** (see §4).

## 3. Pipeline

```
song folder ──► 1 analyse ──► 2 direct ──► 3 make assets ──► 4 render ──► 5 publish
(master, stems,   analysis.json  storyboard.yaml  plates, depth, vector  frames→ffmpeg  mp4, thumbnail,
 lyrics, song card)                                characters, docs                     vertical cut, lrc
```

Each stage writes files to `~/band/songs/<slug>/70-video/` and only reads the
stage before it, so any stage can be redone alone. Stages 1, 4 and 5 are pure
CPU and deterministic, so they can run on the laptop **or in a cloud session**.
Only stage 3's image generation needs the laptop GPU.

### Stage 1: musical and lyrical analysis (CPU)

Output `analysis.json`, sampled at the video frame rate (30 fps):

- tempo, beats, downbeats, bars (beat_this, already used for mastering checks);
- sections, taken from our own lyric tags (`[Verse 1]`, `[Chorus]`, …) aligned
  to the audio, then checked against novelty peaks;
- four stems (Demucs through the existing `tools/sep` venv), each turned into a
  smoothed envelope plus onsets: **bass, drums, vocals, other** (banjo, fiddle, guitars);
- word timings: faster-whisper word timestamps, matched in order against the
  lyrics we wrote (we never trust the transcription's words, only its timing),
  with a hand-override file for any wrong word;
- macro curves for the whole song (the cymatic idea): **energy, tension
  (build-ups), density, stillness (breakdowns/silence)**, heavily smoothed.

### Stage 2: the creative director

The director is **Claude in a Claude Code session** (no API keys or cost). It
reads the song card (story, villain, act of *Natural Capital*), the lyrics and
`analysis.json`, then writes **`storyboard.yaml`**: a short, readable plan that
Reuben can review and edit before anything renders.

```yaml
song: deed-to-the-glen
style: glen-gothic            # the house style (section 4)
preset: satirical-dark        # per-song visual signature (section 6)
motifs: [red-line, crow, deed, the-hill, fund-logo]
shots:
  - id: s01
    at: intro                 # snapped to bars from analysis.json
    plate: hill-dawn          # a recurring location, state 0
    camera: {move: push-in, speed: slow, parallax: 0.6}
    weather: [mist-low]
    motif: {crow: perched-left}
    text: {mode: spoken-title, lines: intro}
  - id: s07
    at: chorus-1
    plate: hill-dawn@fenced   # same hill, later state
    camera: {move: drift-right}
    react: {bass: ground, drums: cut-allowed}
    text: {mode: chorus-sign, phrase: "I own the glen"}
```

A validator then enforces the rules automatically: every shot is bar-aligned,
the whole song is covered, shot lengths stay between 4 and 16 bars, every motif
used is defined, and every lyric line has a text treatment. A rule-based
fallback can build a plain storyboard for a song without any director.

### Stage 3: assets

Three kinds, each made once per song and saved with its seed and prompt so
they can be remade exactly:

1. **Painted plates** (SD 1.5 on the GPU): landscapes, the village, interiors,
   weather. Wide 16:9 with headroom for camera moves. A depth map for each
   plate is used to split it into 3–4 parallax layers.
2. **Recurring locations as "master plates."** A place like *the hill* is
   blocked out once as a simple vector layout, generated once, then **changed
   by img2img/inpainting**: fence added, development added, lights on. Every
   video that visits the hill starts from the same master, which is what makes
   the world feel continuous.
3. **Vector assets drawn in code**: characters, company logos, the deed, maps,
   newspaper cuttings, the red boundary line, signage. Perfectly consistent
   from video to video, cheap, and editable.

### Stage 4: the deterministic compositor (CPU)

Every frame is a pure function: `frame(t) = render(storyboard, analysis, assets, seed, t)`.
Same inputs, same video, bit for bit. Python + numpy + skia-python, rendering
in parallel chunks (low priority through `heavy.sh` on the laptop) and piping
raw frames to ffmpeg.

Layers, back to front: sky → far plate layer → weather (fog banks from seeded
noise, rain/snow particles) → mid layer → motifs → near layer → in-scene lyrics →
overlay text → film finish (paper grain, halftone, slight gate weave, vignette).

Camera: push/pull, pan, drift and parallax driven by depth layers, with eased
starts and stops. Transitions: cut on downbeat, match-cut on a motif, ink-bleed
wipe, paper tear, fade through fog. Light changes are colour grades over time.

Target speed: under 0.3 s per 1080p frame on 8 laptop threads, so a 4-minute
video (7,200 frames at 30 fps) renders in about 35 minutes. If it's slower we
render at 24 fps (it suits the illustrated look anyway).

### Stage 5: outputs

- `video-1080p.mp4`: H.264 High, yuv420p, CRF 18, AAC 320k from the master,
  `+faststart`, exactly the song's length;
- `thumbnail.jpg` 1280×720, under 2 MB: a hero frame chosen by the director,
  re-composed with the title in the chorus typeface;
- `vertical-cut.mp4` 1080×1920, 30–45 s around the best chorus. **Re-composed,
  not cropped:** the storyboard gives each shot a vertical framing and moves
  the text into the safe zone;
- `lyrics.lrc` and `.srt` (closed captions are separate from the art lyrics);
- `contact-sheet.jpg`: one frame per shot, for review before the full render.

## 4. House style and recurring identity

**Working name: "Glen Gothic."** It's an original look described by technique,
never by naming any artist, studio or existing video:

- **Screen-print palette:** peat black, bone/paper white, heather purple,
  loch teal, fog grey, and one **warning red** that belongs only to money and
  ownership (the boundary line, the fund logo, the price tags). When red
  appears, someone is buying something.
- **Surface:** visible paper grain, coarse halftone in the shadows, woodcut
  hatching on the dark masses, slight misregistration between colour layers.
- **Three registers in one world:** the *tourism poster* (the glen as it's
  sold: flat, bright, too perfect), the *folk-horror painting* (the glen as it
  is: fog, dark water, crows), and the *political cartoon* (the villains: vector
  caricatures, deliberately flat, pasted onto the painted world like
  cuttings). The clash between the cut-out villains and the painted land *is*
  the satire.
- **Characters** are built as reusable vector rigs (head, body, a few
  expressions, a few poses), so **the Laird looks the same in every video**. In
  painted scenes, locals appear as silhouettes in the landscape, which suits
  the folk-horror tone and sidesteps face consistency.
- **Typography:** one blunt condensed sans for slogans and chorus phrases (as
  if stencilled on signs), one plain serif for the documents and deeds, and a
  handwritten face for Ewan's voicemails. Open-licence (SIL OFL) fonts only.

### Motif library (Natural Capital)

Defined once in `video/motifs/`, used across every video on the album:

| Motif | How it behaves |
|---|---|
| **The red line** | A boundary line that creeps across landscapes, maps and eventually faces. Grows a little each time it appears in the album. |
| **The fund's logo** | Turns up on increasingly wrong things: a sheep, a gravestone, a school bell, a salmon. |
| **The crow** | Watches from the edge of the frame. It's the only thing that never changes. |
| **The deed** | A contract that unfolds a little more each chorus until it fills the screen. |
| **The hill** | One master plate whose state changes song by song (bare → surveyed → fenced → developed → bought back). |
| **"Due Friday"** | Each villain's sign-off appears as a stamped notice; in the finale the village paints over it. |

## 5. Making it react to the music, subtly

| Musical element | Visual response |
|---|---|
| Bass (bass stem) | slow camera sway, ground shudder on big 808 hits, fog pulse |
| Drums (drums stem) | shots may only cut on downbeats; snare gives small grain/light kicks; rare flashes only on the biggest hits |
| Vocals (vocal stem + word timing) | lyrics appear and light up as they're sung; spoken parts get documentary title cards |
| Fiddle, banjo, guitars (other stem) | fine particles, threads, drifting embers, hatching that shimmers |
| Build-ups (tension curve) | more particles and fog, faster camera, more text |
| Breakdowns and silence | everything stills: no shake, slow fog, empty frame |

Rules that keep it from being exhausting:

- a **reactivity budget** per section (verses ~30%, chorus ~70%, breakdown ~10%);
- all reactions use attack/release smoothing; nothing jumps on every beat;
- **photosensitivity safety:** at most 3 flashes per second, with brightness
  changes capped, and a check over the finished video.

## 6. Per-song visual signatures (presets)

Each preset is a YAML file in `video/styles/` that sets palette weights, cut
pacing, camera speeds, text modes and the reactivity budget. The director picks
one from the song card and may blend two.

| Preset | For | Direction |
|---|---|---|
| `satirical-dark` | villain songs | cut-out caricatures on painted land, newspaper and document collage, hard cuts on bars |
| `angry-collage` | *Bought and Sold*, protest songs | fast collage, distressed print, hard cuts, stencilled text |
| `reflective` | *Gav's Song*, voicemails | long slow shots, restrained movement, one word at a time |
| `folk` | *We Bought the Hill* | woodcut and tapestry textures, hand-painted animation, the pipes finally get visuals |
| `ominous` | the offshore act | fog, silhouettes, slow push-ins, water |
| `archive` | history parts | a documentary archive (maps, deeds, ledgers, photos) that gets steadily stranger |

## 7. Lyrics as part of the scene

- **Text modes:** `chorus-sign` (painted on fences, signs and walls in the
  scene, with parallax), `slam` (a key phrase fills the screen), `document`
  (typed into the deed or a ledger), `voicemail` (handwritten, Ewan's),
  `spoken-title` (documentary lower-third for posh spoken intros), `quiet`
  (small, integrated, for verses).
- **The chorus keeps one treatment all song**, so it's recognisable, and
  changes over the song: clean on the first chorus, weathered on the second,
  fenced off or burnt on the last.
- Repeated phrases come back in the same style; key words can **break, burn,
  fade or get buried** (by fog, by the red line, by a stamp).

## 8. First milestone: *Deed to the Glen*

Why this song: it's finished (patch 7001 and the pedalboard master approved),
it's the Laird's song in *Natural Capital*, and its story is the one in the
plan: land sold off and fenced.

Rough storyboard, about 16 shots (timings come from `analysis.json`):

| Section | Shots | Idea |
|---|---|---|
| Spoken intro ("Raise a dram, gentlemen") | 2 | The hill at dawn as a perfect tourism poster; slow push-in to a lit lodge window; the Laird's toast as a documentary title. Crow arrives. |
| Verse 1 (the Clearances) | 3 | The poster turns into a painting: burnt roofs, an emigrant ship in fog; then back to the present, coaches of tourists paying £12 at the same spot. |
| Chorus 1 | 2 | "I OWN THE GLEN" stencilled on a new fence across the hill; the red line draws the estate boundary. |
| Verse 2 (salmon, distillery, wind, carbon) | 3 | Collage: salmon cages, a robot distillery, wind turbines stopped while money pours from them, a forest planted in perfect squares. The fund logo starts appearing. |
| Chorus 2 | 1 | The same fence, now with the deed nailed to it; the deed unfolds. |
| Bridge (the psalm) | 2 | Stillness: a chapel interior, the Laird as a stained-glass window, the congregation as silhouettes. Reactivity near zero. |
| Verse 3 (the buyout, the school) | 2 | The school becoming a boutique inn; the "twelve million by the end of the month" price as a stamped notice. |
| Final chorus and outro | 1–2 | The hill fully developed, the red line across everything, chorus text weathered and fenced in; "Rent's due Friday" stamped as the final card. The crow leaves. |

Definition of done: Reuben watches it and would post it, the lyrics are in
sync to within 1 frame, and re-running the render gives the same file.

## 9. Plan of work

1. **This document**, for Reuben to review (style, motifs, first song).
2. **Engine skeleton in the cloud:** analysis format, storyboard schema and
   validator, compositor with camera/parallax/fog/grain/text, a 30-second test
   render on placeholder art. No laptop time needed.
3. **Laptop, when the music jobs are idle** (through the laptop thread so the
   two never clash on the GPU): add `~/band/tools/img` (SD 1.5 fp32, cu126
   torch) and `~/band/tools/video` (CPU) venvs; benchmark one image; run
   Demucs 4-stem and analysis on *Deed to the Glen*; generate the plates.
4. **Look test:** a 30-second chorus at full quality plus a contact sheet. Reuben
   approves or redirects the style here, before the full render.
5. **Full render** of *Deed to the Glen*, then thumbnail and vertical cut.
6. Only then generalise: presets, rule-based fallback, the next song.

### How files move between the laptop and the cloud

Analysis and storyboards are small text files and go in the repo. Plates and
audio are too big for git: the laptop keeps the song folder as the source of
truth, and a cloud render needs the master, stems and plates copied over as one
archive. Simplest first version: render on the laptop at low priority when no
music job is running.

## 10. Risks

- **SD 1.5 quality ceiling.** Mitigation: the style hides it (painterly, print
  texture, fog), vector characters carry the faces, and the upscaler adds detail.
  SDXL through stable-diffusion.cpp is the fallback.
- **8 GB RAM.** The image venv and the compositor never run alongside a music job.
- **Heat.** The laptop already throttles under long jobs; renders go through
  `heavy.sh` on 8 threads, in chunks that can resume.
- **Originality.** Prompts describe techniques and subjects only, never artist
  names or existing works; every character, company and logo is fictional, as
  set in the album bible.
- **Licences.** Only commercially usable models and assets (Apache, MIT, BSD,
  OpenRAIL, OFL). Each checkpoint's licence is checked and noted when it's
  downloaded, with the repo licence and the weights licence checked separately.
