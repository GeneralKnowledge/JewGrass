# Local generation plan: experiment batches 1b–4

Working doc for the band's sound and vocal discovery, run on local open models
instead of Suno. Everything here is original: no artist names, song titles or
lyrics from the reference catalogue go into any prompt.

**Status:** Batch 1 hasn't been run yet. Sections marked **ADJUST AFTER BATCH 1**
are written so we can swap in whatever wins, without rewriting the rest.

---

## 1. The local toolchain

| Job | Tool | Why |
|---|---|---|
| Full songs with vocals | **ACE-Step 1.5** (main engine) | MIT licence for code *and* weights, generated output usable commercially, runs on under 4 GB VRAM, up to 10-minute songs, reference audio, covers, repaint, LoRA training |
| Stem separation | **Demucs** (htdemucs) or ACE-Step's own extraction | Pull vocals/drums/bass apart to judge, fix or re-use |
| Persistent band voices | **RVC** (e.g. via Applio) trained on our *own* generated takes | The local answer to "three guaranteed persistent voices" |
| Textures and one-shots | Stable Audio Open (optional) | Glitch beds, metallic hits, field noise. Check its community licence before any commercial use |
| Assembly | Any DAW (Reaper is cheap and fine) | Stitch sections, add glitch edits by hand, final rough-but-intentional mix |

**Not recommended as the main engine:** YuE2. It is good, but it needs a 24 GB
NVIDIA card and its weights are CC BY-NC (non-commercial without separate
permission), which conflicts with an eventual release. MusicGen weights are
also non-commercial and it has no vocals.

### Install ACE-Step 1.5

```bash
# 1. install uv (https://docs.astral.sh/uv/), then:
git clone https://github.com/ace-step/ACE-Step-1.5.git
cd ACE-Step-1.5
uv sync
uv run acestep          # Gradio UI at http://localhost:7860
# optional: uv run acestep-api   (REST API at http://localhost:8001)
```

Models download on first run. Python 3.11–3.12. Works on NVIDIA (Windows/Linux),
AMD ROCm, Apple Silicon (MLX) and CPU (slow).

### Pick settings for your GPU

| VRAM | Model to use | Language model (LM) |
|---|---|---|
| ≤ 6 GB | 2B **turbo** | off (quantised, CPU offload) |
| 6–8 GB | 2B turbo | 0.6B |
| 8–16 GB | 2B turbo for sketching, 2B **sft** for keepers | 0.6B or 1.7B |
| 16–24 GB | **XL** turbo / sft | 1.7B |
| ≥ 24 GB | XL sft | 4B |

Mapping to the old Suno plan: **turbo = v6-mini** (fast, cheap sketching, 8 steps,
no guidance control), **sft = v6** (slower, 50 steps, supports guidance; use for
anything we might keep), **base = v6-wild** (less tuned, wider variety; also the
only one with "complete/extend").

### Running on our machine: Dell G3 17, GTX 1060 6 GB, 8 GB RAM, Ubuntu Server (headless)

**Verdict: borderline. Worth a one-evening test, but plan for a fallback.**

Three problems, in order of severity:

1. **PyTorch build.** ACE-Step pins `torch==2.10.0+cu128` on Linux. The CUDA 12.8
   builds of PyTorch no longer include kernels for Pascal cards (GTX 10-series,
   compute capability 6.1), so out of the box it will likely fail with
   "no kernel image is available". Fix: swap in the CUDA 12.6 build (below).
2. **8 GB system RAM.** The ≤ 6 GB VRAM tier works by offloading parts of the
   model to system RAM. 8 GB is tight for that, so add swap. Expect it to be slow
   when it swaps.
3. **Pascal is slow at half precision** and has no native bf16. Use the smallest
   setup (2B **turbo**, LM **off**) and expect minutes per song rather than
   seconds. If it errors about bf16, look for a dtype/precision option in the UI or
   CLI help and set fp32 or fp16.

**Setup**

```bash
# 1. NVIDIA driver. The 580 branch is the last one that supports Pascal.
sudo apt install nvidia-headless-580 nvidia-utils-580
sudo reboot
nvidia-smi            # should show the GTX 1060 and its VRAM

# 2. Keep running with the lid closed
sudo sed -i 's/^#\?HandleLidSwitch=.*/HandleLidSwitch=ignore/; s/^#\?HandleLidSwitchExternalPower=.*/HandleLidSwitchExternalPower=ignore/' /etc/systemd/logind.conf
sudo systemctl restart systemd-logind

# 3. 16 GB of swap to cover the 8 GB RAM
sudo fallocate -l 16G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# 4. ACE-Step, then replace PyTorch with the Pascal-compatible CUDA 12.6 build
git clone https://github.com/ace-step/ACE-Step-1.5.git && cd ACE-Step-1.5
uv sync
uv pip install --reinstall torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu126

# 5. Check the GPU actually works (should list sm_61 and print a tensor)
uv run --no-sync python -c "import torch; print(torch.cuda.get_arch_list()); print(torch.ones(1).cuda()+1)"

# 6. Launch. --no-sync stops uv from putting the cu128 build back.
uv run --no-sync acestep
```

If step 4 says there is no cu126 build of 2.10.0, try the newest version listed at
`https://download.pytorch.org/whl/cu126/torch/` that still works with the repo.

**Using it headless:** from your other computer, tunnel the UI over SSH instead of
exposing it to the network:

```bash
ssh -L 7860:localhost:7860 you@the-laptop
# then open http://localhost:7860 in your browser
```

Run long batches inside `tmux` so they survive the SSH session dropping. Watch
temperatures with `watch -n 5 nvidia-smi`. With the lid shut, keep the laptop
raised so the vents underneath get air.

**What this machine can and can't do**

| Task | On the G3? |
|---|---|
| Batch 1–4 generation on 2B turbo, 120–150 s songs | Probably, slowly |
| sft/base models, the language model, XL | Unlikely |
| Demucs stem separation | Yes (slow but fine) |
| RVC voice training and conversion | Yes, it fits in 6 GB; training takes hours |
| ACE-Step LoRA training (Phase 5) | No, rent a GPU for that |

**Fallback if it won't run or is unbearably slow:** rent a cloud GPU by the hour
(a 24 GB card is enough for everything in this plan) and run the exact same setup
there. Seeds, prompts and the MIT licence carry over unchanged, so results stay
comparable with anything made on the laptop.

### The big local advantage: fixed seeds

Suno can't hold randomness still; ACE-Step can. For every controlled comparison:

- Pick **two seeds** (e.g. `1111` and `2222`) and use them for every prompt in
  the batch. That gives the "two versions" per experiment, and differences
  between prompts are now caused by the prompt, not luck.
- Keep duration, BPM and key fixed within a batch.
- Only once a batch is judged do we run extra random seeds on the winner.

**Fixed parameters for Batches 1–2:** duration 150 s, BPM 140 (feels 70 half-time),
key F minor, time signature 4/4, seeds 1111 and 2222, same model and LM.

---

## 2. Prompt format

ACE-Step takes two text inputs:

- **Caption / tags:** comma-separated descriptors (genre, instruments, mix,
  vocal). Put the most important ones first.
- **Lyrics:** with section tags on their own lines, e.g. `[intro]`, `[verse]`,
  `[chorus]`, `[bridge]`, `[outro]`, `[instrumental]`.

Before the first run, open one of the built-in examples in the Gradio UI and
copy its exact tag style. If it uses lower-case tags or a different separator,
match that everywhere below. Bracketed performance notes like
`[verse - fast, distorted]` may or may not be honoured; treat them as a test,
not a guarantee.

There is no "exclude styles" box, so steer away from polish with explicit
mix words (`raw`, `lo-fi`, `saturated`, `dry vocal`, `unpolished`) and, on sft/base
models, a negative prompt if your UI exposes one:

> **Negative (if available):** country pop, pop country, glossy, polished radio mix, clean vocals, EDM drop, autotune pop, bright acoustic strumming, stadium reverb

---

## 3. Batch 1 (as planned, adapted)

Run prompts A–D from the original plan as captions, with the
*Dirt Under the Nails* lyric (Section 8), 2 seeds each = **8 outputs**.
Add one tag to all four so they share a mix character:
`raw underground mix, dry close vocal`.

Then score with the sheet in Section 7 and fill in the **CORE** block:

```
CORE = <the 6–10 tags from the winning caption that you'd keep no matter what>
```

---

## 4. Batch 1b: the experimental production batch

Same lyric, same seeds, same fixed parameters as Batch 1. Each prompt pushes
one variable much harder than Batch 1 did. Six prompts × 2 seeds = **12 outputs**.

Each caption is written to stand alone. Once Batch 1 has a winner, prefix it
with CORE and drop anything that contradicts it.

### E1 — Blown-out low end (matrix C: distorted 808s)

> heavy southern trap, clipped distorted 808 bass, overdriven sub that breaks up on every hit, banjo riff buried under the bass, sparse trap drums, rattling hi-hats, tape saturation, rough low male rap vocal, southern drawl, gang chant chorus, dark rural atmosphere, raw underground mix

*Learn:* how far the bass can be pushed before the banjo and vocal disappear.

### E2 — Broken machine (matrix D: glitch)

> experimental glitch trap, chopped and stuttering banjo samples, bitcrushed acoustic guitar, granular vocal chops, sudden dropouts and tape stops, skipping hi-hats, metallic percussion, deep 808, menacing low male rap, southern drawl, digital noise between phrases, raw underground mix

Add to the lyric for this one only: put `[instrumental]` before Verse 2 and a
spoken line `(signal's breaking up)` at the end of Verse 1.

*Learn:* whether glitch can be a band signature or just an effect.

### E3 — Beat switch (structure)

> dark country trap that switches beats mid-song, first half slow half-time trap with dobro slide and 808, second half double-time bluegrass breakdown with frantic banjo and clapping, gritty male rap, southern drawl, chant chorus, raw underground mix

Lyric change: label Verse 2 `[verse - double time]`.

*Learn:* whether the model can honour a mid-song switch, which would give us a
signature move without hand editing.

### E4 — Porch tape (rough character)

> lo-fi southern trap recorded on a porch, field recording ambience, crickets and screen door, out-of-tune acoustic guitar, foot stomps and knee slaps as percussion, distorted 808 underneath, cassette hiss, warped tape pitch, gravelly male rap, close dry vocal, unpolished, raw

*Learn:* whether "rough" reads as character or just as bad audio.

### E5 — Swamp phonk crossover

> dark swamp phonk fused with outlaw country, cowbell melody replaced by banjo, heavy distorted 808, slowed and pitched-down vocal chops, drifting dobro slide, eerie organ drone, menacing low male rap, southern drawl, hypnotic repetitive hook, raw saturated mix

*Learn:* whether a slower, hypnotic lane suits the band better than aggression.

### E6 — Wildcard (run on the **base** model, higher temperature/guidance variety)

> industrial bluegrass trap, banjo played through a distortion pedal like a lead synth, fiddle through a vocoder, junkyard percussion, hammer on sheet metal, sub bass drones, chaotic arrangement, shouted male rap, southern drawl, call and response chant, abrasive underground mix

*Learn:* unexpected combinations. We aren't looking for a finished track here,
just one or two ideas worth stealing.

### What to do with Batch 1b

- Keep **at most two** ideas from it. Add their key tags to CORE.
- If E2 or E3 wins, note whether the effect would be easier to do by hand in
  the DAW. If yes, keep the cleaner generation and add the effect ourselves.

### ADJUST AFTER BATCH 1 — which experiments to prioritise

| Batch 1 result | Run first | Skip or deprioritise |
|---|---|---|
| A (core country trap) won | E1, E3 | E6 |
| B (industrial glitch) won | E2, E6 | E4 |
| C (Appalachian) won | E3, E4 | E5 |
| D (melodic Americana) won | E5, E4 | E1 |
| Everything sounded too polished | E4, then rerun the winner with the Section 2 negative prompt | — |
| Everything sounded like country pop | Move trap/bass tags to the front of the caption, cut "melodic" and "haunting" | E5 |
| Banjo vanished every time | Put `banjo` in the first three tags and add `[instrumental]` banjo break | E1 |

---

## 5. Batches 2–4: the band members

### How we make the voices persist locally

Prompts alone won't keep a voice identical across songs. The local recipe:

1. **Find the voice** with caption descriptions (Batch 2 / 3 below).
2. **Lock the vibe** by using the best take as ACE-Step **reference audio** for later generations.
3. **Lock the actual timbre** with **RVC**: separate the vocal from our 3–5 best takes
   of that member (Demucs), train an RVC model on those clean vocals (~10 minutes of
   audio is enough to start), and run every future take of that member through it.
4. Later (Phase 5), train an ACE-Step **LoRA** on our own finished tracks. The
   project reports about 8 songs, around 1 hour on a 12 GB card. This is the local
   equivalent of Suno Custom Models.

Only ever train RVC on vocals we generated or recorded ourselves (or a singer
who has agreed in writing), never on a real artist's voice.

### Batch 2 — Find the Frontman

Caption = **CORE + one vocal block**. Same *Dirt Under the Nails* lyric, same seeds.
4 prompts × 2 seeds = **8 outputs**. Put the vocal block **first** in the caption.

| | Vocal block | What it tests |
|---|---|---|
| V1 | low gravelly male baritone rap, heavy southern drawl, laid-back behind-the-beat flow, dry close vocal | The plan's baseline |
| V2 | rough shouted male rap, barking delivery, strained throat, aggressive drawl, gang vocals doubling the ends of lines | Pure aggression |
| V3 | gravelly male talk-singing, half-rapped half-sung, southern drawl, melodic edges on the hook, weary storyteller | Range for slower songs |
| V4 | deep menacing male rap, quiet and controlled, almost whispered verses that explode into a shouted chorus, heavy drawl, low ad-libs | Dynamics |

**Pick the Frontman** by asking: could this voice carry five songs without
getting tiring? Then save the two best takes as reference audio and start the
RVC dataset.

### Batch 3 — Find the Wildcard and the Ghost

Each member gets their own new song so they aren't judged on the Frontman's
material. 2 prompts × 2 seeds per member = **8 outputs** total.

**Wildcard** (song: *Rattlesnake Sunday*, Section 8)

| | Caption (append CORE production tags) |
|---|---|
| W1 | fast raspy male rap, higher register, erratic triplet flow, abrupt stops, manic laughter ad-libs, distorted bitcrushed vocal, dark humour, glitch trap, banjo stabs, distorted 808 |
| W2 | nasal sneering male rap, rapid-fire delivery, stuttering vocal edits, megaphone distortion, confrontational, chaotic glitch trap, chopped banjo, metallic percussion, heavy 808 |

**Ghost** (song: *Creek Don't Remember*, Section 8)

| | Caption |
|---|---|
| G1 | haunting female alto, Appalachian folk inflection, high lonesome tone, sparse vibrato, close intimate vocal, layered harmonies on the chorus, mournful fiddle, fingerpicked acoustic guitar, restrained trap drums, distorted low 808 entering late, night-time atmosphere |
| G2 | haunting male tenor, high lonesome Appalachian tone, cracked and fragile falsetto, gospel-tinged harmonies, banjo drone, reverse reverb, slow half-time trap, distorted sub bass, eerie cinematic atmosphere |

G1 vs G2 also decides the Ghost's gender. The original plan leaned female, which
gives the most contrast with two male rappers; G2 is there in case a male
falsetto feels more like the band.

**Distinctness test:** play the Frontman, Wildcard and Ghost picks back to back
with your eyes shut. If any two could be the same person, rerun that member with
a more extreme vocal block before moving on.

### Batch 4 — The band together

Two collaborations. Caption = CORE + a short description of *both* voices, and
every lyric section labels who sings it. 2 songs × 2 seeds = **4 outputs**.
If the model blends the voices together, fall back to the DAW route: generate
each member's sections separately with their own caption and reference audio,
run each through that member's RVC model, and splice.

**Collab 1 — Frontman + Wildcard** (song: *Gravel and Static*)

> two contrasting male rappers, first voice low gravelly southern drawl and slow, second voice fast raspy and distorted, call and response chant chorus, aggressive country trap, banjo riff, distorted 808, glitch edits, raw underground mix

**Collab 2 — Frontman + Ghost** (song: *Porchlight Funeral*)

> dark outlaw Americana trap, haunting female country vocal on the chorus, low gravelly male rap verses with southern drawl, mournful fiddle, banjo drone, distorted 808, restrained trap drums, layered harmonies, cinematic night-time atmosphere

(Swap "female" for "male tenor" if G2 won.)

---

## 6. Generation budget

| Batch | Prompts | Seeds | Outputs |
|---|---|---|---|
| 1 | 4 | 2 | 8 |
| 1b | 6 | 2 | 12 |
| 2 | 4 | 2 | 8 |
| 3 | 4 | 2 | 8 |
| 4 | 2 | 2 | 4 |
| **Total** | | | **40** |

Locally this costs nothing but time: on turbo, a full song takes seconds on a
modern card and around a minute or two on a small one.

---

## 7. Scoring

Use the original weighted sheet (Genre identity 25, Production and bass 20,
Vocal character 20, Country instrumentation 15, Originality 10, Hook 10).
Above 80 is worth developing.

For Batches 2–4, add two pass/fail checks:

- **Consistency:** do seed 1111 and seed 2222 sound like the same person?
- **Distinctness:** could this voice be confused with another band member?

Log every output in [`experiment-log.md`](experiment-log.md) so results come back
here comparable, with seed and model recorded.

---

## 8. Lyrics

All original working lyrics. Bracketed tags go in the lyrics box exactly as shown.

### Dirt Under the Nails (Batches 1, 1b, 2 — unchanged from the plan)

```
[intro]
(spoken, distorted)
Ain't much left out here.
Just the dirt, the dark, and the noise.

[chorus]
Got dirt under my nails, got smoke in my lungs
Got a long black road and a loaded-up drum
Ain't no gold in the ground, ain't no silver spoon
Just a busted-up life underneath that moon

[verse]
Gravel in the teeth, got a rusted-out gate
Whole town sleeping but we're running too late
Old boots kicking up the dust in the lane
Radio screaming through the windowpane

[instrumental]
(banjo and distorted 808)

[verse]
No clean hands, no clean slate
No front door, no escape
Everything earned, everything bent
Everything broken but the money well spent

[chorus]
Got dirt under my nails, got smoke in my lungs
Got a long black road and a loaded-up drum
Ain't no gold in the ground, ain't no silver spoon
Just a busted-up life underneath that moon
```

### Rattlesnake Sunday (Wildcard, Batch 3)

```
[intro]
(spoken, bitcrushed)
Church let out early.
Somebody brought the snake.

[verse]
Preacher got a twitch and the choir got a cough
Uncle's in the back pew taking his boots off
I ain't saying I did it, I'm just saying I was there
Got a rattle in my pocket and a briar in my hair
Mama said behave, I said ma'am I'll try
Then I winked at the deacon and he started to cry
Whole county calls me trouble like it's something brand new
I been trouble since the stork took a wrong turn through

[chorus]
Rattle rattle Sunday, shake it off the pew
Rattle rattle Sunday, who invited you
Rattle rattle Sunday, bite and let it go
Ain't nobody holy when the snake's in the show

[verse]
Stop.
Breathe.
Count to three.
Nah, I skipped two, it's a habit with me
Sheriff's at the door with a sandwich and a frown
Says son, you're the reason they're repainting this town
I said sir, I'm an artist, and the barn was my page
Now I'm signing autographs from a chicken-wire cage

[chorus]
Rattle rattle Sunday, shake it off the pew
Rattle rattle Sunday, who invited you
Rattle rattle Sunday, bite and let it go
Ain't nobody holy when the snake's in the show

[outro]
(laughing, glitching out)
Same time next week.
```

### Creek Don't Remember (Ghost, Batch 3)

```
[intro]
(lone fiddle, vinyl crackle)

[verse]
I washed my name in the cold creek water
Left it on a stone for the morning to take
Daddy's old truck rusting down by the culvert
I still got the key, I still got the ache

[pre-chorus]
And the hills keep the secrets
That the people let go

[chorus]
Oh, the creek don't remember
Who went under, who came home
It just runs through the holler
Singing low, singing low
Carry me down where the sycamores bend
The creek don't remember, so I'll learn how to forget

[verse]
Sunday dress in a cedar chest
Mama's voice on a tape I can't play
Somebody's porch light still burning at midnight
For a boy who ain't coming back this way

[bridge]
(808 drops in, half-time, distorted)
Low, low, the water goes
Where it's going, nobody knows

[chorus]
Oh, the creek don't remember
Who went under, who came home
It just runs through the holler
Singing low, singing low
Carry me down where the sycamores bend
The creek don't remember, so I'll learn how to forget

[outro]
(humming, tape stop)
```

### Gravel and Static (Collab 1, Batch 4)

```
[intro - Frontman, spoken low]
Turn it up till it hurts.

[chorus - both, call and response]
Gravel in the engine
STATIC IN THE WIRE
Mud on the Bible
GASOLINE AND FIRE
Two boys from nowhere
LOUDER THAN THE CHOIR
Gravel in the engine
STATIC IN THE WIRE

[verse - Frontman, slow, low drawl]
I was raised on a fence line, raised on a grudge
Raised where the river don't care who you judge
Slow walk, low talk, heavy in the chest
Every word I say, I already said best
You can hear me coming from the county line
Bass in the tailgate shaking the pines

[verse - Wildcard, fast, distorted]
And I'm the other thing, the twitch in the wire
Kid with a match and a heart full of tires
Count me in, count me out, count me twice
Cut the power to the barn and I'll rap in the ice
He the thunder, I'm the stutter in the speaker
Every time I glitch, your Sunday gets weaker

[chorus - both, call and response]
Gravel in the engine
STATIC IN THE WIRE
Mud on the Bible
GASOLINE AND FIRE
Two boys from nowhere
LOUDER THAN THE CHOIR
Gravel in the engine
STATIC IN THE WIRE

[bridge - trading lines]
(Frontman) Slow.
(Wildcard) Fast.
(Frontman) Built to last.
(Wildcard) Built to crash.
```

### Porchlight Funeral (Collab 2, Batch 4)

```
[intro]
(fiddle, crickets, low hum)

[chorus - Ghost, haunting]
Leave the porchlight on for the ones who ran
Leave the porchlight on, I don't understand
How a house so small held a ghost so loud
Leave the porchlight on, burn it down

[verse - Frontman, low drawl]
Funeral was Tuesday, rain came Wednesday
Cousins in the kitchen arguing who gets the shotgun
Nobody cried till the dog did
Then the whole room cracked like a cold lid
I stood out back by the burn pile
Smoking his brand just to keep him a while

[chorus - Ghost, haunting]
Leave the porchlight on for the ones who ran
Leave the porchlight on, I don't understand
How a house so small held a ghost so loud
Leave the porchlight on, burn it down

[verse - Frontman, building]
He left me the truck and a debt and a Bible
A pocketknife, a rifle, and a list of who's liable
I don't want the money, I want the noise back
His boots on the floor and the screen door slap

[bridge - Ghost and Frontman together]
Burn it down, burn it down
Light it up so he can find his way round

[outro - Ghost, alone]
Leave the porchlight on.
```

Before any release, check every title against the reference catalogue and a
general search so we don't collide with an existing song.
