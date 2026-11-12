# Local generation plan (active again)

**Status: back on as of 2026-11-12.** The Suno runs (T4 to T10) gave us a sound
and a set of songs we like. This plan moves production onto the GTX 1060 laptop
with ACE-Step 1.5. Start with the **first smoke test** below: it answers
"can this laptop make a song at all?" in one evening, before we invest more.

Everything here is original: no artist names, song titles or lyrics from the
reference catalogue go into any prompt.

## First smoke test: Your Landlord on the laptop

Do the "Running on our machine" setup in Section 1 first, then in the Gradio UI:

1. Model: 2B **turbo**, language model **off**.
2. Duration **60 s** for the first try (full songs come later). BPM 140, seed 1111.
3. Paste the caption and lyrics below, generate, and note how long it took.
4. If it works, run the same thing with seed 2222, then try 180 s with the full
   song from [`lyrics.md`](lyrics.md) (strip the descriptive lines such as
   `[Sneering raspy male rap]` and keep only the section tags).

**Caption**

```
dark country trap, satirical villain rap, raspy male vocal, Scottish accent, banjo, distorted 808 bass, hard trap drums, rapid hi-hats, glitch, stuttering vocal chops, fiddle, bagpipe drone, heavy guitars on chorus, gang chant chorus, raw lo-fi mix
```

**Lyrics** (verse 1 and chorus only)

```
[verse]
I own the flat that you rent and the bank with your loan
The water in your tap and the signal on your phone
Sewage in the river? That's a dividend, son
Fined a million quid and I made a hundred and one
Black mould on the ceiling, black mould on the cot
Letting agent says ventilate, and that's all that you got
Went to school with the Chancellor, shared a wee dorm
Now the rules get written in my old uniform

[chorus]
I'm your landlord, landlord, keys in my hand
Every brick in the city, every acre of land
You can graft all your life and you'll never own a stair
'Cause my grandad's grandad already got there
```

**What to report back:** whether it ran, how long 60 s took, peak VRAM from
`nvidia-smi`, any error text, and whether the result sounds anything like the
Suno version. Expect it to be rougher than Suno; the question here is speed and
stability, not quality.

**If it won't run at all:** paste the error. The usual suspects, in order: the
PyTorch build (Section 1, step 4), running out of RAM (check the swap file),
or bf16 not being supported (look for a precision setting).

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

See [`lyrics.md`](lyrics.md).
