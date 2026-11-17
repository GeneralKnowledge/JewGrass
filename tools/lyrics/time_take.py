"""Timed lyrics for a take: when each of our written words is sung, as an LRC file next to the take.

  python time_take.py TAKE.mp3 [--vocals STEM.wav] [--words WHISPER.json] [--model medium.en]
  python time_take.py --all ROOT [--picks ~/band/picks.json] [--model medium.en]

For each take with lyrics in its sidecar (<take>.json, "lyrics") and no <take>.lrc yet:
  1. faster-whisper hears the words, with a time for each (on the GPU when there is one, else the CPU; a
     vocal stem works better than the full mix);
  2. its words are matched to the written lyrics (difflib over normalised words), so the screen shows our
     words and sections, not Whisper's spelling; words it missed get times spread across their line;
  3. <take>.lrc is written in enhanced LRC: "[00:12.30] <00:12.30>One <00:12.61>more <00:12.90>night".
     A line Whisper heard poorly (few matched words, or low confidence) keeps only its line time, so the phone
     lights the line rather than guessing words. Section tags go in at their first line's time.

The take server sends the .lrc to the phone (tools/player, "lrc"), where Soundscape highlights the line and
word being sung. --all skips takes that already have a .lrc, so it can be rerun; --picks does only each
song's picked take. On the laptop, run it through the lock like every heavy job (laptop/tools/asr.md).
"""
import argparse
import difflib
import json
import os
import re
import subprocess
import sys
import time

AUDIO_EXTS = (".mp3", ".wav", ".flac", ".m4a", ".ogg")
# A line gets word times only when at least this share of its words were heard, at this confidence.
MIN_MATCHED = 0.6
MIN_PROBABILITY = 0.5


# ------------------------------------------------------------------ matching (no model needed: tested on Linux)

def norm(word):
    return re.sub(r"[^a-z0-9']", "", word.lower().replace("’", "'")).strip("'")


def written_lines(lyrics):
    """[(section or None, [words])] for the sung lines, in order. A section tag applies to the next sung line;
    performance notes in brackets (commas or more than three words) are dropped."""
    out, section = [], None
    for raw in lyrics.splitlines():
        line = raw.strip()
        if not line:
            continue
        tag = re.fullmatch(r"\[([^\]]+)\]", line)
        if tag:
            name = tag.group(1).strip()
            if "," not in name and len(name.split()) <= 3:
                section = name
            continue
        words = line.split()
        if any(norm(w) for w in words):
            out.append((section, words))
            section = None
    return out


def heard_words(segments):
    """Whisper's words in order: [(normalised, start, end, probability)]."""
    out = []
    for s in segments:
        for w in s.get("words") or []:
            n = norm(w["word"])
            if n:
                out.append((n, float(w["start"]), float(w["end"]), float(w.get("probability", 1.0))))
    return out


def match(lyrics, segments):
    """The written lines with a start time per word: [{"section", "words": [[text, start or None, prob]],
    "start"}]. Times come from Whisper's matching words; the rest are spread within their line."""
    lines = written_lines(lyrics)
    flat = [(li, wi, norm(w)) for li, (_, words) in enumerate(lines) for wi, w in enumerate(words)]
    heard = heard_words(segments)
    timed = {}
    sm = difflib.SequenceMatcher(a=[f[2] for f in flat], b=[h[0] for h in heard], autojunk=False)
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            li, wi, _ = flat[a + k]
            _, start, _, prob = heard[b + k]
            timed[(li, wi)] = (start, prob)
    out = []
    for li, (section, words) in enumerate(lines):
        row = [[w, *timed.get((li, wi), (None, 0.0))] for wi, w in enumerate(words)]
        fill(row)
        out.append({"section": section, "words": row, "start": next((w[1] for w in row if w[1] is not None), None)})
    # lines nobody heard: between their neighbours
    known = [i for i, l in enumerate(out) if l["start"] is not None]
    for i, line in enumerate(out):
        if line["start"] is None:
            before = max((k for k in known if k < i), default=None)
            after = min((k for k in known if k > i), default=None)
            if before is not None and after is not None:
                a, b = out[before]["start"], out[after]["start"]
                line["start"] = a + (b - a) * (i - before) / (after - before)
            elif before is not None:
                line["start"] = out[before]["start"] + 2.0 * (i - before)
    # never earlier than the line before (a stray match can't jump back)
    last = 0.0
    for line in out:
        if line["start"] is not None:
            line["start"] = max(line["start"], last)
            last = line["start"]
    return out


def fill(row):
    """Spread untimed words between the timed ones around them, in place."""
    idx = [i for i, w in enumerate(row) if w[1] is not None]
    if not idx:
        return
    for i, w in enumerate(row):
        if w[1] is not None:
            continue
        before = max((k for k in idx if k < i), default=None)
        after = min((k for k in idx if k > i), default=None)
        if before is not None and after is not None:
            a, b = row[before][1], row[after][1]
            w[1] = a + (b - a) * (i - before) / (after - before)
        elif after is not None:
            w[1] = row[after][1]
        else:
            w[1] = row[before][1] + 0.3 * (i - before)


def stamp(seconds):
    seconds = max(0.0, seconds)
    m = int(seconds // 60)
    return f"{m:02d}:{seconds - 60 * m:05.2f}"


def to_lrc(matched):
    """Enhanced LRC: word stamps where the line was heard well, a line stamp otherwise. Lines with no time at
    all (nothing heard before them) are left out."""
    out = []
    for line in matched:
        if line["start"] is None:
            continue
        if line["section"]:
            out.append(f"[{stamp(line['start'])}] [{line['section']}]")
        words = line["words"]
        heard = [w for w in words if w[2] >= MIN_PROBABILITY]
        if words and len(heard) / len(words) >= MIN_MATCHED:
            text = " ".join(f"<{stamp(w[1])}>{w[0]}" for w in words)
        else:
            text = " ".join(w[0] for w in words)
        out.append(f"[{stamp(line['start'])}] {text}")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ hearing (faster-whisper)

_model = None


def transcribe(audio, lyrics, model_name):
    global _model
    import numpy as np
    from faster_whisper import WhisperModel
    if _model is None:
        try:
            import ctranslate2
            cuda = ctranslate2.get_cuda_device_count() > 0
        except Exception:
            cuda = False
        _model = WhisperModel(model_name, device="cuda" if cuda else "cpu",
                              compute_type="float16" if cuda else "int8", cpu_threads=6)
    prompt = " ".join(l for l in lyrics.splitlines() if l.strip() and not l.strip().startswith("["))[:200]
    # ffmpeg decodes (faster-whisper's own decoder breaks with PyAV 19, see laptop/tools/asr.md)
    pcm = np.frombuffer(subprocess.run(["ffmpeg", "-loglevel", "error", "-i", audio, "-ac", "1", "-ar", "16000",
                                        "-f", "f32le", "-"], check=True, capture_output=True).stdout, dtype=np.float32)
    segs, _ = _model.transcribe(pcm, language="en", beam_size=5, word_timestamps=True, vad_filter=False,
                                condition_on_previous_text=False, initial_prompt=prompt)
    return [{"start": s.start, "end": s.end, "text": s.text,
             "words": [{"start": w.start, "end": w.end, "word": w.word, "probability": w.probability}
                       for w in (s.words or [])]} for s in segs]


def sidecar(audio):
    side = os.path.splitext(audio)[0] + ".json"
    try:
        meta = json.load(open(side))
        return meta if isinstance(meta, dict) else {}
    except (OSError, ValueError):
        return {}


def time_take(audio, model_name, vocals=None, words=None):
    """Writes <take>.lrc; returns its path, or None when the take has no lyrics."""
    lyrics = (sidecar(audio).get("lyrics") or "").strip()
    if not lyrics:
        return None
    segments = json.load(open(words)) if words else transcribe(vocals or audio, lyrics, model_name)
    out = os.path.splitext(audio)[0] + ".lrc"
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(to_lrc(match(lyrics, segments)))
    os.replace(tmp, out)
    return out


def takes(root, picks=None):
    """Every take under root (one audio file per take: mp3 first) with lyrics and no .lrc yet."""
    chosen = None
    if picks:
        # {"songs": {slug: {"pick": take id (its path under ~/band, no extension)}}}
        songs = json.load(open(os.path.expanduser(picks))).get("songs", {})
        chosen = {v.get("pick") for v in songs.values() if isinstance(v, dict) and v.get("pick")}
    seen = set()
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and not d.endswith(".venv"))
        for f in sorted(files, key=lambda f: (os.path.splitext(f)[0], AUDIO_EXTS.index(os.path.splitext(f)[1].lower())
                                              if os.path.splitext(f)[1].lower() in AUDIO_EXTS else 99)):
            base, ext = os.path.splitext(os.path.join(dirpath, f))
            if ext.lower() not in AUDIO_EXTS or base in seen:
                continue
            seen.add(base)
            if os.path.exists(base + ".lrc") or not (sidecar(base + ext).get("lyrics") or "").strip():
                continue
            if chosen is not None and not any(base == c or base.endswith("/" + c) for c in chosen):
                continue
            yield base + ext


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("take", nargs="?")
    p.add_argument("--all", metavar="ROOT")
    p.add_argument("--picks", help="only each song's picked take (the take server's picks.json)")
    p.add_argument("--vocals", help="a vocal stem of the take, heard instead of the mix")
    p.add_argument("--words", help="Whisper's words already made (whisper_words.py output)")
    p.add_argument("--model", default="medium.en")
    a = p.parse_args()
    if a.all:
        todo = list(takes(a.all, a.picks))
        print(f"{len(todo)} takes to time", flush=True)
        deadline = float(os.environ.get("BAND_DEADLINE", "0") or 0)
        for i, t in enumerate(todo, 1):
            if deadline and time.time() > deadline:
                print("time budget reached; rerun to time the rest", flush=True)
                break
            t0 = time.time()
            try:
                out = time_take(t, a.model)
                print(f"[{i}/{len(todo)}] {out} in {time.time() - t0:.0f} s", flush=True)
            except Exception as e:  # one bad take shouldn't end the batch
                print(f"[{i}/{len(todo)}] FAILED {t}: {e!r}", flush=True)
    elif a.take:
        print(time_take(a.take, a.model, a.vocals, a.words) or "no lyrics in the sidecar")
    else:
        p.error("give a take, or --all ROOT")


if __name__ == "__main__":
    sys.exit(main())
