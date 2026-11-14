#!/usr/bin/env python3
"""arrangement.json -> a DAW project package.

    python3 tools/daw/export.py build/daw/gavs-song/arrangement.json OUT_DIR [--preview]

OUT_DIR/
  <Title>.rpp                 REAPER project: one track per part, MIDI inline and editable,
                              section regions, tempo/meter, reference audio + stems, a Lyrics
                              track (one empty item per section, lyrics in the item notes)
  <Title> (all tracks).mid    Standard MIDI File type 1: tempo, meter, key, section markers,
                              lyrics, one track per part. Opens in any DAW (Ardour, LMMS,
                              Ableton, Bitwig, Logic, MuseScore...)
  midi/NN <Track>.mid         one file per part, same timing
  audio/                      reference mix and stems (copied, only if the arrangement has them)
  preview/midi-preview.mp3    the MIDI played through a General MIDI SoundFont (with --preview)
  lyrics.txt, sections.csv, arrangement.json, README.txt

Positions: everything is shifted by whole "lead-in" bars when the reference
audio has material before beat 0, so the audio stays aligned and the bar grid
stays on the downbeats.
"""
import csv
import math
import os
import re
import shutil
import sys

import mido

sys.path.insert(0, os.path.dirname(__file__))
import arrangement as A  # noqa: E402

PPQ = 960


def safe(name):
    return re.sub(r"[^\w\- ()']+", "", name).strip()


def lead_in_beats(arr):
    off = arr.get("audio_offset_seconds") or 0.0
    if off <= 0:
        return 0.0
    bar_sec = A.beat_to_sec(arr, A.beats_per_bar(arr))
    return math.ceil(off / bar_sec) * A.beats_per_bar(arr)


def channel_for(track, idx):
    if track["instrument"].get("drums"):
        return 9
    ch = idx % 15
    return ch if ch < 9 else ch + 1


def abs_notes(track):
    for c in track.get("clips", []):
        for s, d, p, v in c["notes"]:
            yield c["start"] + s, d, int(p), int(v)


# ---------------------------------------------------------------- Standard MIDI

def _key_sig(key):
    if not key:
        return None
    m = re.match(r"([A-G][#b]?)\s*(major|minor)", key)
    if not m:
        return None
    return m.group(1) + ("m" if m.group(2) == "minor" else "")


def conductor(arr, lead):
    t = mido.MidiTrack()
    t.append(mido.MetaMessage("track_name", name=arr["title"], time=0))
    t.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(arr["tempo_bpm"]), time=0))
    num, den = arr["time_signature"]
    t.append(mido.MetaMessage("time_signature", numerator=num, denominator=den, time=0))
    ks = _key_sig(arr.get("key"))
    if ks:
        t.append(mido.MetaMessage("key_signature", key=ks, time=0))
    events = []
    for s in arr["sections"]:
        b = lead + A.bar_to_beat(arr, s["start_bar"])
        events.append((b, mido.MetaMessage("marker", text=s["name"])))
        lines = [ln for ln in (s.get("lyrics") or "").splitlines() if ln.strip()]
        span = A.bar_to_beat(arr, s["bars"])
        for i, ln in enumerate(lines):          # spread evenly: a reading guide, not sung timing
            events.append((b + span * i / len(lines), mido.MetaMessage("lyrics", text=ln.strip() + "\r")))
    _append_sorted(t, events)
    return t


def _append_sorted(track, events, end_beat=None):
    events.sort(key=lambda e: (e[0], 0 if getattr(e[1], "type", "") == "note_off" else 1))
    now = 0
    for beat, msg in events:
        tick = int(round(beat * PPQ))
        track.append(msg.copy(time=tick - now))
        now = tick
    track.append(mido.MetaMessage("end_of_track", time=0 if end_beat is None else max(0, int(end_beat * PPQ) - now)))


def part_track(arr, track, idx, lead):
    ch = channel_for(track, idx)
    t = mido.MidiTrack()
    t.append(mido.MetaMessage("track_name", name=track["name"], time=0))
    t.append(mido.MetaMessage("instrument_name", name=track["instrument"]["name"], time=0))
    if not track["instrument"].get("drums"):
        t.append(mido.Message("program_change", channel=ch, program=track["instrument"]["program"], time=0))
    ev = []
    for s, d, p, v in abs_notes(track):
        ev.append((lead + s, mido.Message("note_on", channel=ch, note=p, velocity=v)))
        ev.append((lead + s + d, mido.Message("note_off", channel=ch, note=p, velocity=0)))
    _append_sorted(t, ev)
    return t


def write_midi(arr, out_dir, lead):
    parts = [t for t in arr["tracks"] if t.get("clips")]
    full = mido.MidiFile(type=1, ticks_per_beat=PPQ)
    full.tracks.append(conductor(arr, lead))
    os.makedirs(os.path.join(out_dir, "midi"), exist_ok=True)
    for i, tr in enumerate(parts):
        full.tracks.append(part_track(arr, tr, i, lead))
        one = mido.MidiFile(type=1, ticks_per_beat=PPQ)
        one.tracks.append(conductor(arr, lead))
        one.tracks.append(part_track(arr, tr, i, lead))
        one.save(os.path.join(out_dir, "midi", f"{i + 1:02d} {safe(tr['name'])}.mid"))
    full.save(os.path.join(out_dir, f"{safe(arr['title'])} (all tracks).mid"))


# ---------------------------------------------------------------- REAPER

def _q(s):
    s = str(s).replace('"', "'")
    return f'"{s}"'


def _midi_source(arr, clip, ch, program, drums):
    """Inline MIDI for a REAPER item. E lines: delta ticks (decimal), then status/data bytes in hex."""
    ev = []
    if not drums:
        ev.append((0, 0, f"{0xC0 | ch:02x} {program:02x} 00"))
    for s, d, p, v in clip["notes"]:
        ev.append((int(round(s * PPQ)), 1, f"{0x90 | ch:02x} {p:02x} {v:02x}"))
        ev.append((int(round((s + d) * PPQ)), 0, f"{0x80 | ch:02x} {p:02x} 00"))
    ev.sort()
    lines, now = [], 0
    for tick, _, data in ev:
        lines.append(f"E {tick - now} {data}")
        now = tick
    end = int(round(clip["length"] * PPQ))
    lines.append(f"E {max(0, end - now)} {0xB0 | ch:02x} 7b 00")
    return ["<SOURCE MIDI", "HASDATA 1 960 QN"] + lines + [">"]


def _item(pos, length, name, body, extra=()):
    return ["<ITEM", f"POSITION {pos:.6f}", f"LENGTH {length:.6f}", "LOOP 0", f"NAME {_q(name)}", *extra, *body, ">"]


def _track(name, items, mute=False, vol=1.0, extra=()):
    return ["<TRACK", f"NAME {_q(name)}", f"VOLPAN {vol:.4f} 0 -1 -1 1", f"MUTESOLO {int(mute)} 0 0", *extra] + \
        [ln for it in items for ln in it] + [">"]


def _audio_source(path):
    kind = {".wav": "WAVE", ".mp3": "MP3", ".flac": "FLAC", ".ogg": "VORBIS"}[os.path.splitext(path)[1].lower()]
    return [f"<SOURCE {kind}", f"FILE {_q(path)}", ">"]


def write_rpp(arr, out_dir, lead, audio_files):
    sec = lambda b: A.beat_to_sec(arr, b)  # noqa: E731
    num, den = arr["time_signature"]
    total = sec(lead + A.total_beats(arr))
    L = ['<REAPER_PROJECT 0.1 "7.0" 0', "RIPPLE 0", f"TEMPO {arr['tempo_bpm']:g} {num} {den}",
         "SAMPLERATE 44100 0 0", f"PROJLEN {total + 4:.3f}"]
    for i, s in enumerate(arr["sections"], 1):          # regions: start line named, end line empty
        a = sec(lead + A.bar_to_beat(arr, s["start_bar"]))
        b = sec(lead + A.bar_to_beat(arr, s["start_bar"] + s["bars"]))
        L += [f"MARKER {i} {a:.6f} {_q(s['name'])} 1", f"MARKER {i} {b:.6f} \"\" 1"]

    # lyrics: one empty item per section, lyrics in the item notes
    lyr = []
    for s in arr["sections"]:
        if not s.get("lyrics"):
            continue
        a = sec(lead + A.bar_to_beat(arr, s["start_bar"]))
        notes = ["<NOTES"] + [f"|{ln}" for ln in s["lyrics"].splitlines()] + [">"]
        lyr.append(_item(a, sec(A.bar_to_beat(arr, s["bars"])), s["name"], notes))
    if lyr:
        L += _track("Lyrics", lyr)

    # reference audio and stems, positioned so beat 0 lines up with the lead-in
    audio_pos = sec(lead) - (arr.get("audio_offset_seconds") or 0.0)
    for name, rel, dur, mute, *pos in audio_files:      # pos given: already on the project timeline
        L += _track(name, [_item(pos[0] if pos else audio_pos, dur, name, _audio_source(rel))], mute=mute)

    parts = [t for t in arr["tracks"] if t.get("clips")]
    for i, tr in enumerate(parts):
        ch = channel_for(tr, i)
        items = [_item(sec(lead + c["start"]), sec(c["length"]), c["name"],
                       _midi_source(arr, c, ch, tr["instrument"]["program"], tr["instrument"].get("drums")))
                 for c in tr["clips"]]
        L += _track(tr["name"], items)
    L.append(">")
    path = os.path.join(out_dir, f"{safe(arr['title'])}.rpp")
    # indent like REAPER does, so diffs stay readable
    depth, out = 0, []
    for ln in L:
        if ln == ">":
            depth -= 1
        out.append("  " * depth + ln)
        if ln.startswith("<"):
            depth += 1
    open(path, "w").write("\n".join(out) + "\n")
    return path


# ---------------------------------------------------------------- package

def _duration(path):
    import soundfile as sf
    try:
        return sf.info(path).duration
    except Exception:
        import subprocess
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                           capture_output=True, text=True)
        return float(r.stdout.strip())


def export(arr_path, out_dir, preview=False):
    arr = A.load(arr_path)
    base = os.path.dirname(os.path.abspath(arr_path))
    os.makedirs(out_dir, exist_ok=True)
    lead = lead_in_beats(arr)

    audio_files = []        # (track name, path relative to out_dir, seconds, muted)
    refs = []
    if arr.get("reference_audio"):
        refs.append(("Reference mix", arr["reference_audio"], False))
    for t in arr["tracks"]:
        if t.get("audio"):
            refs.append((f"{t['name']} (stem)", t["audio"], True))
    if refs:
        os.makedirs(os.path.join(out_dir, "audio"), exist_ok=True)
    for name, rel, mute in refs:
        src = rel if os.path.isabs(rel) else os.path.join(base, rel)
        dst = os.path.join("audio", os.path.basename(src))
        if not os.path.exists(os.path.join(out_dir, dst)):
            shutil.copy(src, os.path.join(out_dir, dst))
        audio_files.append((name, dst, _duration(src), mute))

    write_midi(arr, out_dir, lead)
    rpp = write_rpp(arr, out_dir, lead, audio_files)
    shutil.copy(arr_path, os.path.join(out_dir, "arrangement.json"))

    with open(os.path.join(out_dir, "sections.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "bar", "bars", "start_seconds", "chords"])
        for s in arr["sections"]:
            b = lead + A.bar_to_beat(arr, s["start_bar"])
            w.writerow([s["name"], int(b / A.beats_per_bar(arr)) + 1, s["bars"], f"{A.beat_to_sec(arr, b):.2f}",
                        " ".join(c for _, c in s.get("chords", []))])
    with open(os.path.join(out_dir, "lyrics.txt"), "w") as f:
        for s in arr["sections"]:
            f.write(f"[{s['name']}]\n{s.get('lyrics') or ''}\n\n")

    if preview:
        import subprocess
        import synth
        sf2 = synth.default_soundfont()
        if sf2:
            y = synth.render_sf2(arr, sf2, lead_beats=lead)
            os.makedirs(os.path.join(out_dir, "preview"), exist_ok=True)
            wav = os.path.join(out_dir, "preview", "midi-preview.wav")
            synth.write_wav(wav, y)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", wav, "-b:a", "192k", wav[:-4] + ".mp3"], check=True)
            os.remove(wav)
            # also as a muted audio track in the REAPER project, so the MIDI can be heard before instruments exist
            audio_files.append(("MIDI preview (GM SoundFont)", "preview/midi-preview.mp3", len(y) / synth.SR, True, 0.0))
            rpp = write_rpp(arr, out_dir, lead, audio_files)

    write_readme(arr, out_dir, lead, audio_files)
    return rpp


def write_readme(arr, out_dir, lead, audio_files):
    num, den = arr["time_signature"]
    lines = [
        f"{arr['title']} - DAW project package (JewGrass)",
        "",
        f"Tempo {arr['tempo_bpm']:g} bpm (quarter notes), {num}/{den}, key {arr.get('key') or 'unknown'}.",
        f"Made by the {arr['provenance'].get('route', '?')} route from {arr['provenance'].get('source', '?')}.",
        "",
        "REAPER: open the .rpp. Sections are regions, lyrics are on the Lyrics track (View > item notes),",
        "every MIDI item is editable. MIDI tracks have no instrument yet: add one per track",
        "(Insert FX: ReaSynth, ReaSamplOmatic5000 with a sample, or any free SFZ/SoundFont player).",
        "",
        "Any other DAW (Ardour, LMMS, Ableton, Bitwig, Logic, Cubase): import the '(all tracks).mid' file",
        f"for one track per part, then set the project tempo to {arr['tempo_bpm']:g} and {num}/{den} by hand if",
        "your DAW ignores the tempo in MIDI files. Section names are MIDI markers; sections.csv has the bars.",
        "",
    ]
    if lead:
        lines += [f"The project starts with {int(lead / A.beats_per_bar(arr))} lead-in bar(s) so the audio before the",
                  "first downbeat fits; the song's bar 1 is after them.", ""]
    lines += ["Tracks:"]
    for t in arr["tracks"]:
        lines.append(f"  {t['name']:<14} {t['origin']:<11} {t['instrument']['name']}. {t.get('notes', '')}")
    for name, rel, _, mute, *_ in audio_files:
        lines.append(f"  {name:<14} audio       {rel}{' (muted: unmute to compare)' if mute else ''}")
    open(os.path.join(out_dir, "README.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    print(export(args[0], args[1], preview="--preview" in sys.argv))
