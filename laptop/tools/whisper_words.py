"""whisper_words.py AUDIO LYRICS_TXT OUT_JSON [MODEL]: word timestamps with faster-whisper on the CPU.

Settings: model medium.en (or the 4th argument), int8, 6 CPU threads, English, beam 5, word timestamps,
no VAD, no conditioning on previous text, initial prompt = first ~200 characters of the lyrics.
Writes a list of segments: {start, end, text, words: [{start, end, word, probability}]}.
Runs in ~/band/tools/asr/.venv through flock + heavy.sh.
"""
import json
import subprocess
import sys
import time

import numpy as np

from faster_whisper import WhisperModel

audio, lyrics_path, out = sys.argv[1:4]
model_name = sys.argv[4] if len(sys.argv) > 4 else "medium.en"
lyr = open(lyrics_path).read()
prompt = " ".join(l for l in lyr.splitlines() if l.strip() and not l.startswith("["))[:200]
t0 = time.time()
model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=6)
t1 = time.time()
# Decode with ffmpeg to 16 kHz mono float32 (faster-whisper 1.2.1's own decoder breaks with PyAV 19).
pcm = np.frombuffer(subprocess.run(["ffmpeg", "-loglevel", "error", "-i", audio, "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                                  check=True, capture_output=True).stdout, dtype=np.float32)
segs, info = model.transcribe(pcm, language="en", beam_size=5, word_timestamps=True, vad_filter=False,
                              condition_on_previous_text=False, initial_prompt=prompt)
res = []
for s in segs:
    res.append({"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip(),
                "words": [{"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word,
                           "probability": round(w.probability, 3)} for w in (s.words or [])]})
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({"model": model_name, "load_s": round(t1 - t0, 1), "transcribe_s": round(time.time() - t1, 1),
                  "segments": len(res), "words": sum(len(s["words"]) for s in res),
                  "audio_s": round(info.duration, 1)}))
