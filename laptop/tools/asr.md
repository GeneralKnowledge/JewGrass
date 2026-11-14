# Word timestamps with faster-whisper (CPU)

`~/band/tools/asr` is a separate venv, kept apart from ACE-Step's. It runs on the CPU only, so it needs no CUDA torch.

```bash
mkdir -p ~/band/tools/asr && cd ~/band/tools/asr
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python faster-whisper      # faster-whisper 1.2.1, ctranslate2 4.9.0, av 19.0.2
# model download (flaky network: retry; Xet off)
HF_HUB_DISABLE_XET=1 .venv/bin/python -c "from faster_whisper.utils import download_model; print(download_model('medium.en'))"
```

Run it through the lock, like every heavy job:

```bash
flock ~/band/.gpu.lock ~/band/heavy.sh ~/band/tools/asr/.venv/bin/python ~/band/tools/whisper_words.py \
  <vocal_stem.wav> <lyrics.txt> <out/whisper-words.json> [medium.en]
```

`whisper_words.py` settings:
- model medium.en, `device="cpu"`, `compute_type="int8"`, `cpu_threads=6`
- `language="en"`, `beam_size=5`, `word_timestamps=True`, `vad_filter=False`, `condition_on_previous_text=False`
- `initial_prompt` = the first ~200 characters of the lyrics, without the [section] lines

The script decodes audio with ffmpeg (16 kHz mono float32) and passes the samples to `transcribe`. faster-whisper 1.2.1's own decoder fails with PyAV 19 (`open() got an unexpected keyword argument 'metadata_errors'`).

Output is a list of segments: `{start, end, text, words: [{start, end, word, probability}]}`.

Deed to the Glen (240 s vocal stem): 166 s on the laptop CPU (i7-8750H, 6 threads), 3.4 GB peak RAM, 510 words, median word probability 0.95.
