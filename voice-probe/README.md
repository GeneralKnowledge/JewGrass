# Voice probe: Reuben's RVC voice, checkpoint by checkpoint

The same 18 s of Swim at Your Own Risk (take 5555, 88-106 s) every time, so each checkpoint is directly comparable.

| file | what |
|---|---|
| 00-original-ai-vocal.mp3 | the original generated vocal (BS-RoFormer stem), alone |
| 00-original-mix.mp3 | that vocal over the instrumental stem (same path as the converted mixes) |
| 00-your-recording.mp3 | 15 s of Reuben's own cleaned recording, for reference |
| eNN-your-voice.mp3 | the vocal converted with the epoch-NN checkpoint, alone |
| eNN-mix.mp3 | that converted vocal over the same instrumental |

Vocal-only files are loudness-matched to -20 LUFS, mixes to -16 LUFS (peaks capped at about -1 dBFS), 320k mp3.
Training: Applio RVC, 48 kHz, HiFi-GAN, batch 4, pretrained base, 3.1 min of audio (one 16 kHz phone recording).
