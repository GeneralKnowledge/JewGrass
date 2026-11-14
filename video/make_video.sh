#!/usr/bin/env bash
# Full pipeline for one song, once its real audio is in SONG_DIR/media/:
#   media/master.flac   the approved master
#   media/vocals.flac   the vocal stem of that same take (for lyric timing)
#   media/beats.json    {tempo, beats[], downbeats[]} (beat_this)
#   media/lyrics.txt    the lyrics with section tags
# SONG_DIR/storyboard.yaml must exist (written by the creative director).
#
#   video/make_video.sh video/songs/deed-to-the-glen [JOBS]
#
# On the laptop: run through ~/band/heavy.sh, only when no music job is running.
set -euo pipefail
SONG=${1:?song dir}
JOBS=${2:-$(( $(nproc) - 1 ))}
cd "$(dirname "$0")"
SONG=$(realpath --relative-to=. "$SONG" 2>/dev/null || echo "$SONG")
M=$SONG/media

echo "== 1. lyric timing from the vocal stem"
python3 tools/align_lyrics.py "$M/vocals.flac" "$M/lyrics.txt" "$M/beats.json" "$SONG/timing.yaml" --beats-out "$SONG/grid.json"

echo "== 2. analysis of the master"
ln -sf "$(realpath "$M/master.flac")" "$SONG/master.flac"
rm -f "$SONG/master.wav"   # the click-track guide, if a preview was made
python3 -m engine.analysis "$SONG/master.flac" "$SONG/analysis.json" --beats "$SONG/grid.json" ${STEMS:+--stems "$STEMS"}

echo "== 3. contact sheet (check it before the long render)"
python3 -m engine.render "$SONG" --stills

echo "== 4. full render"
python3 -m engine.render "$SONG" --jobs "$JOBS"

echo "== 5. thumbnail + vertical cut"
python3 -m engine.extras "$SONG" --jobs "$JOBS"
ls -la "$SONG/out"
