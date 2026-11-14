#!/usr/bin/env bash
# outro-job.sh SRC CAPTION LYRICS BPM START END OUTDIR SEEDS(comma) TAG
# Repaints [START,END] of SRC once per seed (one process each, under the GPU lock), then separates the window
# of the source and of every result (sep venv) so vocal presence can be compared. Writes OUTDIR/<TAG>.done.
src=$1 cap=$2 lyr=$3 bpm=$4 st=$5 en=$6 out=$7 seeds=$8 tag=$9
A=~/band/tools/sep/.venv/bin/audio-separator; mkdir -p $out/vocalcheck
cd ~/band/ACE-Step-1.5
for seed in $(echo $seeds | tr , " "); do
  echo "=== OUTRO $tag $seed $(date +%T)" >> ~/band/monitor.log
  PYTORCH_ALLOC_CONF=expandable_segments:True flock ~/band/.gpu.lock ~/band/heavy.sh uv run --no-sync python ../repaint.py $src $cap $lyr $bpm $st $en $out $seed
  echo "EXIT_REPAINT_$seed=$?"
  [ -f ~/band/stop.hot ] && exit 1
done
dur=$(python3 -c "print($en-$st)")
for f in $src $(for s in $(echo $seeds | tr , " "); do echo $out/repaint_$(python3 -c "print(f'{$st:g}-{$en:g}')")_$s.wav; done); do
  [ -f $f ] || continue
  b=$(basename $f .wav); x=$out/vocalcheck/$b-window.wav
  ffmpeg -loglevel error -y -ss $st -t $dur -i $f $x
  flock ~/band/.gpu.lock ~/band/heavy.sh $A $x -m model_bs_roformer_ep_317_sdr_12.9755.ckpt --model_file_dir ~/band/models/separator --output_dir $out/vocalcheck --output_format WAV > /dev/null 2>&1
  echo "EXIT_SEP_$b=$?"
done
date +%s > $out/$tag.done
