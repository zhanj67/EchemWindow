#!/bin/bash
# Runs all missing Roost jobs (skips finished ones and ones already running), P per pool, 3 GPUs.
cd "$(dirname "$0")"; mkdir -p logs; P=${P:-6}
i=0
for s in 0 1 2 3 4; do
  for spec in "left dedup" "right dedup" "cls dedup" "cls all"; do
    set -- $spec; t=$1; d=$2
    [ -f results/metrics_roost_${t}_${d}_s${s}.json ] && continue
    pat="python roost_train.py --task $t --seed $s --data $d"
    if [ $d = dedup ]; then re="python roost_train.py --task $t --seed $s( --data dedup)?\$"; else re="^$pat\$"; fi
    pgrep -f "$re" >/dev/null && continue
    echo "CUDA_VISIBLE_DEVICES=$((i % 3)) conda run --no-capture-output -n ecw $pat > logs/${t}_${d}_s${s}.log 2>&1"
    i=$((i+1))
  done
done | xargs -P $P -I{} bash -c "{} || true"
