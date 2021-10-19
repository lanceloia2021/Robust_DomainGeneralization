#!/bin/bash
# domain=("clipart" "painting" "real" "sketch")
domain=("P" "A" "C" "S")

times=1
max=$((${#domain[@]}-1))
seed=146

echo "Random Seed: ${seed}"
for j in `seq 0 $max`
do
  for i in `seq 0 $max`
  do
    if (($i == $j)); then
      continue
    fi
    echo "From ${domain[$i]} to ${domain[$j]}"
    CUDA_VISIBLE_DEVICES=0,1,3,4 python train.py --targetid=$j --labeledid=$i --model=dgresnet18 --name=${domain[i]}2${domain[j]} --seed=$seed
    sleep 5s
  done
done
