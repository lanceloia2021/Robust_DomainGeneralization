#!/bin/bash
# domain=("clipart" "painting" "real" "sketch")
domain=("P" "A" "C" "S")
max=$((${#domain[@]}-1))

times=1
cuda="2,3"
seed=1046

echo "CUDA: $cuda"
echo "Random seed: $seed"

for j in `seq 0 $max`
do
  for i in `seq 0 $max`
  do
    if (($i == $j )); then
      continue
    fi
    echo "From ${domain[$i]} to ${domain[$j]}"
    CUDA_VISIBLE_DEVICES=$cuda python train.py --targetid=$j --labeledid=$i --model=DGresnet --name=${domain[i]}2${domain[j]} --seed=$seed
    sleep 5s
  done
done
