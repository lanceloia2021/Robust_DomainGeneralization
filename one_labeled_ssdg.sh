#!/bin/bash
# domain=("clipart" "painting" "real" "sketch")
domain=("P" "A" "C" "S")
max=$((${#domain[@]}-1))

times=1
cuda="3"
seed=146

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
    CUDA_VISIBLE_DEVICES=$cuda python train.py --target=$j --labeled=$i --model=DGresnet --name=${domain[i]}2${domain[j]} --seed=$seed
    sleep 5s
  done
done
