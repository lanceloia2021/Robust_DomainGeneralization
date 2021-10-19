#!/bin/bash
# domain=("clipart" "painting" "real" "sketch")
domain=("photo" "art_painting" "cartoon" "sketch")

times=1
seed=146

echo "Seed: $seed"
CUDA_VISIBLE_DEVICES=0,1,3,4 python train.py --targetid=0 --labeledid=1,2,3 --model=dgresnet18 --name=deepall_P_$seed --seed=$seed
sleep 5
CUDA_VISIBLE_DEVICES=0,1,3,4 python train.py --targetid=1 --labeledid=0,2,3 --model=dgresnet18 --name=deepall_A_$seed --seed=$seed
sleep 5
CUDA_VISIBLE_DEVICES=0,1,3,4 python train.py --targetid=2 --labeledid=0,1,3 --model=dgresnet18 --name=deepall_C_$seed --seed=$seed
sleep 5
CUDA_VISIBLE_DEVICES=0,1,3,4 python train.py --targetid=3 --labeledid=0,1,2 --model=dgresnet18 --name=deepall_S_$seed --seed=$seed
sleep 5