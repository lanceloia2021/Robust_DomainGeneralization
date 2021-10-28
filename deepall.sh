#!/bin/bash
# domain=("clipart" "painting" "real" "sketch")
domain=("photo" "art_painting" "cartoon" "sketch")

times=1
cuda="2,3"
seed=1046

echo "CUDA: $cuda"
echo "Random seed: $seed"
CUDA_VISIBLE_DEVICES=$cuda python train.py --target=0 --labeled=1,2,3 --model=DGresnet --name=deepall_P_$seed --seed=$seed
sleep 5
CUDA_VISIBLE_DEVICES=$cuda python train.py --target=1 --labeled=0,2,3 --model=DGresnet --name=deepall_A_$seed --seed=$seed
sleep 5
CUDA_VISIBLE_DEVICES=$cuda python train.py --target=2 --labeled=0,1,3 --model=DGresnet --name=deepall_C_$seed --seed=$seed
sleep 5
CUDA_VISIBLE_DEVICES=$cuda python train.py --target=3 --labeled=0,1,2 --model=DGresnet --name=deepall_S_$seed --seed=$seed
sleep 5

# CUDA_VISIBLE_DEVICES=0,1,3,4 python train.py --target=0 --labeled=1,2,3 --model=DGresnet --name=deepall