#!/bin/bash

for seed in {42..91};
    do
    python train.py --registration=non_registered --run_name="ResNet (Base) Non Registered seed$seed Naive" --checkpoint_period=1 --gpu_num=0 --seed=$seed
done