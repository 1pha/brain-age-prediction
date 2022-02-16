#!/bin/bash

for seed in {92..141};
    do
    python train.py --registration=non_registered --run_name="ResNet (Base) Non Registered seed$seed Augment" --checkpoint_period=1 --gpu_num=1 --seed=$seed --augment_replacement=True
done