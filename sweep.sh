#!/bin/bash

# for seed in {42..91};
#     do
#     python train.py --registration=non_registered --run_name="ResNet (Base) Non Registered seed$seed Augment" --checkpoint_period=1 --gpu_num=0 --seed=$seed --augment_replacement=True
# done

python train.py --registration=non_registered --run_name="ResNet (Base) Non Registered seed77 Augment" --checkpoint_period=1 --seed=77 --augment_replacement=True
python train.py --registration=non_registered --run_name="ResNet (Base) Non Registered seed92" --checkpoint_period=1 --seed=92 --augment_replacement=False
python train.py --run_name="ResNet (Base) 256 seed88 W/O AUG" --checkpoint_period=1 --seed=88 --augment_replacement=False