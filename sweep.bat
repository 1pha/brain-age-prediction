FOR /L %%x IN (133, 1, 142) DO (
    python train.py --seed=%%x --run_name="ResNet (Base) 256 seed%%x AUG_REPLACEMENT" --checkpoint_period=1
)