FOR /L %%x IN (43, 1, 52) DO (
    python train.py --seed=%%x --run_name="ResNet (Base) 256 seed%%x" --checkpoint_period=1
)