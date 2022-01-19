python train.py --lr=1e-5 --augment_replacement=False --run_name="Naive lr1e-5"
python train.py --lr=1e-6 --augment_replacement=False --run_name="Naive lr1e-6"
python train.py --lr=1e-7 --augment_replacement=False --run_name="Naive lr1e-7"

python train.py --lr=1e-5 --augment_replacement=True --run_name="Augment lr1e-5"
python train.py --lr=1e-6 --augment_replacement=True --run_name="Augment lr1e-6"
python train.py --lr=1e-7 --augment_replacement=True --run_name="Augment lr1e-7"