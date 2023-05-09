# MRI Data 3D Brain Convolution

### Setup

Make a softlink for biobank

```bash
conda create -n age python=3.10 -y
conda activate age
pip install -r requirements.txt
```

### Commands
Confusing configs.

* Note that `module.load_model_ckpt` and `module.load_from_checkpoint` is different.
  - The former only calls models weights
  - While the latter recalls every configurations including optimizers and others.
#### Train
```
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python train.py logger.name="ResNet10t-masked"
```

#### Inference
```
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python inference.py module.load_model_ckpt=AA.CKPT
```

#### Resume Training
```
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python train.py **+logger.version=HASH +logger.resume=must** module.load_from_checkpoint=AA.CKPT