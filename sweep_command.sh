export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=1

python sweep.py --sweep_cfg_name=ppmi_sweep.yaml\
                --wandb_project=ppmi\
                --config_name=train_binary.yaml\
                --overrides="['module.load_model_ckpt=meta_brain/weights/default/resnet10-42/156864-valid_mae3.465.ckpt',\
                              '+module.load_model_strict=False']"