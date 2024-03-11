export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=1

python sweep.py --sweep_cfg_name=ppmi_sweep.yaml\
                --config_name=train_binary.yaml\
                --overrides="['dataset=ppmi_binary', 'model=convnext_binary']"