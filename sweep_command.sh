export HYDRA_FULL_ERROR=1

read -p "Enter devices: " device
export CUDA_VISIBLE_DEVICES=$device

read -p "Enter devices: ppmi|adni " ds
dataset=$ds

sweep_ppmi() {
    echo "Sweep on PPMI"
    python sweep.py --sweep_cfg_name=ppmi_sweep.yaml\
                    --wandb_project=ppmi\
                    --config_name=train_binary.yaml\
                    --sweep_prefix='Scratch'\
                    --overrides="['dataset=ppmi_binary', \
                                  '+dataset.modality=[t2]', \
                                  'dataloader.batch_size=4', \
                                  'dataloader.num_workers=2', \
                                  'trainer.accumulate_grad_batches=8']"
}

sweep_adni() {
    echo "Sweep on ADNI"
    python sweep.py --sweep_cfg_name=adni_sweep.yaml\
                    --wandb_project=adni\
                    --config_name=train_cls.yaml\
                    --sweep_prefix='Scratch'\
                    --overrides="['dataloader.batch_size=8']"
}

# Check the input argument and call the appropriate function
if [ $dataset = "ppmi" ]; then
    sweep_ppmi
elif [ $dataset = "adni" ]; then
    sweep_adni
else
    echo "Invalid argument. Usage: $0 [ppmi|adni]. Got $dataset instead"
    exit 1
fi