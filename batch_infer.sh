read -p "CUDA_VISIBLE_DEVICES=" device
device=${device:-1}

read -p "--path=" path
path=${path}

export CUDA_VISIBLE_DEVICES=${device}
export HYDRA_FULL_ERROR=1

for xai_method in gbp ig gcam_avg
do
    echo ${xai_method}
    python infer_ckpt.py --path=${path}\
                         --xai_method=${xai_method}\
                         --infer_xai=True
done