read -p "CUDA_VISIBLE_DEVICES=" device
device=${device:-1}

read -p "--path=" path
path=${path}

read -p "--xai_method=" xai
xai=${xai:-gbp}

CUDA_VISIBLE_DEVICES=${device}
HYDRA_FULL_ERROR=1

# For a single checkpoint infer the followings
# Mask vs. No-mask
# Top-indiv vs. Aggregate

python infer_ckpt.py --path=${path}\
                     --xai_method=${xai}\
                     --mask=mask\
                     --batch_size=1\
                     --infer_xai=True\
                     --top_individual=True\
                     --top_k=0.99

python infer_ckpt.py --path=${path}\
                     --xai_method=${xai}\
                     --mask=mask\
                     --batch_size=1\
                     --infer_xai=True\
                     --top_individual=False\
                     --top_k=0.99

python infer_ckpt.py --path=${path}\
                     --xai_method=${xai}\
                     --mask=nomask\
                     --batch_size=1\
                     --infer_xai=True\
                     --top_individual=True\
                     --top_k=0.99

python infer_ckpt.py --path=${path}\
                     --xai_method=${xai}\
                     --mask=nomask\
                     --batch_size=1\
                     --infer_xai=True\
                     --top_individual=False\
                     --top_k=0.99