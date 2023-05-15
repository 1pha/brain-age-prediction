read -p "CUDA_VISIBLE_DEVICES=" device
device=${device:-1}

CUDA_VISIBLE_DEVICES=${device}
HYDRA_FULL_ERROR=1

python infer_ckpt.py --path="resnet10t-mask" --mask="mask"
python infer_ckpt.py --path="resnet10t-mask" --mask="nomask"

python infer_ckpt.py --path="swinvit(pre)-aug" --mask="mask"
python infer_ckpt.py --path="swinvit(pre)-aug" --mask="nomask"

python infer_ckpt.py --path="resnet10t-naive" --mask="mask"
python infer_ckpt.py --path="resnet10t-naive" --mask="nomask"
