export CUDA_VISIBLE_DEVICES=2
export HYDRA_FULL_ERROR=1

echo $CUDA_VISIBLE_DEVICES
for xai_method in deconv deeplift gbp gcam_avg ggcam_avg ggcam gradxinput ig
do
    echo $xai_method
    python inference.py --infer_xai=True\
                        --path=resnet10-42\
                        --batch_size=32\
                        --infer_xai=True\
                        --xai_method=$xai_method
done
