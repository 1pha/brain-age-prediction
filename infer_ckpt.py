import os
import argparse
from pathlib import Path

import hydra

import sage


logger = sage.utils.get_logger(name=__name__)


MASK_DIR = Path("assets/masks")


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--path", type=str, help="Leaf node directory name. e.g. resnet10t-mask")
    parser.add_argument("--root", default="assets/weights/", type=str, help="Root directory where weights resides")
    parser.add_argument("--weight", type=str, default="best", help="Choosing which weights to infer")
    
    parser.add_argument("--mask", type=str, default=False, help="Masking inference")
    
    parser.add_argument("--batch_size", type=int, default="4", help="batch size during inference")
    
    parser.add_argument("--infer_xai", type=str, default="False", help="Infer xai or not")
    parser.add_argument("--top_k", type=float, default=0.99, help="")
    parser.add_argument("--xai_method", type=str, default="gbp", help="Which explainability method to use")
    parser.add_argument("--baseline", type=str, default=False, help="Baseline brain for Integrated gradients")
    
    args = parser.parse_args()
    return args


def main(args):
    root = Path(args.root) / args.path
    weight = sorted(root.glob("*.ckpt"))[int(args.weight == "last")]
    
    mask = sage.utils.parse_bool(args.mask)
    overrides = ["misc.modes=[train,valid,test]",
                 f"module.load_model_ckpt={weight}",
                 f"dataloader.batch_size={args.batch_size}",
                 f"module.mask={MASK_DIR/mask if mask else 'False'}"]
    
    infer_xai: bool = sage.utils.parse_bool(args.infer_xai)
    if infer_xai:
        logger.info("Infer XAI map")
        overrides += [
            "+module.target_layer_index=-1",
            "module._target_=sage.xai.trainer.XPLModule",
            f"+module.top_k_percentile={args.top_k}",
            f"+module.xai_method={args.xai_method}",
            "+trainer.inference_mode=False"
        ]
        if args.xai_method == "ig":
            overrides += [f"+module.baseline={args.baseline}"]
    else:
        logger.info("Infer Metrics")
    
    with hydra.initialize(config_path=str(root / ".hydra"), version_base="1.1"):
        config = hydra.compose(config_name="config.yaml", overrides=overrides)

    logger.info("Start Inference")
    root = root / ("mask" if mask else "no-mask")
    os.makedirs(root, exist_ok=True)
    sage.trainer.inference(config, root_dir=root)


if __name__=="__main__":
    args = parse_args()
    main(args)