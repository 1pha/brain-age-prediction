import os
import argparse
from pathlib import Path

import hydra
import omegaconf

import sage


logger = sage.utils.get_logger(name=__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--path", type=str, help="Leaf node directory name. e.g. resnet10t-mask")
    parser.add_argument("--root", default="assets/weights/", type=str, help="Root directory where weights resides")
    parser.add_argument("--weight", type=str, default="best", help="Choosing which weights to infer")
    parser.add_argument("--mask", type=bool, default=False, help="Masking inference")
    parser.add_argument("--batch_size", type=int, default="4", help="batch size during inference")
    
    args = parser.parse_args()
    return args


def main(args):
    root = Path(args.root) / args.path
    weight = sorted(root.glob("*.ckpt"))[int(args.weight == "last")]
    with hydra.initialize(config_path=str(root / ".hydra"), version_base="1.1"):
        config = hydra.compose(config_name="config.yaml",
                               overrides=["misc.modes=[train,valid,test]",
                                          f"module.load_model_ckpt={weight}",
                                          f"dataloader.batch_size={args.batch_size}",
                                          f"module.mask={'assets/mask.npy' if args.mask == 'mask' else False}"])
    logger.info("Start Training")
    root = root / ("mask" if args.mask == "mask" else "no-mask")
    os.makedirs(root, exist_ok=True)
    sage.trainer.inference(config, root_dir=root)


if __name__=="__main__":
    args = parse_args()
    main(args)
