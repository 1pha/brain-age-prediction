import os
import sys
import argparse
from pathlib import Path

import hydra

import sage


logger = sage.utils.get_logger(name=__name__)


MASK_DIR = Path("assets/masks")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, help="Leaf node directory name. e.g. resnet10t-mask")
    parser.add_argument("--ckpt_step", default=None, type=int, help="Finds checkpoint step.")
    parser.add_argument("--root", default="meta_brain/weights/default/", type=str, help="Root directory where weights resides")

    parser.add_argument("--batch_size", type=int, default=1, help="batch size during inference")

    parser.add_argument("--infer_xai", type=str, default="False", help="Infer xai or not")
    parser.add_argument("--top_k", type=float, default=0.99, help="")
    parser.add_argument("--xai_method", type=str, default="gbp", help="Which explainability method to use")
    parser.add_argument("--baseline", type=bool, default=False, help="Baseline brain for Integrated gradients")

    args = parser.parse_args()
    return args


def main(args):
    root = Path(args.root) / args.path
    # Starting with numbers is the checkpoint recorded by best monitoring checkpoint via save_top_k=1
    ckpts = sorted(root.glob("*.ckpt"))
    if args.ckpt_step is None:
        weight = ckpts[0]
    else:
        ckpts_step = [ckpt.stem for ckpt in ckpts]
        ckpt_idx = [idx for idx, ckpt in enumerate(ckpts_step)
                    if (ckpt.startswith("step") and int(ckpt.split("-")[0][4:]) == args.ckpt_step)]
        if len(ckpt_idx):
            # Yes there is a finding ckpt_step
            weight = ckpts[ckpt_idx[0]]
        else:
            # No step of checkpoint looking for.
            logger.info("No step of checkpoint you are looking for %s", args.ckpt_step)
            logger.info("Weight list: %s", ckpts)
            # 안되는건 그냥 나중에 step이름 쑤셔넣는거로 대체
            sys.exit()

    overrides = ["misc.modes=[train,valid,test]",
                 f"module.load_model_ckpt={weight}",
                 f"dataloader.batch_size={args.batch_size}"]
    
    infer_xai: bool = sage.utils.parse_bool(args.infer_xai)
    if infer_xai:
        logger.info("Infer XAI map")
        overrides += [
            "+module.target_layer_index=-1",
            "module._target_=sage.xai.trainer.XPLModule",
            f"+module.top_k_percentile={args.top_k}",
            f"+module.xai_method={args.xai_method}",
            "+trainer.inference_mode=False",
            "trainer.accelerator=gpu"
        ]
        if args.xai_method == "ig":
            overrides += [f"+module.baseline={sage.utils.parse_bool(args.baseline)}"]
    else:
        # overrides += ["+trainer.inference_mode=True"]
        logger.info("Infer Metrics")
    
    with hydra.initialize(config_path=str(root / ".hydra"), version_base="1.1"):
        config = hydra.compose(config_name="config.yaml", overrides=overrides, return_hydra_config=True)
        # TODO: Hydra key-interpolation did not work
        for callback in config.callbacks:
            _cb = config.callbacks[callback]
            _cb.update({"dirpath": root} if "dirpath" in _cb else {})

    logger.info("Start Inference")
    os.makedirs(root, exist_ok=True)
    sage.trainer.inference(config, root_dir=root, ckpt_step=args.ckpt_step)


if __name__=="__main__":
    args = parse_args()
    main(args)