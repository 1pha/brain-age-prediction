import wandb

from sage.config import load_config
from sage.args import parse_args
from sage.training import runner
from utils.misc import seed_everything


if __name__=="__main__":
    
    cfg = load_config()
    args = parse_args()
    cfg.update(args)
    run_name = cfg.run_name if cfg.get('run_name') else 'DEFAULT NAME'
    seed_everything(cfg.seed)

    wandb.login()
    wandb.init(
        project='3d_smri',
        config=vars(cfg),
        name='MNI / IXI+Dallas / Resnet No Maxpool'
    )

    model = runner.run(cfg)
    wandb.finish()