import os
import wandb

from sage.config import load_config
from sage.args import parse_args
from sage.training import runner, unlearner
from utils.misc import seed_everything, get_today


if __name__=="__main__":
    
    cfg = load_config()
    args = parse_args()
    cfg.update(args)
    run_name = cfg.run_name if cfg.get('run_name') else 'DEFAULT NAME'

    cfg.registration = 'mni'
    cfg.unused_src = []

    cfg.unlearn = True
    cfg.unlearn_cfg.encoder.name = 'resnet'

    cfg.unlearn_cfg.opt_conf.point = 50
    cfg.epochs = 200
    cfg.early_patience = 40
    # cfg.unlearn_cfg.opt_conf.use = False
    cfg.unlearn_cfg.domainer.num_dbs = 4 - len(cfg.unused_src)

    name = 'RES(128) 4DB CONF50 '
    cfg.RESULT_PATH = os.path.join(cfg.RESULT_PATH, name + get_today())
    seed_everything(cfg.seed)

    wandb.login()
    wandb.init(
        project='3d_smri',
        config=vars(cfg),
        name=name
    )

    # model = runner.run(cfg)
    unlearner.run(cfg)
    wandb.finish()