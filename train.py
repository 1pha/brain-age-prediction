import wandb

from sage.config import load_config
from utils.misc import seed_everything


if __name__=="__main__":

    
    cfg = load_config()
    seed_everything(cfg.seed)

    wandb.login()
    wandb.init(project='3d_smri', config=vars(cfg))