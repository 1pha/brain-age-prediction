import hydra
import omegaconf

import sage


logger = sage.utils.get_logger(name=__name__)


@hydra.main(config_path="config", config_name="train_cls.yaml", version_base="1.1")
def main(config: omegaconf.DictConfig):
    logger.info("Start Training")
    sage.trainer.train(config)

if __name__=="__main__":
    main()
