import torch.nn as nn

from sage.config import load_config
from sage.training.trainer import MRITrainer

cfg = load_config()
cfg.force_cpu = True
cfg.encoder.name = "resnet"
trainer = MRITrainer(cfg)
trainer.models["encoder"].avgpool = nn.MaxPool3d(3)

from sage.visualization.utils import Assembled

model = Assembled(trainer.models["encoder"], trainer.models["regressor"])

sample_tensor = next(iter(trainer.train_dataloader))[0]

# from sage.visualization.lrp.innvestigator import InnvestigateModel

# inn_model = InnvestigateModel(model, lrp_exponent=2, method="e-rule", beta=.5)
# model_prediction, heatmap = inn_model.innvestigate(in_tensor=sample_tensor)

from sage.visualization.PyTorchRelevancePropagation.src.lrp import LRPModel

lrp_model = LRPModel(model=model)
r = lrp_model.forward(sample_tensor)
