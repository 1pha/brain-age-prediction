# MRI Data 3D Brain Convolution

### TODO
+ Change config.debug+print to logger.debug
+ Change every print functions to logger
+ Multiple GPU handling code
+ Smarter ways for multiple metrics.
+ Parcellate/Encapsulate/Separate functions in dataloader & trainer.
+ Move out save directory to `train.py` from `trainer.py` and add `logging.FileHandler`

### Main Task
+ Age Prediction with 1.4k structural MRI data from 4 sources
+ Unlearning through confusion loss (nearly failure)
+ Saliency map to find brain age biomarker
### Trainer Information
+ Models
  + Vanilla CNN
  + [Resnet]()
  + [EfficientNet]()
  + [ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases](https://arxiv.org/abs/2103.10697)
### Related Works
+ Deep Learning approach to Age Prediction
  + [Levakov et al.](https://onlinelibrary.wiley.com/doi/pdf/10.1002/hbm.25011)
  + [Brain MRI-based 3D Convolutional Neural Networks for Classification of Schizophrenia and Controls
](https://arxiv.org/abs/2003.08818)

+ Brain Age Biomarkers
  + (Further updated)
+ Represnetation Disentanglement
  + (Further updated)
