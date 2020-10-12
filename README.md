# MRI Data 3D Brain Convolution
## 10.12
+ PR Curve is **not good**:rotating_light: 
  + currently working predicting only one side
  + even though it's nearly balanced...
+ **Task** Old/Young Binary Classification
+ [Current Model](./src/model/vanilla.py)
  + `optimizer`: Adam, lr=1e-3
  + `loss`: BCELoss
  + `scheduler`: CosineAnnealing
+ :construction:Trying to implement **Inception** Module
  + Reference
    + [Brain MRI-based 3D Convolutional Neural Networks for Classification of Schizophrenia and Controls
](https://arxiv.org/abs/2003.08818)
    + [Implementing Inception(KR)](https://wingnim.tistory.com/36)