# MRI Data 3D Brain Convolution
### 10.18(Sun)
+ Changes
  + **MinMaxScaler**
    + instead of dividing 255 to every subjects
    + scaler fitted to each subjects respectively
    + `sklearn.preprocessing.MinMaxScaler`
  + **Interpolate**
    + 256cube too big
    + narrowed down to 64cube
+ Results
  + currently running with
    + `Model` Levakov
    + `Epochs` 400
    + `Batch_size` 8
    + `Scheduler` Cosine Annealing, with initial lr=1e-4
  + not good... (18:50)
  + Reference
    + [Levakov et al.](https://onlinelibrary.wiley.com/doi/pdf/10.1002/hbm.25011)
### 10.12
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