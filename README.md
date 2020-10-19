# MRI Data 3D Brain Convolution
### 10.19(Mon)
+ Changes
  + Display
    + Accuracy available on the terminal
    + Plot with results from run.py, not re-calculating(too much waste of time)
  + Tensorboard
    + Accuracy available on Tensorboard as well
  + Save
    + Save model with best validation accuracy from now on(since each run takes too much time)
    + Save me... tired...
+ Results
  + Currently running with extra `Dropout` on the end(with p=.3)
  + Since sever overfitting is has occurred, need to find a way through it...
  + F1
    + **train** .94
    + **test** .51
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
  + yes good(21:28)
    + found a **severe** error in my code...
    + since this is originally from regression with LOGs, my prediction was collecting informations transforming back with exp! which is not needed for classification task...(last layer of sigmoid will do everything...) So stupid...
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