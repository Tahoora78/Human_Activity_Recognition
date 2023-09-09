# Deep Learning for Human Activity Recognition
> Keras implementation of CNN, DeepConvLSTM, bi-LSTM, and stacked-lstm for sensor-based Human Activity Recognition (HAR).  

This repository contains keras (tensorflow.keras) implementation of Convolutional Neural Network (CNN), Deep Convolutional LSTM (DeepConvLSTM), Stacked LSTM, and bidirectional LSTM for human activity recognition (HAR) using smartphone sensor dataset, *UCI smartphone*.

**Table 1.** The summary of the results amongst five methods on the UCI smartphone dataset.

| Method | Accuracy | Precision | Recall | F1-score |
| --- | :---: | :---: | :---: | :---: |
| stacked-lstm | 94.45 | 94.50 | 94.62 | 94.44 |
| CNN | 93.13 | 93.32 | 93.45 |  93.29 |
| DeepConvLSTM | 94.38 | 94.44 | 94.54 | 94.36 |
| Stacked bi-lstm | **94.95** | **95.10** | **95.13** |  **95.02** |

# Setup

# Dataset
The dataset includes types of movement amongst three static postures (*STANDING*, *SITTING*, *LYING*) and three dynamic activities (*WALKING*,*WALKING UPSTAIRS*, *WALKING DOWNSTAIRS*).

![](https://img.youtube.com/vi/XOEN9W05_4A/0.jpg)  
[Watch video](https://www.youtube.com/watch?v=XOEN9W05_4A)  
[Download dataset](http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions)

# Methods
## Feature engineering 
To represent raw sensor signals as a feature vector, 621 features are created by preprocessing, dast Fourier transform (FFT), statistics, etc. 
 ([generate_features.py](https://github.com/Tahoora78/Human_Activity_Recognition/blob/main/generate_features.py).)  
The 621 features of the training dataset were trained on a LightGBM classifier by 5-fold cross-validation, and evaluated on the test dataset.


## Convolutional Neural network (CNN)
The preprocessed raw sensor signals are classified as CNN. The CNN architecture is a baseline for DeepConvLstm.

## Deep Convolutional LSTM (DeepConvLSTM) 
The preprocessed raw sensor signals are classified as DeepConvLSTM. The fully connected layers of the CNN are replaced with LSTM layers.

## Stacked LSTM
The preprocessed raw sensor signals are trained with SDAE, the softmax layer is superimposed on top of the Encoder, then the whole network is fine-tuned for the target classification task.

### Stacked bi-LSTM
The preprocessed raw sensor signals are trained with SDAE, the softmax layer is superimposed on top of the Encoder, then the whole network is fine-tuned for the target classification task.



# Prerequisits
<br>
python 3+
<br>
Tensorflow 2.0
<br>
Numpy
<br>
Pandas
<br>
Keras
