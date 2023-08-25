# Deep Learning for Human Activity Recognition
> Keras implementation of CNN, DeepConvLSTM, and bi-LSTM and stacked-lstm for sensor-based Human Activity Recognition (HAR).  

This repository contains keras (tensorflow.keras) implementation of Convolutional Neural Network (CNN), Deep Convolutional LSTM (DeepConvLSTM), Stacked LSTM and bidirectional LSTM for human activity recognition (HAR) using smartphones sensor dataset, *UCI smartphone*.

**Table 1.** The summary of the results amongst five methods on UCI smartphone dataset.

| Method | Accuracy | Precision | Recall | F1-score |
| --- | :---: | :---: | :---: | :---: |
| LightGBM | **96.33** | **96.58** | **96.37** |  **96.43** |
| CNN [1] | 95.29 | 95.46 | 95.50 |  95.47 |
| DeepConvLSTM [1] | 95.66 | 95.71 | 95.84 | 95.72 |
| SDAE [2] | 78.28 | 78.83 | 78.47 | 78.25 |
| MLP | 93.81 | 93.97 | 94.04 |  93.85 |

# Setup

# Dataset
The dataset includes types of movement amongst thre static posture (*STANDING*, *SITTING*, *LYING*) and three dynamic activities (*WALKING*,*WALKING UPSTAIRS*, *WALKING DOWNSTAIRS*).

![](https://img.youtube.com/vi/XOEN9W05_4A/0.jpg)  
[Watch video](https://www.youtube.com/watch?v=XOEN9W05_4A)  
[Download dataset](http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions)

# Methods
## Feature engineering 
To represent raw sensor signals as a feature vector, 621 features are created by preprocessing, dast Fourier transform (FFT), statistics, etc. 
 ([generate_features.py](https://github.com/Tahoora78/Human_Activity_Recognition/blob/main/generate_features.py).)  
The 621 features of the training dataset were traind on a LightGBM classifier by 5-fold cross-validation, and evaluated on the test dataset.


## Convolutional Neural network (CNN)
The preprocessed raw sensor signals are cassified CNN. The CNN architecture, which is a baseline to DeepConvLstm.

## Deep Convolutional LSTM (DeepConvLSTM) 
The preprocessed raw sensor signals are classified DeepConvLSTM. The fully connected layers of the CNN are replaced with LSTM layers.

## Stacked LSTM
The preprocessed raw sensor signals are trained with SDAE, softmax layer is superimposed on top of Encoder, then whole network is fine-tuned for the target classification task.

### Stacked bi-LSTM
The preprocessed raw sensor signals are trained with SDAE, softmax layer is superimposed on top of Encoder, then whole network is fine-tuned for the target classification task.



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