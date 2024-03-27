# README

## An Yu, Ferhat Demirkiran, Jeremy Varghese, Peter Buonaiuto

```
Computer Science Department, University at Albany – SUNY
```
## 1 Module I: Animal Identification

## 1.1 Data Preparation

This guide provides detailed instructions for preparing your video dataset for training and prediction
using the Zamba package, please refer to zambapreparedata.py and zambavideoprepare.py files in our
github website

1.1.1 Training Data

For training the model, your data must be organized into two key components:

- Video Directory: A directory containing all the video files you wish to use for training.
    Example directory structure:

```
/../dataset/zamba_train_video
```
- Labels CSV File: A CSV file listing each video file along with its corresponding species label.
    This file should have two columns:

```
1.filepath: Relative or absolute path to the video file.
2.specieslabel: The label of the species recorded in the video.
```
```
Example CSV (zambatrain.csv):
file path , species label
video1.mp4 , bird
video2.mp4 , mammal
video3.mp4 , reptile
```
1.1.2 Testing Data

To prepare for making predictions with your trained model, simply organize your videos within a specific
directory.
Example directory structure:

```
/../dataset/zamba_pred_video
```
- Example of the path to each video file for prediction:
    testvideo1.mp4
    testvideo2.mp4


### 1.2 Setting Up Zamba Working Environment

In this part, we gave the steps to configure our specific working environment for using Zamba.
```
! python−−version
```
# python version 3.
```
! conda install ffmpeg
! ffmpeg −version
```
# ffmpeg version 4.4.
# I n s t a l l Zamba from the latest release
```
! pip install https://github.com/drivendataorg/zamba/releases/latest/download/zamba.tar.gz
```
# Verify Zamba installation, it is 2.3.
```
!pip show zamba
!pip install opencv−python−headless==4.5.5.62
!pip install typer==0.9
```
### 1.3 Training the Model with Zamba

```
Before initiating the training process we modified the zamba configuration for the timedistributed
model.
```
# Early stopping criteria: mode = min , monitor = valloss , patience = 3

```
To train your model using Zamba, run the following command in your terminal, adjusting paths as
necessary to match your directory structure:
zamba train --data-dir /../dataset/zamba_train_video
--labels /../dataset/zamba_prep/zamba_trainV.csv
--save-dir /../zamba_prep/zamba_out
```
### 1.4 Making Predictions

```
After training, you can use the trained model to make predictions on new videos. Update your trainconfiguration.yaml
file with the appropriate paths for conducting the prediction, predictconfig is required as we will be run-
ning inference on unseen test data.
predict config :

checkpoint:/../zamba_prep/zamba_out /../timedistributed.ckpt
data dir:/../dataset/zamba_pre_dvideo/
savedir:/.. /zamba_prep/zamba_out/

Important parameters in the ‘trainconfiguration.yaml‘ include:
```
- checkpoint: Path to the checkpoint file of your trained model, which is required to load the model
    for making predictions.
- datadir: The directory that holds the videos for which you intend to predict.
- savedir: The directory where the outcomes of your predictions will be stored.
With your videos appropriately placed in the specified directory, proceed to predictions by executing
the following command, referring to your configured YAML file:
zamba predict --config /../zamba_prep/zamba_out/train_configuration.yaml

```
Please refer to the zamba github website for training and testing details:
https://github.com/drivendataorg/zamba
```
## 2 Module II: Behavior Recognition

### 2.1 Setup

```
Install all dependencies for the program
pip install -r ’requirements.txt’
```

### 2.2 Data Preparation

Our primary objective with data preparation was to split the dataset into multiple more specific datasets,
which we hypothesized would improve our test scores if we have different specialized models which are in-
voked based on Zamba’s prediction on which model is most applicable to the input. datasets/transformsss.py
has functionality to prepare the data through various means of transformation to ensure input consis-
tency.

2.2.1 Purging Redundancies

The data input is train.csv, with an accessory file ARmetadata.xlsx. The two files together define the
animals and actions found in each frame of each video.train.csv was minimized to train_light.csv
(and the same for val.csv) to remove redundant information.

2.2.2 Training Specialized Models

The data is split by running split_train.py which generates different data csv files for each subclass
of animal category.
```
split_train.py
```

### 2.3 Training and Testing

2.3.1 Training

Run main_split.py to train one model for each subclass of animal detected from the training data. One
can specify to perform this in parallel if there is sufficient hardware by using the argument

```
--parallel True
```
The output will consist of a checkpoint.pth file containing weights for evaluation for each specialized
model, located in Output/currentdate.

2.3.2 Testing

To test the model on the Animal Behavior dataset, you can run main.py, either full or split, with the
parameter

```
--train False
```
The validation data provided in the datasets will also be split during data preparation.

## 3 Module III: Evaluation

Before running evaluation, please modify the all relative file paths in evaluation.py and the weight path in main_eval.py
Please execute the following command:

```
!python3 evaluation.py
```
## 4 Access to AnimalKingdom dataset

Please refer to the AnimalKingdom github website:
https://sutdcv.github.io/Animal-Kingdom/

For all checkpoints files and results from zamba, please contact any one of the authors.
Link to zamba package: https://github.com/drivendataorg/zamba
Link to original MSQNet page: https://github.com/mondalanindya/MSQNet


