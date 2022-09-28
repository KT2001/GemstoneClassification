# GemstoneClassification

## Introduction

* This Project Is To Implement Gemstone classification By Cnn. Used Basic Pretrained Model To Train Real Life Image classification.
* Libraries Used In This Project tensorflow ,Matplotlib, numpy, Pytorch, PIL, pathlib.
* This Model has achieved 95% Accuracy on Training data and 70% accuracy on Testing data.
* This is the First and Basic version Of this Project.

## Project workflow

* Downloading the data: The data is downloaded from [Kaggle](https://www.kaggle.com/datasets/lsind18/gemstones-images)
    - Train Data:- https://github.com/KT2001/GemstoneClassification/tree/main/train
    - Test Data:- https://github.com/KT2001/GemstoneClassification/tree/main/test

* Project code:- https://github.com/KT2001/GemstoneClassification/blob/main/GemstoneClassification.ipynb

* Saved model:- https://github.com/KT2001/GemstoneClassification/blob/main/GemstoneClassificationModel.pth

## Major Libraries and Frameworks

* torch, torchvision: These are pytorch AI and deeplearning libraries with torchvision prominently being used in computer vision projects.

* numpy, matplotlib, PIL: These are used for data analysis, and data visualization.

## Project Tasks

* Download and setup dataset.

* Perform data analysis for this project.

* Setup a pretrained model in this case the model being used is [EfficientNet-B0](https://arxiv.org/abs/1905.11946v5).

* Transforming the data.

* Freezing the model layers and training the top layers on our custom data.

* Evaluating the result.

## Future Goals

* **Reducing the code complexity** - As of now the project is in extremely simple state, in future versions the project code can be more simplified and cleaned as such to make it for user friendly.

* **Increasing the efficiency** - currently the model is in basic state the accuracy can be something which we can improve upon by fine tuning the model, data augumentation, and many other experimental changes.

* **Deploy the model as a web application** - As of now the ipynb file can be run in google colab or the local machine and the model itself can be downloaded and be run on any device, but in future can work on turning it into a web application so that even people without a programming background can efficiently make use of this.