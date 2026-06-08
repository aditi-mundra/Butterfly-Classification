# Butterfly Species Classification

A deep learning project for butterfly species classification using transfer learning and computer vision techniques.

## Project Structure

```text
butterfly-classification/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│
├── outputs/
│   ├── figures/
│   ├── models/
│   └── reports/
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

## Objective

The goal of this project is to classify butterfly species from images using convolutional neural networks and transfer learning.

## Dataset Overview

- Dataset: Leeds Butterfly Dataset
- Images: 832
- Species: 10
- Image Resolution Used: 224 × 224

## Exploratory Data Analysis

### Class Distribution

![Class Distribution](outputs/figures/class_distribution.png)

### Sample Butterfly Images

![Sample Images](outputs/figures/sample_images.png)

## Data Preparation

- Label Encoding
- One-Hot Encoding
- Stratified Train/Test Split
- Image Normalization

## Model Architecture

- Backbone: Xception (ImageNet Pretrained)
- Transfer Learning
- Global Feature Extraction
- Dense Classification Head
- Dropout Regularization

## Model

Backbone: Xception (ImageNet Pretrained)

- Transfer Learning
- Frozen Feature Extractor
- Dense Classification Head
- Dropout Regularization
- 10 Butterfly Classes