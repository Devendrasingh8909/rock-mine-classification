# Rock and Mine Classification Project

## Overview
This project aims to classify sonar data into two categories: rock and mine. Utilizing deep learning techniques, specifically a multi-layer perceptron (MLP), the model analyzes features from sonar signals to accurately identify the type of material present.

## Table of Contents
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Technologies Used
- **Programming Language**: Python
- **Deep Learning Framework**: TensorFlow (with Keras)
- **Data Manipulation**: Pandas, NumPy
- **Data Visualization**: Matplotlib
- **Machine Learning Libraries**: Scikit-learn

## Dataset
The dataset used for this project is the Sonar dataset, which consists of sonar signals used to identify objects underwater. The dataset includes:
- **Features**: 60 sonar measurements
- **Labels**: 2 classes (rock and mine)

The dataset can be found at [Sonar Dataset](https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mineral+Detection)).

## Features
- **Data Preprocessing**: The dataset is cleaned, shuffled, and split into training and testing sets. Labels are encoded using one-hot encoding.
- **Model Training**: A deep learning model is trained to classify the sonar signals, with features normalized for better performance.
- **Evaluation**: The model's accuracy is evaluated on a separate test set.

## Model Architecture
The model consists of:
- Input Layer: 60 neurons
- Hidden Layers: 3 layers with 60 neurons each, using ReLU and Sigmoid activation functions
- Output Layer: 2 neurons with a softmax activation function for classification

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rock-mine-classification.git
