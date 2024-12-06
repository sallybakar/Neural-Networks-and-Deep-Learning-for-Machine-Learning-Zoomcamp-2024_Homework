
---

# Neural Networks and Deep Learning Homework - Hair Type Classification

This repository contains my work for the **Neural Networks and Deep Learning for Machine Learning Zoomcamp 2024**. The task involves building a Convolutional Neural Network (CNN) from scratch to classify hair types (straight or curly). The dataset, architecture, and evaluation metrics are described below.

---

## Table of Contents
- [Task Overview](#Task-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Setup and Installation](#setup-and-installation)
- [Data Preparation](#data-preparation)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

---

## Project Overview
The objective of this project is to classify hair types (straight vs. curly) using a CNN. The model is trained from scratch on a dataset of hair images, implementing data augmentation techniques to improve model generalization.

---

## Dataset
The dataset contains approximately 1,000 images of straight and curly hair types, divided into training and test sets. It can be downloaded from the [release link here](https://github.com/SVizor42/ML_Zoomcamp/releases/download/straight-curly-data/data.zip).

---

## Model Architecture
The Convolutional Neural Network (CNN) is built using TensorFlow/Keras with the following structure:

1. **Input Layer**: Shape `(200, 200, 3)`
2. **Convolutional Layer**:
   - Filters: 32
   - Kernel size: `(3, 3)`
   - Activation: ReLU
3. **Max Pooling Layer**:
   - Pool size: `(2, 2)`
4. **Flatten Layer**
5. **Dense Layer**:
   - Neurons: 64
   - Activation: ReLU
6. **Output Layer**:
   - Neurons: 1
   - Activation: Sigmoid (for binary classification)

Optimizer: SGD (Stochastic Gradient Descent)  
- Learning Rate: `0.002`  
- Momentum: `0.8`

Loss Function: Binary Cross-Entropy

---

## Setup and Installation

### **Environment Requirements**
This project requires a Python environment with the following libraries:
- `numpy`
- `tensorflow` (v2.17.1)
- `matplotlib`
- `Pandas`

Ensure you have a GPU-enabled setup for efficient training. Tools like [Saturn Cloud](https://saturncloud.io/) or Google Colab can be used for GPU access.

### **Installation**
Clone this repository:
```bash
git clone https://github.com/sallybakar/Neural-Networks-and-Deep-Learning-for-Machine-Learning-Zoomcamp-2024_Homework.git
cd Neural-Networks-and-Deep-Learning-for-Machine-Learning-Zoomcamp-2024_Homework
```

Install the dependencies:
```bash
pip install -r requirements.txt
```

---

## Data Preparation
1. Download the dataset:
   ```bash
   wget https://github.com/SVizor42/ML_Zoomcamp/releases/download/straight-curly-data/data.zip
   unzip data.zip
   ```
2. Rescale the images to normalize pixel values between 0 and 1:
   ```python
   ImageDataGenerator(rescale=1./255)
   ```

---

## Training and Evaluation
1. Train the model for 10 epochs with data augmentation:
   - Rotation: 50 degrees
   - Width/Height shift: 0.1
   - Zoom range: 0.1
   - Horizontal Flip: Enabled
2. Use the `.fit()` method for training:
   ```python
   model.fit(
       train_generator,
       epochs=10,
       validation_data=test_generator
   )
   ```
3. Evaluate the model performance on the test set using metrics like accuracy and loss.

---

## Results
### **Model Performance**
- **Total Parameters**: 20,073,473  
- **Median Training Accuracy**: ~0.72  
- **Standard Deviation of Training Loss**: ~0.128  
- **Mean Test Loss with Augmentation**: ~0.56  
- **Average Test Accuracy (Last 5 Epochs)**: ~0.71  

---

## Acknowledgements
This project is part of the **Neural Networks and Deep Learning for Machine Learning Zoomcamp 2024**. Special thanks to the instructors and Kaggle for providing the dataset.

---

