# 🌥️ Cloud Image Classification with PyTorch
---
This project implements a Convolutional Neural Network (CNN) to classify cloud images into 7 categories. It covers the full deep learning workflow — from preprocessing and training to evaluation and inference — using PyTorch.
---
# 📌 Project Overview
---
The goal of this project is to automatically recognize and classify different types of clouds from images. By training a CNN model on a labeled dataset, the system learns to identify cloud categories based on visual features.

**Key stages include:**

Data preprocessing and augmentation

Building a CNN model from scratch

Training with cross-entropy loss and Adam optimizer

Evaluating using metrics like Accuracy, Precision, Recall, and F1 Score

Visualizing model performance with confusion matrix and loss curves

Saving and reloading the trained model for inference
---
# 📂 Dataset
---
Training set: 474 cloud images stored in structured folders (by class).

Testing set: Separate set of unseen cloud images.

Images are resized to 128x128 pixels and normalized.

Data Augmentation applied:

Random horizontal flipping

Random rotation

Color jitter (brightness, contrast, saturation)

Normalization
---
# 🏗️ Model Architecture
---
The CNN consists of:

Feature extractor:

3 convolutional blocks with ELU activation + max pooling

Adaptive average pooling

Flattening into feature vectors

Classifier:

Fully connected layers with dropout for regularization

Final softmax layer for 7-class prediction
---
# ⚙️ Training
---
Loss function: CrossEntropyLoss

Optimizer: Adam (learning rate = 0.001)

Epochs: 100

Training and validation losses were tracked and visualized across epochs.
---
# 📊 Results
---
After 100 epochs, the model achieved:

Accuracy: 78.14%

Precision: 80.39%

Recall: 78.14%

F1 Score: 78.24%

A confusion matrix was generated to visualize class-level performance.
---
# 📈 Visualizations
---
Training vs Validation Loss curves

Confusion Matrix (heatmap)

These plots help track the model’s learning progress and highlight areas for improvement.
---
# 💾 Model Saving & Loading
---
The trained model is saved as cloud_classifier.pth.
It can be reloaded later for evaluation or prediction without retraining.
---
# 🔮 Inference
---
A utility function allows prediction on new cloud images.
Steps:

Load the trained model

Apply the same preprocessing transforms

Pass the image through the model

Output the predicted cloud class
