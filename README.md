# 🚗 Driver Drowsiness Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

This project aims to develop a robust deep learning model for detecting driver drowsiness from facial images to improve road safety. The system integrates a custom **MobileNetV2-based architecture (DrowsyNet)** with **MediaPipe** for real-time inference and utilizes **Bayesian Optimization** for hyperparameter tuning.

## Project Overview

Driver drowsiness is a critical factor in traffic accidents. This study implements a computer vision-based approach to automatically classify the driver's state as "Drowsy" or "Non-Drowsy" (Alert). Unlike simple heuristic methods, this project combines deep learning features with geometric analysis (Eye Aspect Ratio) for higher reliability.

### Key Features

* **High Performance:** Achieved a validation accuracy of **93.06%** and a low validation loss of ~0.21.
* **Hybrid Detection Logic:** Combines CNN-based facial classification with Eye Aspect Ratio (EAR) estimation using MediaPipe Face Mesh.
* **Bayesian Optimization:** Automated fine-tuning of Learning Rate, Weight Decay, and Dropout Rate to maximize model generalization.
* **Real-Time Alert System:**
    * **Stage 1:** "Take a short break!" (after 3 seconds of drowsiness).
    * **Stage 2:** "SERIOUS ALERT: Pull Over Immediately" (after 6 seconds).
* **Robust Dataset:** Trained on a combined dataset of 35,997 training and 8,940 validation images sourced from multiple repositories.

## Methodology & Architecture

### 1. DrowsyNet Model
The core classifier uses **MobileNetV2** as a backbone due to its balance of accuracy and computational efficiency.
* **Transfer Learning:** Frozen base layers of pre-trained MobileNetV2 with a custom trainable classification head.
* **Architecture:** Feature Extractor $\rightarrow$ Adaptive Avg Pooling $\rightarrow$ Dropout $\rightarrow$ Fully Connected Layer (2 classes).
* **Optimization:** Hyperparameters were selected via Bayesian Optimization (iterative search) rather than manual tuning.

### 2. Real-Time Pipeline
The system processes webcam feeds through the following steps:
1.  **Face Detection:** MediaPipe Face Mesh identifies 468 facial landmarks.
2.  **Geometric Analysis:** Calculates EAR (Eye Aspect Ratio). If $EAR < 0.25$, eyes are considered closed.
3.  **CNN Inference:** The face region is cropped, transformed, and passed to DrowsyNet. Drowsiness is flagged if confidence > 0.6.
4.  **Temporal Smoothing:** A rolling window (deque) checks for sustained drowsiness to reduce false positives.

## Results

The model demonstrates strong generalization capabilities without overfitting.

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Accuracy** | **93.06%** | Overall validation accuracy |
| **Precision** | 0.94 (Drowsy) | Ratio of correct positive predictions |
| **Recall** | 0.93 (Drowsy) | Ratio of actual positives identified |
| **F1-Score** | 0.93 | Harmonic mean of precision and recall |

**Confusion Matrix:**
* **True Positives (Drowsy):** 4417
* **True Negatives (Non-Drowsy):** 3903
* **False Negatives:** 343

## Installation

To run this project locally, clone the repository and install the dependencies.

1.  **Clone the Repo**
    ```bash
    git clone https://github.com/sevvallaydogann/Driver-Drowsiness-Detection.git
    cd Driver-Drowsiness-Detection
    ```

2.  **Install Requirements**
    ```bash
    pip install torch torchvision opencv-python mediapipe numpy scikit-learn bayesian-optimization matplotlib
    ```

## Usage

### 1. Training (Optional)
The training scripts handle dataset loading, augmentation, and Bayesian Optimization.
```bash
jupyter notebook driver_drowsiness_detection.ipynb
```
## 2. Real-Time Detection
To start the webcam and detection system, load the trained weights (best_drowsy_model.pth) by running the inference notebook:
```bash
jupyter notebook real_time_webcam.ipynb
```
Note: Open the notebook and run all cells to activate the webcam.

* Status Indicators: The screen displays "DROWSY" (Red) or "Normal" (Green).
* Exit: Press 'q' to quit the application.





