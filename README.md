# 🧠 Deep Learning — Image Classification with CNN

**Multi-class image classification using Convolutional Neural Networks (93.25% accuracy)**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red.svg)](https://keras.io/)

---

## 📌 Overview

This project implements a **Convolutional Neural Network (CNN)** for multi-class image classification using TensorFlow and Keras. The model achieves **93.25% test accuracy** through systematic architecture design and hyperparameter tuning.

---

## 🏗️ Model Architecture

- **Conv layers:** Multiple convolutional blocks with ReLU activation
- **Pooling:** MaxPooling after each block
- **Regularisation:** Dropout to prevent overfitting
- **Output:** Softmax for multi-class probability distribution
- **Optimiser:** Adam with tuned learning rate

---

## 📊 Results

| Metric | Value |
|---|---|
| Test Accuracy | **93.25%** |
| Validation Method | Cross-validation |
| Key Technique | Dropout regularisation + learning rate tuning |

---

## 🔧 Tech Stack

`Python` · `TensorFlow` · `Keras` · `NumPy` · `Matplotlib`

---

## 🚀 How to Run

```bash
git clone https://github.com/chandr-csgp/DeepLearning.git
cd DeepLearning
pip install tensorflow numpy matplotlib
python Keras.py
```

---

## 👤 Author

**Chandra Sekar Putta** — University of Adelaide, MDS 2024–2026  
[LinkedIn](https://linkedin.com/in/chandra-sekar-p-b4b033402) · [GitHub](https://github.com/chandr-csgp)
