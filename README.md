# 🧬 Diabetes Prediction — Single-Layer Perceptron

**Binary classification using a Perceptron neural network on the Pima Indians Diabetes Dataset**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg)](https://www.tensorflow.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Pima%20Indians%20Diabetes-green.svg)](https://www.kaggle.com/uciml/pima-indians-diabetes-database)

---

## 📌 Overview

This project implements a **Perceptron** — the simplest neural network architecture — to predict whether a patient has diabetes based on 8 clinical features. The model is built in Keras and trained with a binary cross-entropy loss and SGD optimizer.

University of Adelaide · Deep Learning Assessment 1 · **Grade: 74.5 / 100**

---

## 📊 Dataset — Pima Indians Diabetes

- **Source:** `diabetes.csv` (Pima Indians Diabetes Database)
- **Samples:** 768 patients
- **Features:** 8 clinical measurements (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
- **Target:** `Outcome` — 1 (diabetic) / 0 (non-diabetic)
- **Split:** 80% train / 20% test (`random_state=42`)

---

## 🏗️ Model Architecture

```python
model = Sequential()
model.add(Dense(1, input_dim=8, activation='sigmoid'))  # Single Perceptron
model.compile(
    optimizer=SGD(learning_rate=0.2),
    loss=BinaryCrossentropy(),
    metrics=['accuracy']
)
```

| Parameter | Value |
|---|---|
| Architecture | Single Dense layer (Perceptron) |
| Activation | Sigmoid |
| Optimizer | SGD (lr = 0.2) |
| Loss | Binary Cross-Entropy |
| Epochs | 30 |
| Batch size | 10 |

---

## ⚙️ Preprocessing

- **Standardisation:** `StandardScaler` applied to all 8 input features
- **Label:** `Outcome` column — binary (0 / 1)

---

## 📈 Evaluation Metrics

- **Accuracy** — percentage of correct predictions
- **F1-Score** — harmonic mean of precision and recall
- **Confusion Matrix** — visualised with Seaborn heatmap

---

## 🗂️ Key Code

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Preprocess
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model.fit(X_train, y_train, epochs=30, batch_size=10, validation_data=(X_test, y_test))

# Evaluate
y_pred = (model.predict(X_test) > 0.5).astype("int32")
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
```

---

## 🚀 How to Run

```bash
git clone https://github.com/chandr-csgp/DeepLearning.git
cd DeepLearning

pip install tensorflow scikit-learn pandas seaborn matplotlib

python Keras.py
```

---

## 🔧 Tech Stack

`Python` · `TensorFlow` · `Keras` · `scikit-learn` · `Pandas` · `NumPy` · `Seaborn`

---

## 👤 Author

**Chandra Sekar Putta** — University of Adelaide, MDS 2024–2026  
[LinkedIn](https://linkedin.com/in/chandra-sekar-p-b4b033402) · [GitHub](https://github.com/chandr-csgp)
