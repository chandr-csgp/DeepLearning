import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('diabetes.csv')  # Update with the correct path
print(data)

# Preprocessing
X = data.drop('Outcome', axis=1)  # Features
y = data['Outcome']  # Labels

# Standardize the dataset
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple neural network (Perceptron-like)
model = Sequential()
model.add(Dense(1, input_dim=X_train.shape[1], activation='sigmoid'))  # Sigmoid activation for binary classification

# Compile the model with a chosen optimizer and loss function
model.compile(optimizer=SGD(learning_rate=0.2), loss=BinaryCrossentropy(), metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=10, validation_data=(X_test, y_test))

# Training accuracy
train_acc = history.history['accuracy'][-1]  # Get the final training accuracy from the history
print(f"Final Training Accuracy: {train_acc * 100:.2f}%")


# Evaluate the model
y_pred = (model.predict(X_test) > 0.5).astype("int32")  # Threshold the sigmoid output
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Output results

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'F1-Score: {f1:.2f}')
print("Confusion Matrix:\n", cm)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

