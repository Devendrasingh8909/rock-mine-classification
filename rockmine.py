import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os

# Reading Dataset
def read_dataset(file_path):
    df = pd.read_csv(file_path)
    print("Dataset Shape:", df.shape)
    
    X = df.iloc[:, :60].values  # Features
    y = df.iloc[:, 60].values    # Labels

    # Encode the dependent variable
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    Y = one_hot_encode(y_encoded)
    
    print("Features Shape:", X.shape)
    return X, Y

# One-hot encoding function
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encoded = np.zeros((n_labels, n_unique_labels))
    one_hot_encoded[np.arange(n_labels), labels] = 1
    return one_hot_encoded

# File path for the dataset
file_path = 'C:/Users/rajiv/Downloads/rockAndMineClassification_DL/dataSet.csv'

# Read the Dataset
X, Y = read_dataset(file_path)

# Shuffle and split the dataset into train and test sets
X, Y = shuffle(X, Y, random_state=1)
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.20, random_state=415)

# Check shapes of train and test sets
print("\nTrain and Test Shape")
print("Train X Shape:", train_x.shape)
print("Train Y Shape:", train_y.shape)
print("Test X Shape:", test_x.shape)
print("Test Y Shape:", test_y.shape)

# Model parameters
learning_rate = 0.001  # Adjusted learning rate for better convergence
training_epochs = 1000
n_dim = X.shape[1]
n_class = 2

# Define the model using Keras
model = tf.keras.Sequential([
    tf.keras.layers.Dense(60, activation='relu', input_shape=(n_dim,)),  # Input layer
    tf.keras.layers.Dropout(0.5),                                       # Dropout for regularization
    tf.keras.layers.Dense(60, activation='relu'),                       # Hidden layer
    tf.keras.layers.Dense(60, activation='relu'),                       # Hidden layer
    tf.keras.layers.Dense(n_class, activation='softmax')                # Output layer
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),  # Use Adam optimizer
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_x, train_y, epochs=training_epochs, batch_size=10, verbose=1, validation_data=(test_x, test_y))

# Plot loss and accuracy graphs
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_x, test_y, verbose=2)
print("Test Accuracy: %.4f" % test_accuracy)
