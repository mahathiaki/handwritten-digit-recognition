
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense
def calculate_metrics(y_true, y_pred):
    cm = np.zeros((10, 10), dtype=int)

    for actual, predicted in zip(y_true, y_pred):
        cm[actual, predicted] += 1

    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    precision = np.divide(
        tp, tp + fp,
        out=np.zeros(10, dtype=float),
        where=(tp + fp) != 0
    )

    recall = np.divide(
        tp, tp + fn,
        out=np.zeros(10, dtype=float),
        where=(tp + fn) != 0
    )

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(10, dtype=float),
        where=(precision + recall) != 0
    )

    weights = cm.sum(axis=1) / len(y_true)

    accuracy = np.trace(cm) / len(y_true)
    weighted_precision = np.sum(precision * weights)
    weighted_recall = np.sum(recall * weights)
    weighted_f1 = np.sum(f1 * weights)

    return (
        accuracy,
        weighted_precision,
        weighted_recall,
        weighted_f1,
        cm
    )
# Create folders
os.makedirs("model", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)

# Step 1: Load MNIST dataset
print("Loading MNIST dataset...")

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Training images:", X_train.shape)
print("Testing images:", X_test.shape)

# Step 2: Display sample images
plt.figure(figsize=(10, 4))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i], cmap="gray")
    plt.title(f"Digit: {y_train[i]}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("screenshots/sample_digits.png")
plt.show()

# Step 3: Normalize images
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

print("Image preprocessing completed.")

# Step 4: Build ANN model
model = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax")
])

# Step 5: Compile ANN
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nANN Model Summary:")
model.summary()

# Step 6: Train ANN
print("\nStarting model training...")

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1
)

# Step 7: Evaluate model
test_loss, test_accuracy = model.evaluate(
    X_test, y_test, verbose=1
)

# Step 8: Predict test images
y_prob = model.predict(X_test)
y_pred = np.argmax(y_prob, axis=1)

# Step 9: Calculate performance metrics
accuracy, precision, recall, f1, cm = calculate_metrics(
    y_test, y_pred
)

print("\n===== PERFORMANCE RESULTS =====")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Step 10: Plot training graphs
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("ANN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("ANN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("screenshots/training_graphs.png")
plt.show()

# Step 11: Confusion matrix

plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Digit")
plt.ylabel("Actual Digit")
plt.xticks(range(10))
plt.yticks(range(10))
plt.colorbar()

for i in range(10):
    for j in range(10):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig("screenshots/confusion_matrix.png")
plt.show()

# Step 12: Save trained model
model.save("model/digit_model.keras")

print("\nANN model saved successfully!")
print("Training completed!")
