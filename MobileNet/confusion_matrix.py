import tensorflow as tf
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

# -----------------------------
# LOAD MODEL
# -----------------------------
model = tf.keras.models.load_model(
    "rice_disease_mobilenet.h5"
)

# -----------------------------
# DATASET PATH
# -----------------------------
dataset_path = "Rice_Leaf_AUG"

# -----------------------------
# LOAD VALIDATION DATASET
# -----------------------------
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

# -----------------------------
# CLASS NAMES
# -----------------------------
class_names = validation_dataset.class_names

print("\nClasses:")
print(class_names)

# -----------------------------
# TRUE LABELS
# -----------------------------
y_true = np.concatenate(
    [y for x, y in validation_dataset],
    axis=0
)

# -----------------------------
# PREDICTIONS
# -----------------------------
predictions = model.predict(validation_dataset)

y_pred = np.argmax(predictions, axis=1)

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

# -----------------------------
# PLOT CONFUSION MATRIX
# -----------------------------
plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.title("Confusion Matrix - MobileNetV2")

plt.tight_layout()

# -----------------------------
# SAVE IMAGE
# -----------------------------
plt.savefig("mobilenet_confusion_matrix.png")

print("\nConfusion Matrix Saved Successfully")