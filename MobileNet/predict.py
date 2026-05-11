import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# -----------------------------
# LOAD SAVED MODEL
# -----------------------------
model = tf.keras.models.load_model("rice_disease_mobilenet.h5")

# -----------------------------
# CLASS NAMES
# -----------------------------
class_names = [
    'Bacterial Leaf Blight',
    'Brown Spot',
    'Healthy Rice Leaf',
    'Leaf Blast',
    'Leaf scald',
    'Sheath Blight'
]

# -----------------------------
# IMAGE PATH
# -----------------------------
img_path = "test2.jpg"

# -----------------------------
# LOAD IMAGE
# -----------------------------
img = image.load_img(
    img_path,
    target_size=(224, 224)
)

# -----------------------------
# IMAGE TO ARRAY
# -----------------------------
img_array = image.img_to_array(img)

# -----------------------------
# EXPAND DIMENSIONS
# -----------------------------
img_array = np.expand_dims(img_array, axis=0)

# -----------------------------
# NORMALIZATION
# -----------------------------
img_array = img_array / 127.5 - 1

# -----------------------------
# PREDICTION
# -----------------------------
prediction = model.predict(img_array)

predicted_class = class_names[np.argmax(prediction)]

confidence = np.max(prediction) * 100

# -----------------------------
# OUTPUT
# -----------------------------
print("\nPredicted Disease:", predicted_class)

print("Confidence:", confidence)