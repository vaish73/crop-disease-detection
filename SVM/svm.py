import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# -----------------------------
# LOAD DATASET
# -----------------------------
data = pd.read_excel("rice_dataset.xlsx")

print("Dataset Loaded Successfully\n")

# -----------------------------
# PREPROCESSING
# -----------------------------
# data = data.replace({
#     "yes": 1,
#     "no": 0,
#     "normal": 0,
#     "abnormal": 1
# })

data = data.replace({
    "yes": 1,
    "no": 0,
    "normal": 0,
    "abnormal": 1,
    "o": 0,
    "O": 0
})


print("Preprocessing Completed\n")

data = data.fillna(0)


data = data.infer_objects(copy=False)

# -----------------------------
# INPUT FEATURES
# -----------------------------
X = data.drop(columns=["Disease"])

# -----------------------------
# OUTPUT LABELS
# -----------------------------
y = data["Disease"]

print("Features Shape:", X.shape)
print("Labels Shape:", y.shape)

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain-Test Split Completed")

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# -----------------------------
# CREATE SVM MODEL
# -----------------------------
svm_model = SVC(kernel='linear')

print("\nSVM Model Created")

# -----------------------------
# TRAIN MODEL
# -----------------------------
print("\nTraining Started...")

svm_model.fit(X_train, y_train)

print("Training Completed")

# -----------------------------
# PREDICTION
# -----------------------------
y_pred = svm_model.predict(X_test)

print("\nPrediction Completed")

# -----------------------------
# ACCURACY
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy * 100)