
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# -----------------------------------------
# 1. LOAD DATASET
# -----------------------------------------

df = pd.read_csv("loan.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

# Change this to your actual target column
TARGET = "Loan_Status"

if TARGET not in df.columns:
    raise ValueError(
        f"Target '{TARGET}' not found. Available columns: {list(df.columns)}"
    )

# -----------------------------------------
# 2. DATA PREPROCESSING
# -----------------------------------------

df = df.drop_duplicates()
df = df.dropna(subset=[TARGET])

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Encode categorical columns
X = pd.get_dummies(X, drop_first=True)

# Encode target if it contains text
if y.dtype == "object":
    le = LabelEncoder()
    y = le.fit_transform(y.astype(str))

# Remove infinite values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

# -----------------------------------------
# 3. TRAIN / VALIDATION / TEST SPLIT
# -----------------------------------------

# First split: 60% train, 40% remaining
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

# Second split: 20% validation, 20% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nDataset Split:")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))

# Keep test set aside until final evaluation

# -----------------------------------------
# 4. SHOW OVERFITTING USING DECISION TREE
# -----------------------------------------

depths = [1, 3, 5, 10, None]

train_scores = []
val_scores = []

print("\nDECISION TREE RESULTS")

for depth in depths:

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)

    train_scores.append(train_acc)
    val_scores.append(val_acc)

    print(
        f"Depth {depth}: "
        f"Train Accuracy = {train_acc:.2f}, "
        f"Validation Accuracy = {val_acc:.2f}"
    )

# Plot train vs validation accuracy

plt.figure(figsize=(9, 5))

depth_labels = ["1", "3", "5", "10", "None"]

plt.plot(depth_labels, train_scores, marker="o",
         label="Training Accuracy")

plt.plot(depth_labels, val_scores, marker="s",
         label="Validation Accuracy")

plt.xlabel("Decision Tree Depth")
plt.ylabel("Accuracy")
plt.title("Overfitting: Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("overfitting.png")
plt.show()

# -----------------------------------------
# 5. LOGISTIC REGRESSION - 5 FOLD CV
# -----------------------------------------

# Fit scaler only on training data
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

lr = LogisticRegression(max_iter=1000)

cv_scores = cross_val_score(
    lr,
    X_train_scaled,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("\nLOGISTIC REGRESSION 5-FOLD CV")

print("Individual Scores:", cv_scores)

print(
    f"CV: {cv_scores.mean():.2f} +/- "
    f"{cv_scores.std():.2f}"
)

# -----------------------------------------
# 6. DATA LEAKAGE DEMONSTRATION
# -----------------------------------------

print("\nDATA LEAKAGE DEMONSTRATION")

# WRONG WAY: Scale before splitting
wrong_scaler = StandardScaler()

X_wrong = wrong_scaler.fit_transform(X)

Xw_train, Xw_test, yw_train, yw_test = train_test_split(
    X_wrong,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

wrong_model = LogisticRegression(max_iter=1000)

wrong_model.fit(Xw_train, yw_train)

wrong_pred = wrong_model.predict(Xw_test)

wrong_accuracy = accuracy_score(yw_test, wrong_pred)

print("Accuracy with leakage:", round(wrong_accuracy, 2))

# CORRECT WAY: Split first, fit scaler on training only

correct_scaler = StandardScaler()

X_train_correct = correct_scaler.fit_transform(X_train)
X_val_correct = correct_scaler.transform(X_val)
X_test_correct = correct_scaler.transform(X_test)

correct_model = LogisticRegression(max_iter=1000)

correct_model.fit(X_train_correct, y_train)

correct_pred = correct_model.predict(X_test_correct)

correct_accuracy = accuracy_score(y_test, correct_pred)

print("Accuracy without leakage:", round(correct_accuracy, 2))

# -----------------------------------------
# 7. SUPERVISED / UNSUPERVISED CLASSIFICATION
# -----------------------------------------

print("\nML PROBLEM CLASSIFICATION")

problems = {
    "Spam Detection": "Supervised Learning",
    "Customer Segmentation": "Unsupervised Learning",
    "House Price Prediction": "Supervised Learning",
    "Anomaly Detection": "Unsupervised Learning"
}

for problem, category in problems.items():
    print(problem, ":", category)

# -----------------------------------------
# 8. SEVEN ML WORKFLOW STEPS
# -----------------------------------------

print("\n7 STEPS OF MACHINE LEARNING WORKFLOW")

steps = [
    "1. Problem Definition",
    "2. Data Collection",
    "3. Data Cleaning and Preprocessing",
    "4. Exploratory Data Analysis (EDA)",
    "5. Feature Engineering and Model Training",
    "6. Model Evaluation and Validation",
    "7. Deployment and Monitoring"
]

for step in steps:
    print(step)

# -----------------------------------------
# 9. FINAL TEST EVALUATION
# -----------------------------------------

print("\nFINAL TEST RESULTS")

print("Test Accuracy:", round(correct_accuracy, 2))

print("\nML Workflow Completed Successfully!")
