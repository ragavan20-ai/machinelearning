
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBClassifier


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("loan.csv")

df.columns = df.columns.str.strip()
df = df.drop_duplicates()
df = df.dropna(subset=["Loan_Status"])

print("DATASET SHAPE:", df.shape)

# ==========================================
# 2. PREPARE CLASSIFICATION DATA
# ==========================================

X = df.drop(
    columns=["Loan_Status", "Loan_ID"],
    errors="ignore"
)

le = LabelEncoder()

y = le.fit_transform(
    df["Loan_Status"].astype(str).str.strip()
)

print("Target classes:", le.classes_)

numeric_features = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]

categorical_features = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area"
]

# ==========================================
# 3. TRAIN / VALIDATION / TEST SPLIT
# ==========================================

# 60% training, 20% validation, 20% testing

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nDATA SPLIT")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))

# ==========================================
# 4. PREPROCESSING PIPELINE
# ==========================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# ==========================================
# 5. DEFINE CLASSIFICATION MODELS
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss"
    )
}

trained_models = {}

# ==========================================
# 6. TRAIN AND EVALUATE ALL CLASSIFIERS
# ==========================================

def evaluate_model(name, model, X_data, y_data):

    predictions = model.predict(X_data)

    print("\n", "=" * 45)
    print(name)
    print("=" * 45)

    print(
        "Accuracy:",
        round(accuracy_score(y_data, predictions), 3)
    )

    print(
        "Precision:",
        round(precision_score(
            y_data, predictions, zero_division=0
        ), 3)
    )

    print(
        "Recall:",
        round(recall_score(
            y_data, predictions, zero_division=0
        ), 3)
    )

    print(
        "F1 Score:",
        round(f1_score(
            y_data, predictions, zero_division=0
        ), 3)
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_data, predictions))


print("\n========== CLASSIFIER EVALUATION ==========")

for name, classifier in models.items():

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    trained_models[name] = pipeline

    evaluate_model(
        name,
        pipeline,
        X_val,
        y_val
    )

# ==========================================
# 7. ROC CURVES FOR THREE MODELS
# ==========================================

print("\n========== ROC CURVES AND AUC ==========")

roc_models = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost"
]

plt.figure(figsize=(9, 6))

auc_scores = {}

for name in roc_models:

    model = trained_models[name]

    probabilities = model.predict_proba(X_val)[:, 1]

    fpr, tpr, thresholds = roc_curve(
        y_val,
        probabilities
    )

    auc_value = roc_auc_score(
        y_val,
        probabilities
    )

    auc_scores[name] = auc_value

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {auc_value:.2f})"
    )

    print(name, "AUC:", round(auc_value, 3))

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")

plt.legend()
plt.grid(True)

plt.savefig("roc_curves.png")
plt.show()

# ==========================================
# 8. FIND THRESHOLD FOR AT LEAST 75% RECALL
# ==========================================

print("\n========== THRESHOLD TUNING ==========")

# Use the validation set to choose the best threshold.
# Choose the highest precision among thresholds
# that achieve recall >= 75%.

threshold_model = trained_models["Logistic Regression"]

val_probabilities = threshold_model.predict_proba(X_val)[:, 1]

thresholds = np.arange(0.01, 1.00, 0.01)

valid_thresholds = []

for threshold in thresholds:

    predictions = (
        val_probabilities >= threshold
    ).astype(int)

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    if recall >= 0.75:

        valid_thresholds.append(
            (threshold, precision, recall)
        )

if valid_thresholds:

    best_threshold = max(
        valid_thresholds,
        key=lambda x: (x[1], x[0])
    )

    print("Selected Threshold:", round(best_threshold[0], 2))
    print("Precision:", round(best_threshold[1], 3))
    print("Recall:", round(best_threshold[2], 3))

else:

    best_threshold = None

    print(
        "No threshold achieved 75% recall on this validation set."
    )

# ==========================================
# 9. GRIDSEARCHCV - RANDOM FOREST
# ==========================================

print("\n========== GRID SEARCH - RANDOM FOREST ==========")

rf_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    ))
])

rf_parameters = {
    "classifier__n_estimators": [50, 100, 150],
    "classifier__max_depth": [3, 5, None],
    "classifier__min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    rf_pipeline,
    param_grid=rf_parameters,
    cv=3,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

print("Best Random Forest Parameters:")
print(grid_search.best_params_)

print("Best Random Forest CV F1 Score:")
print(round(grid_search.best_score_, 3))

# ==========================================
# 10. RANDOMIZED SEARCH - XGBOOST
# ==========================================

print("\n========== RANDOMIZED SEARCH - XGBOOST ==========")

xgb_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    ))
])

xgb_parameters = {
    "classifier__n_estimators": [50, 100, 150, 200],
    "classifier__max_depth": [2, 3, 4, 5],
    "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "classifier__subsample": [0.8, 1.0]
}

random_search = RandomizedSearchCV(
    xgb_pipeline,
    param_distributions=xgb_parameters,
    n_iter=15,
    cv=3,
    scoring="f1",
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

best_xgb = random_search.best_estimator_

print("Best XGBoost Parameters:")
print(random_search.best_params_)

print("Best XGBoost CV F1 Score:")
print(round(random_search.best_score_, 3))

# ==========================================
# 11. COMPARE TUNED MODELS ON VALIDATION
# ==========================================

print("\n========== TUNED MODEL COMPARISON ==========")

candidate_models = {
    "Logistic Regression": trained_models["Logistic Regression"],
    "Decision Tree": trained_models["Decision Tree"],
    "Random Forest": best_rf,
    "XGBoost": best_xgb
}

validation_results = {}

for name, model in candidate_models.items():

    predictions = model.predict(X_val)

    probabilities = model.predict_proba(X_val)[:, 1]

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y_val,
        probabilities
    )

    validation_results[name] = {
        "F1": f1,
        "AUC": auc
    }

    print(
        name,
        "| F1:", round(f1, 3),
        "| ROC-AUC:", round(auc, 3)
    )

# Select by F1 first, then ROC-AUC.
best_name = max(
    validation_results,
    key=lambda name: (
        validation_results[name]["F1"],
        validation_results[name]["AUC"]
    )
)

best_model = candidate_models[best_name]

print("\nBEST MODEL:", best_name)

# ==========================================
# 12. REGRESSION EVALUATION
# ==========================================

print("\n========== REGRESSION EVALUATION ==========")

regression_features = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "Loan_Amount_Term",
    "Credit_History"
]

X_reg = df[regression_features]

y_reg = pd.to_numeric(
    df["LoanAmount"],
    errors="coerce"
)

valid = y_reg.notna()

X_reg = X_reg.loc[valid]
y_reg = y_reg.loc[valid]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)

regression_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

regression_pipeline.fit(Xr_train, yr_train)

reg_predictions = regression_pipeline.predict(Xr_test)

# Mean baseline: predict the training mean for every row
baseline_predictions = np.full(
    len(yr_test),
    yr_train.mean()
)

def regression_metrics(actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    r2 = r2_score(actual, predicted)

    return mae, rmse, r2


mae, rmse, r2 = regression_metrics(
    yr_test,
    reg_predictions
)

base_mae, base_rmse, base_r2 = regression_metrics(
    yr_test,
    baseline_predictions
)

print("\nLinear Regression Results")

print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R-squared:", round(r2, 3))

print("\nMean Baseline Results")

print("MAE:", round(base_mae, 2))
print("RMSE:", round(base_rmse, 2))
print("R-squared:", round(base_r2, 3))

# ==========================================
# 13. FINAL TEST EVALUATION - ONLY BEST MODEL
# ==========================================

print("\n========== FINAL TEST EVALUATION ==========")

print("Best model:", best_name)

# Test set is evaluated only once, after model selection.

test_probabilities = best_model.predict_proba(X_test)[:, 1]

if best_name == "Logistic Regression" and best_threshold is not None:

    test_predictions = (
        test_probabilities >= best_threshold[0]
    ).astype(int)

else:

    test_predictions = best_model.predict(X_test)

print("\nFinal Test Accuracy:",
      round(accuracy_score(y_test, test_predictions), 3))

print("Final Test Precision:",
      round(precision_score(
          y_test, test_predictions, zero_division=0
      ), 3))

print("Final Test Recall:",
      round(recall_score(
          y_test, test_predictions, zero_division=0
      ), 3))

print("Final Test F1:",
      round(f1_score(
          y_test, test_predictions, zero_division=0
      ), 3))

print("\nFinal Test Confusion Matrix:")
print(confusion_matrix(y_test, test_predictions))

print("\nFinal Test ROC-AUC:",
      round(roc_auc_score(y_test, test_probabilities), 3))

print("\nTASK 4 COMPLETED SUCCESSFULLY!")
