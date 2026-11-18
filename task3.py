
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.base import clone

from xgboost import XGBClassifier

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("loan.csv")

df.columns = df.columns.str.strip()
df = df.drop_duplicates()
df = df.dropna(subset=["Loan_Status"])

print("DATASET INFORMATION")
print("Shape:", df.shape)
print(df.head())

# ==========================================
# 2. PREPARE CLASSIFICATION DATA
# ==========================================

X = df.drop(columns=["Loan_Status", "Loan_ID"], errors="ignore")
y = df["Loan_Status"].astype(str).str.strip()

# Encode target
le = LabelEncoder()
y = le.fit_transform(y)

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

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
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
# 5. FOUR CLASSIFICATION MODELS
# ==========================================

print("\n========== CLASSIFICATION MODELS ==========")

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        oob_score=True,
        bootstrap=True
    )
}

trained_models = {}

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessing", clone(preprocessor)),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)

    trained_models[name] = pipeline

    print(f"\n{name}")
    print("Validation Accuracy:", round(accuracy, 2))

    if name == "Random Forest":
        print("OOB Score:", round(
            pipeline.named_steps["classifier"].oob_score_, 2
        ))

# ==========================================
# 6. XGBOOST WITH EARLY STOPPING
# ==========================================

print("\n========== XGBOOST ==========")

# Fit preprocessing on training data only.
# Use the same fitted preprocessing for validation data.

xgb_preprocessor = clone(preprocessor)

X_train_xgb = xgb_preprocessor.fit_transform(X_train, y_train)
X_val_xgb = xgb_preprocessor.transform(X_val)

xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
    eval_metric="logloss",
    early_stopping_rounds=10
)

xgb_pipeline = Pipeline([
    ("preprocessing", clone(preprocessor)),
    ("classifier", xgb_model)
])

xgb_pipeline.fit(
    X_train,
    y_train,
    classifier__eval_set=[(X_val_xgb, y_val)],
    classifier__verbose=False
)

xgb_predictions = xgb_pipeline.predict(X_val)

print("XGBoost Validation Accuracy:",
      round(accuracy_score(y_val, xgb_predictions), 2))

trained_models["XGBoost"] = xgb_pipeline

# ==========================================
# 7. LOGISTIC REGRESSION PROBABILITIES
# ==========================================

print("\n========== LOGISTIC REGRESSION PROBABILITIES ==========")

lr_model = trained_models["Logistic Regression"]

first_five = X_test.iloc[:5]

probabilities = lr_model.predict_proba(first_five)
predictions = lr_model.predict(first_five)

print("\nProbability Predictions:")
print(probabilities)

print("\nPredicted Classes:")
print(predictions)

print("\nActual Classes:")
print(y_test[:5])

print("\nProbability and Prediction Comparison:")

for i in range(len(first_five)):

    print(
        f"Row {i+1}: "
        f"Probabilities = {probabilities[i]}, "
        f"Predicted = {predictions[i]}"
    )

# ==========================================
# 8. TEST ACCURACY OF ALL MODELS
# ==========================================

print("\n========== FINAL TEST ACCURACY ==========")

for name, pipeline in trained_models.items():

    test_predictions = pipeline.predict(X_test)

    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    print(name, "Test Accuracy:", round(test_accuracy, 2))

# ==========================================
# 9. RANDOM FOREST FEATURE IMPORTANCE
# ==========================================

print("\n========== RANDOM FOREST TOP 5 FEATURES ==========")

rf_pipeline = trained_models["Random Forest"]

rf_model = rf_pipeline.named_steps["classifier"]

feature_names = (
    rf_pipeline.named_steps["preprocessing"]
    .get_feature_names_out()
)

rf_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_model.feature_importances_
})

rf_importance = rf_importance.sort_values(
    by="Importance",
    ascending=False
)

print(rf_importance.head(5).to_string(index=False))

# ==========================================
# 10. XGBOOST TOP 5 FEATURE IMPORTANCE
# ==========================================

print("\n========== XGBOOST TOP 5 FEATURES ==========")

xgb_feature_names = (
    xgb_pipeline.named_steps["preprocessing"]
    .get_feature_names_out()
)

xgb_importance = pd.DataFrame({
    "Feature": xgb_feature_names,
    "Importance": xgb_model.feature_importances_
})

xgb_importance = xgb_importance.sort_values(
    by="Importance",
    ascending=False
)

print(xgb_importance.head(5).to_string(index=False))

# ==========================================
# 11. REGRESSION: PREDICT LOAN AMOUNT
# ==========================================

print("\n========== REGRESSION MODELS ==========")

regression_features = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "Loan_Amount_Term",
    "Credit_History"
]

X_reg = df[regression_features]
y_reg = pd.to_numeric(df["LoanAmount"], errors="coerce")

valid = y_reg.notna()

X_reg = X_reg.loc[valid]
y_reg = y_reg.loc[valid]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)

# Preprocessing for regression
reg_preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

Xr_train_scaled = reg_preprocessor.fit_transform(Xr_train)
Xr_test_scaled = reg_preprocessor.transform(Xr_test)

regression_models = {
    "Linear Regression": LinearRegression(),

    "Ridge": Ridge(alpha=1.0),

    "Lasso": Lasso(alpha=0.1, max_iter=10000)
}

coefficient_results = []

for name, model in regression_models.items():

    model.fit(Xr_train_scaled, yr_train)

    predictions = model.predict(Xr_test_scaled)

    rmse = np.sqrt(np.mean((yr_test - predictions) ** 2))

    coefficient_results.append({
        "Model": name,
        "Coefficients": model.coef_,
        "RMSE": round(rmse, 2)
    })

    print("\n", name)
    print("RMSE:", round(rmse, 2))
    print("Coefficients:", model.coef_)

    if name == "Lasso":
        zero_count = np.sum(model.coef_ == 0)

        print("Number of Zero Coefficients:", zero_count)

# Compare coefficients
print("\nCOEFFICIENT COMPARISON")

coefficient_table = pd.DataFrame({
    "Feature": regression_features,
    "LinearRegression": regression_models[
        "Linear Regression"
    ].coef_,
    "Ridge": regression_models["Ridge"].coef_,
    "Lasso": regression_models["Lasso"].coef_
})

print(coefficient_table.to_string(index=False))

# ==========================================
# 12. K-MEANS CLUSTERING
# ==========================================

print("\n========== K-MEANS CLUSTERING ==========")

cluster_features = [
    "ApplicantIncome",
    "LoanAmount",
    "CoapplicantIncome"
]

cluster_data = df[cluster_features].copy()

cluster_data = cluster_data.replace(
    [np.inf, -np.inf], np.nan
)

cluster_imputer = SimpleImputer(strategy="median")

cluster_data = cluster_imputer.fit_transform(cluster_data)

cluster_scaler = StandardScaler()

cluster_scaled = cluster_scaler.fit_transform(cluster_data)

# ==========================================
# 13. ELBOW CURVE AND SILHOUETTE SCORES
# ==========================================

inertia_values = []
silhouette_values = []

print("\nK VALUE AND SILHOUETTE SCORES")

for k in range(2, 9):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(cluster_scaled)

    inertia_values.append(kmeans.inertia_)

    score = silhouette_score(
        cluster_scaled,
        labels
    )

    silhouette_values.append(score)

    print(
        f"K = {k}, "
        f"Inertia = {kmeans.inertia_:.2f}, "
        f"Silhouette Score = {score:.2f}"
    )

# ==========================================
# 14. PLOT ELBOW CURVE
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 9),
    inertia_values,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("K-Means Elbow Curve")
plt.grid(True)

plt.savefig("elbow_curve.png")
plt.show()

# ==========================================
# 15. SILHOUETTE SCORE PLOT
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 9),
    silhouette_values,
    marker="s"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("K-Means Silhouette Scores")
plt.grid(True)

plt.savefig("silhouette_scores.png")
plt.show()

# ==========================================
# 16. SELECT BEST K AND DESCRIBE SEGMENTS
# ==========================================

best_k = list(range(2, 9))[
    np.argmax(silhouette_values)
]

print("\nBest K based on Silhouette Score:", best_k)

final_kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

cluster_labels = final_kmeans.fit_predict(cluster_scaled)

df["Customer_Segment"] = cluster_labels

print("\nCUSTOMER SEGMENT SUMMARY")

segment_summary = df.groupby("Customer_Segment")[
    cluster_features
].mean().round(2)

print(segment_summary)

print("\nNUMBER OF CUSTOMERS IN EACH SEGMENT")

print(df["Customer_Segment"].value_counts().sort_index())

print("\nTASK 3 COMPLETED SUCCESSFULLY!")
