
import pandas as pd
import numpy as np
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("loan.csv")

df.columns = df.columns.str.strip()
df = df.drop_duplicates()
df = df.dropna(subset=["Loan_Status"])

# Add an application date if dataset does not contain one.
# For a real project, use the actual application date.
if "ApplicationDate" not in df.columns:
    df["ApplicationDate"] = pd.date_range(
        start="2023-01-01",
        periods=len(df),
        freq="D"
    )

print("Dataset Shape:", df.shape)
print(df.head())

# ==========================================
# 2. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]

le = LabelEncoder()
y = le.fit_transform(y.astype(str).str.strip())

print("\nTarget Classes:", le.classes_)

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================

class FeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        # Feature 1: Date decomposition
        date = pd.to_datetime(
            X["ApplicationDate"],
            errors="coerce"
        )

        X["ApplicationYear"] = date.dt.year
        X["ApplicationMonth"] = date.dt.month
        X["ApplicationDay"] = date.dt.day

        # Feature 2: Income / Loan Amount interaction
        X["IncomeToLoanRatio"] = (
            pd.to_numeric(X["ApplicantIncome"], errors="coerce") /
            pd.to_numeric(X["LoanAmount"], errors="coerce").replace(0, np.nan)
        )

        # Feature 3: Log transformation of income
        X["LogApplicantIncome"] = np.log1p(
            pd.to_numeric(
                X["ApplicantIncome"],
                errors="coerce"
            ).clip(lower=0)
        )

        # Remove ID and original date
        X = X.drop(
            columns=["Loan_ID", "ApplicationDate"],
            errors="ignore"
        )

        return X


# ==========================================
# 4. DEFINE COLUMN TYPES
# ==========================================

numeric_features = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "ApplicationYear",
    "ApplicationMonth",
    "ApplicationDay",
    "IncomeToLoanRatio",
    "LogApplicantIncome"
]

categorical_features = [
    "Gender",
    "Married",
    "Dependents",
    "Self_Employed",
    "Property_Area"
]

ordinal_features = ["Education"]

# ==========================================
# 5. CREATE PREPROCESSING TRANSFORMERS
# ==========================================

# Median imputation for numeric columns
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Most frequent imputation + OneHotEncoder
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Ordinal encoding for Education
ordinal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(
        categories=[["Not Graduate", "Graduate"]],
        handle_unknown="use_encoded_value",
        unknown_value=-1
    ))
])

# Combine all preprocessing
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
    ("ordinal", ordinal_pipeline, ordinal_features)
])

# ==========================================
# 6. BUILD COMPLETE ML PIPELINE
# ==========================================

model_pipeline = Pipeline([
    ("feature_engineering", FeatureEngineer()),
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000))
])

# ==========================================
# 7. 5-FOLD CROSS VALIDATION
# ==========================================

print("\n5-FOLD CROSS VALIDATION")

cv_scores = cross_val_score(
    model_pipeline,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("Individual CV Scores:", cv_scores)

print(
    "Pipeline CV accuracy:",
    round(cv_scores.mean(), 2)
)

print(
    "Standard Deviation:",
    round(cv_scores.std(), 2)
)

# ==========================================
# 8. TRY KNN IMPUTER ON NUMERIC COLUMNS
# ==========================================

print("\nKNN IMPUTER EXPERIMENT")

knn_numeric_pipeline = Pipeline([
    ("imputer", KNNImputer(n_neighbors=3)),
    ("scaler", StandardScaler())
])

knn_preprocessor = ColumnTransformer([
    ("numeric", knn_numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
    ("ordinal", ordinal_pipeline, ordinal_features)
])

knn_pipeline = Pipeline([
    ("feature_engineering", FeatureEngineer()),
    ("preprocessing", knn_preprocessor),
    ("classifier", LogisticRegression(max_iter=2000))
])

knn_scores = cross_val_score(
    knn_pipeline,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("KNN CV Scores:", knn_scores)
print("KNN Mean Accuracy:", round(knn_scores.mean(), 2))

# ==========================================
# 9. TRAIN FINAL PIPELINE
# ==========================================

print("\nTRAINING FINAL PIPELINE")

model_pipeline.fit(X, y)

print("Pipeline training completed!")

# ==========================================
# 10. SAVE MODEL USING JOBLIB
# ==========================================

MODEL_FILE = "loan_pipeline.pkl"

joblib.dump(model_pipeline, MODEL_FILE)

print("\nModel saved successfully:", MODEL_FILE)

# ==========================================
# 11. LOAD SAVED MODEL
# ==========================================

loaded_model = joblib.load(MODEL_FILE)

print("Saved model loaded successfully!")

# ==========================================
# 12. PREDICT 3 NEW RAW APPLICANTS
# ==========================================

new_applicants = pd.DataFrame({

    "Loan_ID": ["NEW001", "NEW002", "NEW003"],

    "Gender": ["Male", "Female", "Male"],

    "Married": ["Yes", "No", "Yes"],

    "Dependents": ["1", "0", "2"],

    "Education": [
        "Graduate",
        "Not Graduate",
        "Graduate"
    ],

    "Self_Employed": ["No", "Yes", "No"],

    "ApplicantIncome": [5000, 2500, 7000],

    "CoapplicantIncome": [1500, 0, 2000],

    "LoanAmount": [140, 80, 220],

    "Loan_Amount_Term": [360, 180, 360],

    "Credit_History": [1, 0, 1],

    "Property_Area": [
        "Urban",
        "Rural",
        "Semiurban"
    ],

    "ApplicationDate": [
        "2024-01-15",
        "2024-02-20",
        "2024-03-10"
    ]
})

# Predict using loaded pipeline
predictions = loaded_model.predict(new_applicants)

print("\nPREDICTIONS FOR 3 NEW APPLICANTS")

print("Predicted Encoded Values:", predictions)

print(
    "Predicted Loan Status:",
    le.inverse_transform(predictions)
)

for i, prediction in enumerate(
    le.inverse_transform(predictions),
    start=1
):
    print(f"Applicant {i}: {prediction}")

print("\nALL TASKS COMPLETED SUCCESSFULLY!")
