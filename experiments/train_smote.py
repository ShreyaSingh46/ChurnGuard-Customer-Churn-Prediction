
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler, OrdinalEncoder, OneHotEncoder
)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

# 1. Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 2. Basic cleaning
df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# 3. Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 4. Identify numerical and categorical columns
num_cols = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

cat_cols = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

# 5. Numerical preprocessing
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 6. Categorical preprocessing
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    ))
])

# 7. Preprocessing before SMOTENC
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# 8. Identify categorical feature positions
num_count = len(num_cols)
cat_indices = list(
    range(num_count, num_count + len(cat_cols))
)

# 9. Convert categorical features back to one-hot format
postprocessor = ColumnTransformer([
    ("num", "passthrough", list(range(num_count))),
    ("cat", OneHotEncoder(handle_unknown="ignore"),
     cat_indices)
])

# 10. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("BEFORE SMOTENC:")
print(y_train.value_counts())

# 11. Create complete ML pipeline
model = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTENC(
        categorical_features=cat_indices,
        random_state=42
    )),
    ("postprocessor", postprocessor),
    ("classifier", XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        eval_metric="logloss"
    ))
])

# 12. Train model
model.fit(X_train, y_train)

# Display balanced training counts
print("\nAFTER SMOTENC:")
print(pd.Series(
    model.named_steps["smote"].sampling_strategy_
))

# 13. Make predictions on original test data
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 14. Evaluate model
print("\nXGBOOST WITH SMOTENC RESULTS")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))
