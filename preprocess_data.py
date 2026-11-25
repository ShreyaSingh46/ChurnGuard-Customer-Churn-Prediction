
import pandas as pd

# Load the original dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 1. Remove customerID
df.drop("customerID", axis=1, inplace=True)

# 2. Convert TotalCharges into numeric values
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# 3. Convert Churn into numerical values
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

# 4. Display cleaned dataset
print("CLEANED DATASET:")
print(df.head())

# 5. Check data types
print("\nDATA TYPES:")
print(df.dtypes)

# 6. Check missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# 7. Check dataset shape
print("\nDATASET SHAPE:")
print(df.shape)
