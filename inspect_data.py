
import pandas as pd

# Load the customer churn dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 1. Display first 5 records
print("\nFIRST 5 CUSTOMERS:")
print(df.head())

# 2. Display dataset size
print("\nDATASET SHAPE:")
print(df.shape)

# 3. Display column information
print("\nDATASET INFORMATION:")
print(df.info())

# 4. Check missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# 5. Check blank TotalCharges entries
print("\nBLANK TOTAL CHARGES:")
print((df["TotalCharges"].astype(str).str.strip() == "").sum())

# 6. Check churn distribution
print("\nCHURN DISTRIBUTION:")
print(df["Churn"].value_counts())

# Display all column names
print("\nCOLUMN NAMES:")
print(df.columns.tolist())

# Check customer churn distribution
print("\nCHURN DISTRIBUTION:")
print(df["Churn"].value_counts())

# Check percentage of churn
print("\nCHURN PERCENTAGE:")
print(df["Churn"].value_counts(normalize=True) * 100)

# Check blank values in TotalCharges
print("\nBLANK TOTAL CHARGES:")
print((df["TotalCharges"].astype(str).str.strip() == "").sum())

# Display basic numerical statistics
print("\nNUMERICAL DATA SUMMARY:")
print(df.describe())


