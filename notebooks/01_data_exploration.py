import pandas as pd

# Load the dataset
df = pd.read_csv("data/dataset.csv")
# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Display number of rows and columns
print("\nDataset shape:")
print(df.shape)

# Display column names
print("\nColumn names:")
print(df.columns.tolist())

# Display dataset information
print("\nDataset information:")
df.info()

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())

# Check target distribution
print("\nDefect label distribution:")
print(df["DEFECT_LABEL"].value_counts())

print("\nDefect label percentage:")
print(df["DEFECT_LABEL"].value_counts(normalize=True) * 100)