import pandas as pd

# Load dataset
df = pd.read_csv("data/dataset.csv")

# Separate features and target
X = df.drop("DEFECT_LABEL", axis=1)
y = df["DEFECT_LABEL"]

print("Features:")
print(X.head())

print("\nTarget:")
print(y.head())

print("\nFeature shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining feature shape:")
print(X_train.shape)

print("\nTesting feature shape:")
print(X_test.shape)

print("\nTraining target shape:")
print(y_train.shape)

print("\nTesting target shape:")
print(y_test.shape)