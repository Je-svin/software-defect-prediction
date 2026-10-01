import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/dataset.csv")

# Check target distribution
print("Defect distribution:")
print(df["DEFECT_LABEL"].value_counts())

# Plot target distribution
df["DEFECT_LABEL"].value_counts().plot(kind="bar")

plt.title("Defect vs Non-Defect")
plt.xlabel("Defect Label")
plt.ylabel("Number of Modules")
plt.xticks(rotation=0)

plt.show()

# Scatter plot: LOC vs Defect Label
plt.figure()

plt.scatter(df["LOC"], df["DEFECT_LABEL"])

plt.title("LOC vs Defect Label")
plt.xlabel("LOC")
plt.ylabel("Defect Label")

plt.show()

# Scatter plot: CYCLO vs Defect Label
plt.figure()

plt.scatter(df["CYCLO"], df["DEFECT_LABEL"])

plt.title("CYCLO vs Defect Label")
plt.xlabel("Cyclomatic Complexity")
plt.ylabel("Defect Label")

plt.show()

# Scatter plot: VOLUME vs Defect Label
plt.figure()

plt.scatter(df["VOLUME"], df["DEFECT_LABEL"])

plt.title("VOLUME vs Defect Label")
plt.xlabel("Software Volume")
plt.ylabel("Defect Label")

plt.show()

# Compare average metrics for defective and non-defective modules

features = [
    "LOC",
    "CYCLO",
    "LENGTH",
    "VOLUME",
    "DIFFICULTY",
    "INT_FAN_IN",
    "INT_FAN_OUT",
    "NUM_OPERATORS",
    "NUM_OPERANDS",
    "BRANCH_COUNT"
]

average_metrics = df.groupby("DEFECT_LABEL")[features].mean()

print("\nAverage metrics by defect label:")
print(average_metrics)

# Correlation matrix

print("\nCorrelation matrix:")
print(df.corr(numeric_only=True))

import seaborn as sns

# Correlation heatmap

plt.figure(figsize=(12, 8))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.show()