import pandas as pd

df = pd.read_csv("data/raw/cat_toxicity.csv")

df.columns = df.columns.str.strip()

df["text"] = df["text"].str.strip().str.lower()
df["label"] = df["label"].str.strip().str.lower()

print("Number of entries:", len(df))

print("\nLabels:")
print(df["label"].value_counts())

print("\nCategories:")
print(df["category"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate entries:")
print(df[df.duplicated(subset=["text"], keep=False)])