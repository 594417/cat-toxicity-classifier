import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/raw/cat_toxicity.csv")

df.columns = df.columns.str.strip()

df["text"] = df["text"].str.lower().str.strip()
df["label"] = df["label"].str.lower().str.strip()

df = df.drop_duplicates(subset=["text"])

x = df["text"]
y = df["label"]

x_train, x_test, y_train, y_test = train_test_split(
  x,
  y,
  test_size=0.2,
  random_state=42,
  stratify=y
)

Path("data/processed").mkdir(parents=True, exist_ok=True)

train_df = pd.DataFrame({
  "text": x_train, 
  "label": y_train
})

test_df = pd.DataFrame({
  "text": x_test,
  "label": y_test
})

train_df.to_csv("data/processed/train.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

print("Training entries:", len(train_df))
print("Testing entries:", len(test_df))