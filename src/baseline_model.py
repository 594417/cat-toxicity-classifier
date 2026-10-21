import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
  accuracy_score, 
  classification_report, 
  confusion_matrix
)

train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

x_train = train_df["text"]
y_train = train_df["label"]

x_test = test_df["text"]
y_test = test_df["label"]

baseline = DummyClassifier(strategy="most_frequent")

baseline.fit(x_train, y_train)

y_pred = baseline.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Baseline Accuracy:{accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(
  y_test, 
  y_pred,
  labels=["toxic", "non-toxic"]
))