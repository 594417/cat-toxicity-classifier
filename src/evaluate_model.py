import pandas as pd
from joblib import load
from sklearn.metrics import (
  accuracy_score, 
  classification_report, 
  confusion_matrix
)

test_df = pd.read_csv("data/processed/test.csv")

x_test = test_df["text"]
y_test = test_df["label"]

model = load("models/classifier.joblib")

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(
  y_test, 
  y_pred,
  labels=["toxic", "non-toxic"]
))

results = pd.DataFrame({
  "text": x_test,
  "actual": y_test,
  "predicted": y_pred
})

mistakes = results[results["actual"] != results["predicted"]]

print("\nMistakes:")
print(mistakes.to_string(index=False))