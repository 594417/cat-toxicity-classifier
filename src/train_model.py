import pandas as pd
from pathlib import Path
from joblib import dump

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

train_df = pd.read_csv("data/processed/train.csv")

x_train = train_df["text"]
y_train = train_df["label"]

model = Pipeline([
    ("tfidf", TfidfVectorizer(
      analyzer="char",
      ngram_range=(2, 4),
    )),
    ("classifier", LogisticRegression(
      max_iter=1000,
      random_state=42
    ))
])

model.fit(x_train, y_train)

Path("models").mkdir(parents=True, exist_ok=True)
dump(model, "models/classifier.joblib")

print("Model trained and saved.")

examples = ["chocolate", "banana", "lily"]

predictions = model.predict(examples)
for item, prediction in zip(examples, predictions):
    print(f"{item}: {prediction}")