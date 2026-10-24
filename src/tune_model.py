import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, recall_score
from pathlib import Path
from joblib import dump

train_df = pd.read_csv("data/processed/train.csv")

x_train = train_df["text"]
y_train = train_df["label"]

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(
      max_iter=1000, 
      random_state=42
    ))
])

param_grid = {
    "tfidf__analyzer": ["char", "char_wb"],
    "tfidf__ngram_range": [(2, 4), (2, 5), (3, 5)],
    "classifier__C": [0.1, 1, 10],
    "classifier__class_weight": [None, "balanced"]
}

cv = StratifiedKFold(
  n_splits=5, 
  shuffle=True, 
  random_state=42
)

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring={
      "f1": "f1_macro",
      "recall_toxic": make_scorer(
        recall_score,
        pos_label="toxic"
      )
    },
    refit="f1",
    n_jobs=-1
)

grid.fit(x_train,y_train)

print("Best parameters:")
print(grid.best_params_)

print ("\nBest cross-validation macro f1:")
print(f"{grid.best_score_:.20%}")

best_index = grid.best_index_

print("\nToxic recall for selected configuration:")
print(
  f"{grid.cv_results_['mean_test_recall_toxic'][best_index]:.2%}"
)

Path("models").mkdir(parents=True, exist_ok=True)

dump(grid.best_estimator_, "models/tuned_classifier.joblib")

print("Tuned model saved")