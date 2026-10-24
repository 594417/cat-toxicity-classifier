import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

train_df = pd.read_csv("data/processed/train.csv")

x_train = train_df["text"]
y_train = train_df["label"]

models = {
    "Logistic Regression": LogisticRegression(
      max_iter=1000, 
      random_state=42
    ),
    "Multinomial Naive Bayes": MultinomialNB()
}

cv = StratifiedKFold(
  n_splits=5, 
  shuffle=True, 
  random_state=42
)

for name, classifier in models.items():
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
          analyzer="char",
          ngram_range=(2, 4),
        )),
        ("classifier", classifier)
    ])
    
    scores = cross_validate(
        pipeline, 
        x_train, 
        y_train, 
        cv=cv, 
        scoring={
            "accuracy": "accuracy",
            "precision": make_scorer(precision_score, pos_label="toxic"),
            "recall": make_scorer(recall_score, pos_label="toxic"),
            "f1": make_scorer(f1_score, pos_label="toxic"),
            "f1_macro": "f1_macro"
        },
    )
    
    print(f"\n{name}")
    print(f"Accuracy: {scores['test_accuracy'].mean():.2%}")
    print(f"Precision: {scores['test_precision'].mean():.2%}")
    print(f"Recall: {scores['test_recall'].mean():.2%}")
    print(f"F1 Score: {scores['test_f1'].mean():.2%}")
    print(f"Macro F1: {scores['test_f1_macro'].mean():.2%}")