# Cat Toxicity Classifier

This is a machine-learning-based software in form of a website, created in DAT158 - Machine Learning Engineering and Advanced Agorithms. It predicts whether a food, plant or household item may be toxic for cats based on its name.

The project demonstrates the machine-learning lifecycle:

- Data collection
- Preprocessing
- Model training
- Evaluation
- Hyperparameter tuning
- Deployment

## Demo of Website

## Dataset

The dataset is manually collected from sources providing information about substances that may or may not be toxic to cats.

It contains 149 entries, including food, plants, household items, medications, drinks and animals.

Each entry contains:

- text - Name of the item
- label - Toxic or non-toxic
- category - Type of item
- source - Source used for finding items

The dataset was split into:

- 80% training data: 119 entries
- 20% testing data: 30 entries

## Machine Learning Models

Two classification algorithms were compared:

1. Logisitc Regression
2. Multinomial Naive Bayes

Text was converted into numerical features using TF-IDF with character n-grams.

Five-fold stratified cross-validation was used to evaluate the models.

| Model                   | Accuracy | Macro F1-score |
| ----------------------- | -------- | -------------- |
| Logistic Regression     | 64,71%   | 60,13%         |
| Multinomial Naive Bayes | 65,54%   | 63,08%         |

## Hyperparameter Tuning

GridSearchCV was used to explore 36 hyperparameter combinations for Logistic Regression

The selected parameters were:

- C = 10
- class_weight = None
- analyzer = "char_wb"
- ngram_range = (2, 4)

The selected configuration achieved a cross-validation macro F1-score of 72,33% during hyperparameter search.

This score was used for model selection and may be optimistic.

## Installation

**1. Clone the repository**

```
git clone https://github.com/594417/cat-toxicity-classifier.git
cd cat-toxicity-classifier
```

**2. Create a virtual environment**

```
python -m venv .venv
```

**3. Activate it:**

```
source .venv/bin/activate
```

**4. Install dependencies**

```
python -m pip install -r requirements.txt
```

## Running the Application

Start the Streamlit website:

```
python -m streamlit run app.py
```

The application should open in your browser.

## Reproducing the Model

To prepare the dataset:

```
python src/prepare_data.py
```

To train the original Logistic Regression model:

```
python src/train_model.py
```

To evaluate the original model:

```
python src/evaluate_model.py
```

To compare classification models:

```
python src/compare_models.py
```

To perform hyperparameter tuning and save the tuned model:

```
python src/tune_model.py
```

## Limitations

The model may produce incorrect predictions, especially for unfamiliar items. A non-toxic prediction does not guarantee that an item is safe for cats.
