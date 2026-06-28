# "Required Assignment 17.1: Comparing Classifiers"

Anna Niu

The Jupyter Notebook for this assignment can be found [here](https://github.com/annaniu02/MLAI-Assignment-17.1/blob/main/prompt_III.ipynb).

The dataset used for this assignment can be found [here](https://github.com/annaniu02/MLAI-Assignment-17.1/blob/main/data/bank-additional-full.csv).

---

This project compares four classification models (K-Nearest Neighbors, Logistic Regression, Decision Tree, and Support Vector Machine) on the Portuguese bank telemarketing dataset. The goal is to predict whether a contacted client will subscribe to a term deposit.

---

## Business Problem

The business objective of this task is to build and compare classification models that predict whether a client will subscribe to a term deposit, i.e. `y = "yes"`. The practical goal is to not just maximize accuracy, but help the bank rank clients by likelihood of subscribing so the their call center can focus on the strongest prospects. This task is a binary classification problem with an imbalanced target, as most clients do not subscribe.

---

## Dataset

- For this project, the dataset used is the UCI Bank Marketing dataset, which can be found in the `data` folder as the file `bank-additional-full.csv`
- This dataset had 41,188 original rows, with 41,176 rows remaining after removing exact duplicates
- The data was collected from 17 telemarketing campaigns that occurred between May 2008 and November 2010
- Our target variable was `y`, which indicated whether the client subscribed to a term deposit (`yes` / `no`)
- The dataset was imbalanced, where about 11.3% subscribed and 88.7% did not subscribe
- The dataset no standard `NaN` values to indicate missing information; several categorical variables use `"unknown"` as a missing-value label
- The `duration` feature was excluded from the realistic models because call duration is only known after the call occurs, so it would leak information unavailable at prediction time

---

## CRISP-DSM Approach

In our project notebook, we followed the below approach:

1. Established business understanding of the task at hand
2. Established dataset understanding and cleaned data by reviewing columns, data types, duplicates, unknown values, and the target class balance
3. Performed exploratory data analysis to visualize target distribution, categorical subscription rates, numeric distributions, and feature correlations
4. Prepared the features and target column for modeling with appropriate encoding and transformation by using `ColumnTransformer` with:
   - `StandardScaler` for numeric features
   - `OneHotEncoder` for categorical features
5. Established and evaluated a baseline model majority-class `DummyClassifier` to compare real model performance against
6. Compared this baseline model performance to Logistic Regression, KNN, Decision Tree, and SVM models
7. Improved Logistic Regression, KNN, Decision Tree, and SVM models performance by using the full feature set (excluding `duration`), tuning hyperparameters with `GridSearchCV`, and evaluating performance with ROC-AUC instead of accuracy

---

## Evaluation Metric

The primary evaluation metric used for our improved models is ROC-AUC.

We found early in the project that accuracy is not enough for this problem because a model that predicts `"no"` for every client is already about 89% accurate, but it finds no subscribers. ROC-AUC is better aligned with the business goal because it measures how well the model ranks likely subscribers above unlikely subscribers.

The notebook also reports accuracy, precision, recall, and F1 score for the positive class.

---

## Results

### Baseline and Simple Models

**Baseline accuracy:** 0.8873

#### Simple Models Using Bank Client Only Features
| Model | Train Time (s) | Train Accuracy | Test Accuracy |
| --- | ---: | ---: | ---: | 
| Logistic Regression | 0.757171 | 0.887341 | 0.887324 |
| KNN | 0.147772 | 0.889557 | 0.876032 |
| Decision Tree | 2.223616 | 0.917608 | 0.863526 |
| SVM | 164.804684 | 0.888100 | 0.887810 |

From these results, we can see that all four models perform similarly to the baseline model. Ultimately, Decision Tree performs best in training accuracy but worst in test accuracy, while SVC performs best in test accuracy but is the slowest model by far. All models performed better than the baseline in training, however Decision Tree and KNN performed worse than the baseline in test accuracy while the other two performed slightly better than baseline.

### Tuned Models Using the Full Realistic Feature Set

| Model | Test ROC-AUC | Test Accuracy | Test Precision (yes) | Test Recall (yes) | Test F1 (yes) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8000 | 0.8295 | 0.3578 | 0.6455 | 0.4604 |
| KNN | 0.7623 | 0.8959 | 0.5937 | 0.2425 | 0.3443 |
| Decision Tree | 0.7966 | 0.8464 | 0.3877 | 0.6272 | 0.4792 |
| SVM | 0.7831 | 0.8448 | 0.3858 | 0.6369 | 0.4805 |

We can see from these final results that the tuned models using the full feature set perform much better than the bank client only models when evaluated by ROC-AUC. 

When evaluated using test ROC-AUC, we see that Logistic Regression performed the best, followed by Decision Tree, SVM, and finally KNN.

The strongest models are close in performance, so interpretability and business usability matter in the final choice of the model. Logistic Regression models are especially useful in this case due to their coefficients being easier to explain to nontechnical stakeholders.

---

## Key Findings

- Subscriptions are rare: only about 11% of clients subscribe, so the bank should not evaluate models by accuracy alone
- Client demographics alone are not enough: models using only bank-client features perform close to the baseline
- Campaign and economic context improve prediction: adding contact information, previous campaign outcomes, and economic indicators increases ROC-AUC substantially
- Past campaign success is one of the strongest signals: clients who subscribed in a previous campaign are much more likely to subscribe again
- Channel and timing matter: cellular contacts and certain months show higher subscription rates
- Logistic Regression is a strong practical choice: the best model may vary slightly by sample, but Logistic Regression is a strong practical option because it balances performance and interpretability

---

## Recommendations

- Prioritize clients with previous campaign success because they are the most responsive group
- Use model scores to rank the call list instead of calling clients in an untargeted order
- Favor cellular outreach when available because it has higher observed conversion
- Track ROC-AUC, recall, precision, and F1 instead of relying only on accuracy
- Use Logistic Regression as a starting deployment model because it balances performance and interpretability

---

## Next Steps

- Tune the decision threshold: the default threshold may not be best for the business; the bank should choose a threshold based on call cost and expected value of a subscription
- Run a lift/gains analysis: marketing managers would benefit from knowing how many subscribers can be captured by calling the top 10%, 20%, or 30% of ranked clients
- Validate over time: since economic conditions change, train on earlier campaigns and test on later campaigns before deployment
- Consider additional models: random Forest or Gradient Boosting may improve performance, but the current assignment focuses on KNN, Logistic Regression, Decision Trees, and SVM
- Collect more pre-call client features: more client-level data could improve targeting without relying on information only known after contact

---

## Repository Structure

```text
.
├── data/
│   └── bank-additional-full.csv   # Full dataset
│   └── bank-additional-names.txt  # Data dictionary
|   └── bank-additional.csv        # Smaller dataset sample
├── CRISP-DM-BANK.pdf              # Reference paper
├── README.md                      # Project summary and findings
└── prompt_III.ipynb               # Completed and executed notebook
```
