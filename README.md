# Gradient Boosting for Credit Default Prediction

This project applies Gradient Boosting to predict credit-card default using the UCI Default of Credit Card Clients dataset.

The analysis focuses on model evaluation under class imbalance, hyperparameter selection, model interpretation, and comparison with a simpler Logistic Regression benchmark.

## Overview

The dataset contains 30,000 credit-card clients and 23 explanatory variables covering:

- Credit limits
- Demographic characteristics
- Repayment history
- Monthly bill balances
- Previous payment amounts

The target variable indicates whether a client defaulted in the following month.

Because approximately 22% of observations belong to the default class, accuracy alone can be misleading. Model performance is therefore evaluated using ROC-AUC, PR-AUC, precision, recall, and F1 score.

## Workflow

The analysis includes:

- Data inspection and feature cleaning
- Stratified train-test splitting
- Majority-class baseline comparison
- Gradient Boosting model training
- Five-fold cross-validation for hyperparameter selection
- Evaluation under class imbalance
- Permutation feature importance
- Logistic Regression benchmarking
- ROC and Precision-Recall curve comparison

## Baseline

A majority-class classifier predicts every client as non-defaulting.

Despite having no discriminatory ability, it achieves approximately:

- Accuracy: **77.9%**
- ROC-AUC: **0.500**
- PR-AUC: **0.221**

This illustrates why accuracy alone is insufficient for evaluating an imbalanced classification problem.

## Results

| Model | Accuracy | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Dummy Classifier | 0.779 | 0.500 | 0.221 | 0.000 | 0.000 | 0.000 |
| Logistic Regression | 0.809 | 0.710 | 0.495 | 0.692 | 0.244 | 0.361 |
| Gradient Boosting | **0.818** | **0.779** | **0.553** | 0.666 | **0.360** | **0.467** |

Gradient Boosting provided the strongest overall discrimination of credit-default risk.

Compared with Logistic Regression, the main improvement came from substantially higher recall and F1 score, suggesting that nonlinear relationships and feature interactions contain useful predictive information beyond the linear benchmark.

## Hyperparameter Selection

Five-fold cross-validation was used to compare combinations of:

- Number of estimators
- Learning rate
- Maximum tree depth

The best cross-validation specification used:

- `n_estimators = 200`
- `learning_rate = 0.1`
- `max_depth = 2`

However, this specification did not improve held-out performance relative to the default Gradient Boosting model.

The default model was therefore retained for the final comparison.

## Model Interpretation

Permutation importance was evaluated using the reduction in test-set ROC-AUC after randomly shuffling each feature.

The most important predictor was the most recent repayment status (`PAY_0`), followed by variables including:

- Credit limit
- Recent bill balance
- Previous payment amounts
- Earlier repayment-status variables

The results indicate predictive importance rather than causal effects.

## Logistic Regression Benchmark

Logistic Regression provides a simpler linear benchmark.

For this model:

- Continuous and ordinal variables are standardized
- Nominal categorical variables are one-hot encoded

Gradient Boosting materially outperformed the Logistic Regression benchmark in both ROC-AUC and PR-AUC.

## Conclusion

Gradient Boosting achieved a test ROC-AUC of approximately **0.779** and a PR-AUC of **0.553**, outperforming both the majority-class baseline and Logistic Regression.

The project highlights three modelling considerations:

1. Accuracy can be misleading under class imbalance.
2. Nonlinear modelling can improve discrimination relative to a linear benchmark.
3. More extensive hyperparameter tuning does not necessarily improve out-of-sample performance.

Overall, the analysis demonstrates the importance of appropriate benchmarking, validation, and interpretation when applying machine-learning methods to credit-risk classification.

## Tools

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib

## Data

The analysis uses the UCI **Default of Credit Card Clients** dataset.

The dataset is loaded using the `ucimlrepo` Python package.

## Notebook

See [`gradient-boosting-credit-default.ipynb`](gradient-boosting-credit-default.ipynb) for the full analysis.
