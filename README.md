# Gradient Boosting Classification

An applied machine learning project exploring Gradient Boosting for binary classification using the Breast Cancer Wisconsin dataset from scikit-learn.
The project focuses not only on predictive performance, but also on model validation, hyperparameter selection, and interpretability.

## Overview

The project investigates the performance and behaviour of a Gradient Boosting classifier, with a focus on model validation, hyperparameter selection, and interpretability.

The workflow includes:

- Data inspection and class-balance checks
- Stratified train-test splitting
- Majority-class baseline comparison
- Gradient Boosting model training
- Hyperparameter experiments involving learning rate, tree depth, and number of estimators
- 5-fold cross-validation for model selection
- Train-vs-test performance comparison
- Feature importance and permutation importance
- Benchmarking against Logistic Regression
- ROC-AUC comparison

## Results

- Majority-class baseline accuracy: **63.2%**
- Gradient Boosting test accuracy: **95.6%**
- Gradient Boosting ROC-AUC: **0.994**
- Logistic Regression test accuracy: **98.2%**
- Logistic Regression ROC-AUC: **0.995**

Cross-validation favoured shallow trees (`max_depth=1`), suggesting that additional tree complexity did not improve generalisation.

Permutation importance identified `worst concave points` as the most influential feature, while differences between permutation importance and built-in tree importance indicated redundancy among several correlated predictors.

Logistic Regression slightly outperformed Gradient Boosting on the held-out test set, showing that additional model complexity did not necessarily improve out-of-sample performance.

## Tools

Python, pandas, scikit-learn, matplotlib

## Notebook

See [`gradient_boosting.ipynb`](gradient_boosting.ipynb) for the full analysis.
