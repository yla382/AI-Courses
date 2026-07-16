# Week 6 — Ensemble Methods + Optimization Math

**Topic:** Random Forests and Gradient Boosting — the workhorses of production ML on tabular data.
**Math Unlocked:** Optimization landscape, convexity, second-order methods (intuition).

## What You'll Build
- Random Forest with feature importance analysis
- XGBoost model — the go-to for Kaggle tabular competitions
- Hyperparameter tuning with grid search and random search

## Algorithms Covered
- Bagging: Random Forest — parallel ensemble, reduces variance
- Boosting: AdaBoost, Gradient Boosting, XGBoost, LightGBM — sequential ensemble, reduces bias
- Stacking: meta-learner on top of base models

## Math: Why Ensembles Work
- Bias-variance decomposition
- Boosting as gradient descent in function space
- Why averaging uncorrelated weak learners reduces variance

## Practical Notes
- XGBoost/LightGBM win most tabular Kaggle competitions
- Feature importance: which features does the model rely on most?
- SHAP values: explain individual predictions (important for production)
