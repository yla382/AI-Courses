# Week 7 — Feature Engineering + Stats III

**Topic:** Feature engineering is often more impactful than model choice. This is where domain knowledge becomes ML value.
**Math Unlocked:** Regularization math (L1/L2), information theory basics (entropy, information gain).

## What You'll Build
- Full feature engineering pipeline: encoding, binning, interactions, transformations
- Regularization comparison: L1 vs L2 on a high-dimensional dataset
- Feature selection: which features actually matter?

## Techniques Covered
- Encoding categorical variables: one-hot, ordinal, target encoding
- Handling skewed distributions: log transform, Box-Cox
- Feature interactions: polynomial features
- Binning continuous variables
- Datetime features: extract hour, day-of-week, month, etc.

## Math: Regularization
- L2 (Ridge): adds λ×Σw² to loss. Shrinks all weights toward zero. Prefers small weights.
- L1 (Lasso): adds λ×Σ|w| to loss. Drives some weights to exactly zero. Built-in feature selection.
- ElasticNet: combination of both
- λ (regularization strength): hyperparameter you tune

## Math: Information Theory
- Entropy: measure of uncertainty/disorder in a variable
- Information gain: how much a feature reduces entropy — used in decision trees
