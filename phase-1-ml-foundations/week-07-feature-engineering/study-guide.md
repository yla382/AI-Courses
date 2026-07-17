# Week 7 Study Guide — Feature Engineering

Work through each section, then immediately do that section in `tutorial.ipynb` before moving on.

---

## Why Feature Engineering?

Better features beat better models. A mediocre model on great features often outperforms a great model on raw features. This is where domain knowledge turns into ML value.

The rule of thumb in industry: spend 80% of your time on data and features, 20% on model selection.

---

## §1 — Encoding Categorical Variables

Machine learning models work with numbers. Categorical variables (city names, product types, yes/no) need to be converted.

### One-hot encoding

Converts each category into a separate binary column:

```
Color: [red, blue, green]

->

color_red  color_blue  color_green
    1           0           0       (red)
    0           1           0       (blue)
    0           0           1       (green)
```

Use when: categories have no natural order (color, city, product type).
Watch out: if you have 100 categories, you get 100 new columns (high cardinality problem).

### Ordinal encoding

Converts categories to integers that preserve order:

```
Size: [small, medium, large] -> [0, 1, 2]
```

Use when: categories have a meaningful order (small < medium < large).
Don't use for unordered categories — the model will assume 2 > 1 > 0.

### Target encoding

Replaces each category with the mean target value for that category:

```
City:    [NYC, LA, NYC, LA, NYC]
Target:  [1,   0,  1,   1,  0 ]

NYC mean target = (1+1+0)/3 = 0.67  ->  NYC encodes to 0.67
LA  mean target = (0+1)/2   = 0.50  ->  LA  encodes to 0.50
```

Use when: high cardinality (many categories) and you want to capture the relationship with the target.
Risk: data leakage if not done carefully — must be computed on training data only.

---

## §2 — Handling Skewed Distributions

Many real-world features are skewed — most values are small but a few are very large (house prices, income, population).

### Why skew is a problem

Models like linear regression assume features are roughly normally distributed. Extreme outliers pull the model toward them.

```
Raw income: [30k, 35k, 40k, 45k, 2000k]
                                  ^--- pulls everything toward it
```

### Log transform

Compresses large values, spreads small values:

```
log(30000)  = 10.3
log(35000)  = 10.5
log(40000)  = 10.6
log(2000000) = 14.5   (not 2000000 anymore)
```

```python
df['income_log'] = np.log1p(df['income'])  # log1p = log(1+x), handles zeros
```

### Box-Cox transform

More flexible than log — finds the best power transformation automatically. Only works on positive values.

```python
from scipy.stats import boxcox
df['income_bc'], best_lambda = boxcox(df['income'] + 1)
# best_lambda: the power boxcox found that makes the distribution most normal
# e.g. lambda=0 means log transform was best, lambda=0.5 means sqrt was best
```

### When to use

- Right-skewed (long tail on right): use log transform
- Check with histogram — if it looks like an exponential curve, transform it

---

## §3 — Feature Interactions

Sometimes two features together are more informative than either alone.

### Polynomial features

Create new features by multiplying existing ones:

```
features: [x1, x2]
degree=2 polynomial: [x1, x2, x1^2, x1*x2, x2^2]
```

```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
```

**Use case example:** predicting house price from size (sqft).

A linear model assumes: `price = w * sqft + b` — a straight line.

But in reality, the relationship is curved — doubling sqft more than doubles price in expensive markets. A polynomial feature `sqft^2` lets the model fit that curve:

```
raw feature:          price = w1 * sqft + b          (straight line only)
with sqft^2 added:    price = w1 * sqft + w2 * sqft^2 + b  (can fit a curve)
```

Without the polynomial feature, a linear model is stuck drawing a straight line through curved data — high bias. Adding `sqft^2` gives it the flexibility to fit the actual shape.

### Manual interactions

Sometimes domain knowledge tells you which interactions matter:

```python
df['rooms_per_person'] = df['total_rooms'] / df['population']
df['price_per_sqft']   = df['price'] / df['sqft']
```

These hand-crafted features often outperform automated polynomial features because they encode real-world meaning.

### Watch out

Adding many interaction features can cause overfitting. Use regularization (§5) when working with polynomial features.

---

## §4 — Binning and Datetime Features

### Binning

Convert continuous values into categories:

```python
# age -> age group
pd.cut(df['age'], bins=[0, 18, 35, 60, 100],
       labels=['child', 'young_adult', 'adult', 'senior'])
```

Use when: the relationship between a feature and target is non-linear and step-like (e.g. tax brackets, age groups).

### Datetime features

Raw timestamps are useless to models. Extract meaningful components:

```python
df['hour']        = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek   # 0=Monday
df['month']       = df['timestamp'].dt.month
df['is_weekend']  = df['day_of_week'].isin([5, 6]).astype(int)
```

A model can learn that sales peak on weekends, or fraud spikes at 3am — but only if you extract those features.

---

## §5 — Regularization

When you have many features (especially after adding interactions), models overfit. Regularization adds a penalty to the loss function that pushes weights toward zero.

### Lambda and how sklearn names it

Lambda is the regularization strength — how hard you penalize large weights. sklearn names it differently per model:

```
Ridge / Lasso:       alpha = lambda
                     alpha = 0.01  -> weak regularization
                     alpha = 100   -> strong regularization

LogisticRegression:  C = 1 / lambda   (inverse!)
                     C = 100  -> weak regularization  (lambda = 0.01)
                     C = 0.01 -> strong regularization (lambda = 100)
```

LogisticRegression uses `C` for historical reasons. Just remember: **smaller C = more regularization**, which is the opposite of alpha.

### L2 regularization (Ridge)

Adds the sum of squared weights to the loss:

```
Loss = original_loss + lambda * sum(w^2)
```

Effect: shrinks all weights toward zero but never exactly to zero. All features stay in the model, just with smaller weights.

### L1 regularization (Lasso)

Adds the sum of absolute weights to the loss:

```
Loss = original_loss + lambda * sum(|w|)
```

Effect: drives some weights to exactly zero — built-in feature selection. Sparse model.

### Comparison

```
L2 (Ridge):  all features kept, weights shrunk
             -> use when you think all features matter a little
             
L1 (Lasso):  some features dropped (weight = 0)
             -> use when you suspect many features are irrelevant

ElasticNet:  combination of both
             -> safe default when unsure
```

### Lambda (regularization strength)

```
lambda = 0    -> no regularization (standard model)
lambda small  -> light penalty, model still fits closely
lambda large  -> heavy penalty, weights shrink toward zero (underfits)
```


---

## §6 — Feature Selection

After engineering features, you often have too many. Feature selection removes the ones that don't help.

### Filter methods

Score each feature independently, keep top N:

```python
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)
```

### Wrapper methods (RFE)

Recursively remove the least important features:

```python
from sklearn.feature_selection import RFE
rfe = RFE(RandomForestClassifier(), n_features_to_select=10)
X_selected = rfe.fit_transform(X, y)
```

### Importance-based

Use Random Forest feature importances to drop low-importance features — you already know how to do this from week 6.

### When to use

- Too many features slowing down training -> SelectKBest or importances
- Want to understand which features matter -> RFE or importances
- Suspect many features are noise -> Lasso (L1) handles this automatically

---

## Checklist

- [ ] Can apply one-hot, ordinal, and target encoding and know when to use each
- [ ] Know when and how to apply log transform
- [ ] Can create polynomial and manual interaction features
- [ ] Understand the difference between L1 and L2 regularization
- [ ] Know what lambda controls and what happens at extremes
- [ ] Can run at least one feature selection method
