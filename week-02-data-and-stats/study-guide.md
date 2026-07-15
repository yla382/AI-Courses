# Study Guide — Week 2: Data Exploration + Stats I

---

## §1 — Pandas: Your Primary Tool for Working with Data

### What is Pandas?

Pandas is a Python library for working with structured (tabular) data. Think of it as a programmable spreadsheet. It has two core data structures you need to know:

- **Series** — a 1D labeled array. Like a single column in a spreadsheet.
- **DataFrame** — a 2D table of rows and columns. Like the entire spreadsheet.

Every dataset you work with in ML will pass through a DataFrame at some point.

### Creating a DataFrame

```python
import pandas as pd

# From a dictionary — keys become column names
data = {
    'age':    [25, 30, 35, 40],
    'salary': [50000, 70000, 90000, 110000],
    'city':   ['Seoul', 'NYC', 'Tokyo', 'London']
}
df = pd.DataFrame(data)
print(df)
#    age  salary    city
# 0   25   50000   Seoul
# 1   30   70000     NYC
# 2   35   90000   Tokyo
# 3   40  110000  London
```

The leftmost column (0, 1, 2, 3) is the **index** — Pandas' way of labeling each row. By default it's integers starting at 0.

### Inspecting a DataFrame — Do This First, Every Time

Before doing anything with a dataset, run these:

```python
df.shape        # (rows, columns) — how big is this dataset?
df.dtypes       # what type is each column? (int, float, object, bool)
df.head(5)      # first 5 rows — what does the data look like?
df.tail(5)      # last 5 rows
df.info()       # column names, types, non-null counts — spot missing data fast
df.describe()   # summary stats for numeric columns (mean, std, min, max, percentiles)
```

`df.info()` and `df.describe()` together give you the fastest overview of a dataset. Always run them before anything else.

### Selecting Data

```python
# Select a single column → returns a Series
df['age']
df.age          # same thing, dot notation

# Select multiple columns → returns a DataFrame
df[['age', 'salary']]

# Select rows by index position (like a list)
df.iloc[0]      # first row
df.iloc[0:3]    # first 3 rows
df.iloc[0, 1]   # row 0, column 1 (salary)

# Select rows by label/condition
df.loc[0]                        # row with index label 0
df.loc[df['age'] > 30]           # rows where age > 30
df.loc[df['age'] > 30, 'salary'] # salary for people over 30
```

**Rule of thumb:** use `iloc` when you know the position, use `loc` when you're filtering by condition or label.

### Filtering with Boolean Masks

A boolean mask is an array of True/False values used to filter rows:

```python
mask = df['age'] > 30          # Series of True/False
df[mask]                       # keep only True rows

# Combine conditions
df[(df['age'] > 30) & (df['salary'] > 80000)]   # AND
df[(df['age'] < 25) | (df['city'] == 'NYC')]     # OR
```

You saw this in week 1 with the Iris visualization. Now you know the full mechanics.

### Adding and Modifying Columns

```python
# Add a new column
df['salary_k'] = df['salary'] / 1000   # salary in thousands

# Modify existing column
df['age'] = df['age'] + 1

# Drop a column
df = df.drop(columns=['city'])

# Rename columns
df = df.rename(columns={'salary': 'annual_salary'})
```

### Missing Values

Real-world datasets almost always have missing values. Pandas represents them as `NaN` (Not a Number).

```python
df.isnull()            # True/False for every cell
df.isnull().sum()      # count of missing values per column
df.isnull().sum() / len(df) * 100   # % missing per column

# Options for handling missing values:
df.dropna()                         # drop rows with any NaN
df.fillna(0)                        # fill NaN with 0
df['age'].fillna(df['age'].mean())  # fill with column mean (common for numeric)
df['city'].fillna('Unknown')        # fill with placeholder (common for categorical)
```

Deciding whether to drop or fill depends on how many values are missing and why. If <5% are missing and it seems random, filling with mean/median is usually fine. If >30% of a column is missing, the column may not be useful at all.

### Deep Dive Pointers
- [Pandas official docs — 10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
- [Pandas cheat sheet (PDF)](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf) — keep this open while coding

---

## §2 — Statistics I: Describing Data with Numbers

Before training any model, you need to understand what your data actually looks like numerically. These are the tools for that.

### Mean, Median, Mode

```python
import numpy as np

ages = np.array([22, 25, 25, 27, 30, 35, 80])

mean   = ages.mean()      # 34.9 — sum divided by count
median = np.median(ages)  # 27   — middle value when sorted
```

**Mean** is sensitive to outliers. One value of 80 pulled the mean from ~27 to ~35.
**Median** is resistant to outliers — it's the actual middle value regardless of extremes.

**When to use which:**
- Salaries, house prices, any skewed data → use median
- Symmetric data, no outliers → mean and median are similar, either works
- In ML: both are used for filling missing values. Pick based on the distribution.

**Mode** = the most frequent value. Mainly useful for categorical data (e.g., most common city).

### Variance and Standard Deviation

These measure how **spread out** your data is.

```python
variance = ages.var()   # average squared distance from the mean
std      = ages.std()   # square root of variance — in the same units as the data
```

**Intuition:**
- Small std → values are clustered tightly around the mean
- Large std → values are spread out widely

```
Ages: [27, 28, 27, 29, 28]  → std ≈ 0.7  (very tight)
Ages: [5, 15, 27, 50, 80]   → std ≈ 27   (very spread)
```

**Why this matters in ML:**
- Features with very different scales (age: 20–80, salary: 20000–200000) cause problems for many models
- `StandardScaler` normalizes each feature to mean=0, std=1 — you used this in week 1
- Now you understand what it's actually doing: shifting by mean, scaling by std

Formula:
```
scaled_value = (value - mean) / std
```

### Percentiles and Quartiles

A percentile tells you what % of data falls below a given value.

```python
np.percentile(ages, 25)   # 25th percentile (Q1) — 25% of data is below this
np.percentile(ages, 50)   # 50th percentile = median
np.percentile(ages, 75)   # 75th percentile (Q3) — 75% of data is below this

# IQR (interquartile range) = Q3 - Q1
# Covers the middle 50% of data
iqr = np.percentile(ages, 75) - np.percentile(ages, 25)
```

**IQR is used for outlier detection:**
- Values below `Q1 - 1.5×IQR` or above `Q3 + 1.5×IQR` are considered outliers
- This is what matplotlib's boxplot shows visually

### Pandas describe() — All of the Above in One Line

```python
df.describe()
```

This gives you count, mean, std, min, 25th percentile, median, 75th percentile, and max for every numeric column. Run this first on any new dataset.

### Deep Dive Pointers
- [StatQuest — Statistics Fundamentals playlist](https://www.youtube.com/playlist?list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9) (watch: mean/variance/std videos)
- [Khan Academy — Descriptive Statistics](https://www.khanacademy.org/math/statistics-probability/summarizing-quantitative-data) — good for building intuition

---

## §3 — Distributions: What Shape Is Your Data?

A distribution describes how values are spread across a range. The shape of a distribution tells you a lot about the data and what to do with it.

### Normal Distribution (Bell Curve)

```
     ████
   ████████
 ████████████
███████████████
```

- Symmetric around the mean
- Mean = Median = Mode
- ~68% of data falls within 1 std of the mean
- ~95% within 2 stds
- ~99.7% within 3 stds

**In ML:** Many algorithms assume features are normally distributed. When they're not, you may need to transform them.

### Skewed Distributions

**Right-skewed (positive skew):** Long tail to the right. Mean > Median.
```
█
██
████
████████
████████████████
```
Examples: income, house prices, number of purchases. A few very large values pull the mean right.

**Left-skewed (negative skew):** Long tail to the left. Mean < Median.
Examples: age at retirement, test scores when the test is easy.

**Why it matters in ML:**
- Skewed features can hurt linear models and distance-based models (kNN, SVM)
- Fix: apply a log transform → `np.log1p(x)` compresses the long tail
- Tree-based models (Random Forest, XGBoost) are less sensitive to skew

### How to Check Distribution Shape

```python
import matplotlib.pyplot as plt

# Histogram — shows shape
plt.hist(df['salary'], bins=30)

# Skewness — numerical measure
df['salary'].skew()
# > 0: right-skewed, < 0: left-skewed, ~0: symmetric

# Kurtosis — how heavy the tails are
df['salary'].kurtosis()
```

### Outliers

An outlier is a value that's far from the rest of the data.

**How to spot them:**
- Visually: boxplot (`df.boxplot()`) — dots outside the whiskers are outliers
- Numerically: IQR method (from §2), or Z-score (`|value - mean| / std > 3`)

**What to do with them:**
- Investigate first — is it a data error, or a real extreme value?
- Cap them (winsorizing): replace values above 99th percentile with the 99th percentile value
- Remove them: only if you're confident they're errors
- Leave them: tree-based models handle outliers well

### Deep Dive Pointers
- [StatQuest — Normal Distribution](https://www.youtube.com/watch?v=rzFX5NWojp0)
- [StatQuest — Skewness and Kurtosis](https://www.youtube.com/watch?v=XSSRrVMOqlQ)

---

## §4 — Correlation: How Features Relate to Each Other

Correlation measures the **linear relationship** between two variables. It ranges from -1 to +1.

```
+1.0 = perfect positive correlation (as X goes up, Y goes up)
 0.0 = no linear relationship
-1.0 = perfect negative correlation (as X goes up, Y goes down)
```

```python
# Pearson correlation between two columns
df['age'].corr(df['salary'])

# Correlation matrix — all pairs at once
df.corr()

# Visualize as a heatmap
import seaborn as sns
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
```

### Why Correlation Matters in ML

**Feature-to-label correlation:** tells you which features are likely useful predictors.
- High correlation with the label → feature is probably informative
- Near-zero correlation with label → feature may not help the model

**Feature-to-feature correlation (multicollinearity):** when two input features are highly correlated with each other.
- Problem: the model can't tell which one is actually causing the effect
- Example: `height_cm` and `height_inches` are perfectly correlated — keep only one
- Rule of thumb: if two features correlate >0.9, consider dropping one

### Important Limitation

**Correlation ≠ causation.** Ice cream sales and drowning rates are correlated (both increase in summer) but ice cream doesn't cause drowning. Always think about *why* a correlation exists before acting on it.

Also: correlation only captures *linear* relationships. Two variables can have a strong nonlinear relationship but a correlation of 0.

### Deep Dive Pointers
- [StatQuest — Correlation](https://www.youtube.com/watch?v=xZ_z8KWkhXE)
- [Spurious Correlations](https://www.tylervigen.com/spurious-correlations) — fun examples of correlation ≠ causation

---

## Week 2 Checklist

- [ ] Can load a CSV and inspect it with `info()`, `describe()`, `head()`
- [ ] Can select rows and columns using `loc` and `iloc`
- [ ] Can filter rows using boolean masks
- [ ] Know the difference between mean and median and when to use each
- [ ] Understand what standard deviation tells you
- [ ] Can identify if a distribution is normal, right-skewed, or left-skewed
- [ ] Know what an outlier is and how to detect one
- [ ] Can compute and interpret a correlation matrix
- [ ] Know what multicollinearity is and why it's a problem
