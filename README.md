# 🏥 Insurance EDA & Medical Cost Prediction

## 📌 Project Overview

This project explores the **Medical Cost Personal Dataset** through Exploratory Data Analysis (EDA), data cleaning, feature encoding, feature engineering, statistical feature selection, and a basic Linear Regression model.

The project starts with an exploratory analysis of demographic, lifestyle, and health-related variables and investigates how they relate to **medical insurance charges**.

The workflow includes:

- Dataset inspection and understanding
- Missing-value and duplicate checking
- Numerical and categorical data visualization
- Outlier inspection
- Correlation analysis
- Data cleaning
- Categorical encoding
- BMI-based feature engineering
- Feature scaling
- Pearson correlation analysis
- Chi-square statistical testing
- Feature selection
- Train/test splitting
- Linear Regression modeling
- R² and Adjusted R² evaluation

> **Project type:** EDA + preprocessing + introductory regression modeling

---

## 🎯 Project Objectives

The main objectives of this project are to:

1. Understand the structure and characteristics of the insurance dataset.
2. Explore numerical and categorical variables visually.
3. Identify duplicate records and check for missing values.
4. Examine relationships between input variables and insurance charges.
5. Convert categorical variables into numerical representations.
6. Create a BMI category feature through feature engineering.
7. Use statistical techniques to investigate potentially relevant features.
8. Select a smaller set of features for regression modeling.
9. Build a Linear Regression model to predict insurance charges.
10. Evaluate the regression model using R² and Adjusted R².

---

## 📂 Project Structure

```text
Insurance-EDA/
│
├── insurance.csv
├── Untitled(1).ipynb
└── README.md
```

### File Description

| File | Description |
|---|---|
| `insurance.csv` | Insurance dataset used for analysis and modeling |
| `Untitled(1).ipynb` | Jupyter Notebook containing the complete EDA, preprocessing, statistical analysis, and Linear Regression workflow |
| `README.md` | Project documentation |

> **Recommendation:** Before publishing the project, the notebook can be renamed from `Untitled(1).ipynb` to a more descriptive name such as `Insurance_EDA.ipynb`.

---

## 📊 Dataset

The project uses `insurance.csv`.

The dataset contains **1,338 rows and 7 columns** before duplicate removal.

### Features

| Feature | Data Type | Description |
|---|---|---|
| `age` | Integer | Age of the individual |
| `sex` | Categorical | Sex of the individual |
| `bmi` | Float | Body Mass Index |
| `children` | Integer | Number of children/dependents |
| `smoker` | Categorical | Whether the individual is a smoker |
| `region` | Categorical | Residential region |
| `charges` | Float | Medical insurance charges; prediction target |

### Target Variable

The target variable is:

```text
charges
```

The project uses the available demographic, BMI, family, smoking, and regional information to model medical insurance charges.

---

## 🛠️ Technologies & Libraries

The notebook uses the following Python libraries:

- **Python**
- **NumPy** — numerical operations
- **Pandas** — data loading and manipulation
- **Matplotlib** — data visualization
- **Seaborn** — statistical visualization
- **SciPy** — statistical analysis
- **Scikit-learn** — preprocessing, train/test splitting, Linear Regression, and model evaluation

### Imports Used

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

For statistical analysis:

```python
from scipy.stats import pearsonr
from scipy.stats import chi2_contingency
```

For preprocessing and modeling:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
```

---

# 🔍 Exploratory Data Analysis

## 1. Loading the Dataset

The dataset is loaded using Pandas:

```python
df = pd.read_csv('insurance.csv')
```

The notebook then examines the dataset using:

- `df.shape`
- `df.head()`
- `df.info()`
- `df.describe()`
- `df.isnull().sum()`
- `df.columns`

---

## 2. Dataset Dimensions

The original dataset contains:

```text
Rows:    1338
Columns: 7
```

After duplicate removal, the cleaned dataset contains:

```text
Rows:    1337
Columns: 7
```

---

## 3. Missing Values

The notebook checks every column for null values.

The dataset contains **no missing values** according to Pandas:

```text
age         0
sex         0
bmi         0
children    0
smoker      0
region      0
charges     0
```

---

## 4. Duplicate Records

The notebook checks for duplicate observations and removes them:

```python
df_cleaned.drop_duplicates(inplace=True)
```

The original dataset contains **1 duplicate row**, leaving **1,337 observations** after cleaning.

---

# 📈 Data Visualization

The project uses several visualizations to understand the dataset.

## Numerical Feature Distributions

Histograms with KDE are created for:

- `age`
- `bmi`
- `children`
- `charges`

Example:

```python
sns.histplot(df[col], kde=True, bins=20)
```

These visualizations help inspect the distribution and spread of numerical variables.

---

## Categorical Feature Distributions

Count plots are used for categorical/discrete variables including:

- `children`
- `sex`
- `smoker`

Example:

```python
sns.countplot(x=df['sex'])
```

These plots provide a visual understanding of the frequency of different categories.

---

# 📦 Outlier Analysis

Box plots are created for the numerical variables:

```python
for col in Numeric_col:
    sns.boxplot(x=df[col])
```

The variables examined include:

- `age`
- `bmi`
- `children`
- `charges`

Box plots help identify the spread of values and potential extreme observations.

---

# 🔗 Correlation Analysis

A correlation heatmap is created using:

```python
sns.heatmap(df.corr(numeric_only=True), annot=True)
```

The project then calculates Pearson correlations between selected features and `charges`.

### Pearson Correlation Results

After preprocessing, the strongest positive relationship with `charges` among the selected features is:

| Feature | Pearson Correlation |
|---|---:|
| `is_smoker` | 0.7872 |
| `age` | 0.2983 |
| `bmi_category_Obese` | 0.1977 |
| `bmi` | 0.1962 |
| `region_southeast` | 0.0736 |
| `children` | 0.0674 |
| `is_female` | -0.0580 |
| `region_southwest` | -0.0436 |
| `region_northwest` | -0.0387 |
| `bmi_category_Normal` | -0.1057 |
| `bmi_category_Overweight` | -0.1183 |

The analysis shows that `is_smoker` has the strongest linear correlation with insurance charges among the selected features.

> Correlation describes statistical association and does not by itself establish causation.

---

# 🧹 Data Cleaning & Preprocessing

A separate cleaned dataframe is created:

```python
df_cleaned = df.copy()
```

## Duplicate Removal

Duplicate records are removed:

```python
df_cleaned.drop_duplicates(inplace=True)
```

## Missing-Value Check

The cleaned dataset is checked again:

```python
df_cleaned.isnull().sum()
```

No missing values are present.

---

# 🔢 Categorical Encoding

The categorical variables are converted into numerical representations.

## Sex Encoding

The `sex` column is mapped as:

```python
{'male': 0, 'female': 1}
```

It is then renamed:

```text
sex → is_female
```

---

## Smoker Encoding

The `smoker` column is mapped as:

```python
{'no': 0, 'yes': 1}
```

It is then renamed:

```text
smoker → is_smoker
```

This makes the variables easier to use in numerical analysis and regression modeling.

---

# 🌍 Region Encoding

The `region` column contains multiple categories, so One-Hot Encoding is applied:

```python
pd.get_dummies(
    df_cleaned,
    columns=['region'],
    drop_first=True
)
```

This creates binary region features while dropping the first category to avoid redundant representation.

The resulting features include:

```text
region_northwest
region_southeast
region_southwest
```

---

# 🧬 Feature Engineering

The project creates a new feature based on BMI.

## BMI Distribution

The notebook first examines the BMI distribution:

```python
sns.histplot(df['bmi'])
```

## BMI Categories

BMI is divided into four categories:

| BMI Range | Category |
|---|---|
| ≤ 18.5 | UnderWeight |
| 18.5–24.9 | Normal |
| 25.0–29.9 | Overweight |
| ≥ 30 | Obese |

The feature is created using:

```python
pd.cut(
    df_cleaned['bmi'],
    bins=[0, 18.5, 24.9, 29.9, float('inf')],
    labels=['UnderWeight', 'Normal', 'Overweight', 'Obese']
)
```

The resulting categorical feature is then one-hot encoded.

---

# ⚖️ Feature Scaling

Standardization is applied to:

```text
age
bmi
children
```

using `StandardScaler`:

```python
scaler = StandardScaler()

df_cleaned[cols] = scaler.fit_transform(df_cleaned[cols])
```

The scaling transforms these numerical features into standardized values.

---

# 📊 Statistical Feature Analysis

The project uses statistical techniques to investigate relationships between features and insurance charges.

## Pearson Correlation

Pearson correlation is calculated using:

```python
pearsonr(feature, charges)
```

This measures the strength and direction of a linear relationship between each selected feature and `charges`.

---

## Chi-Square Test

Categorical features are evaluated using a Chi-square test.

Because `charges` is continuous, the notebook first divides it into four quantile-based groups:

```python
df_cleaned['charges_bin'] = pd.qcut(
    df_cleaned['charges'],
    q=4,
    labels=False
)
```

The categorical features are then compared with the binned charges.

The significance level is:

```text
α = 0.05
```

### Chi-Square Results

| Feature | Chi-Square | p-value | Notebook Decision |
|---|---:|---:|---|
| `is_smoker` | 848.219 | 1.51e-183 | Reject Null |
| `region_southeast` | 15.998 | 0.00113 | Reject Null |
| `is_female` | 10.259 | 0.01649 | Reject Null |
| `bmi_category_Obese` | 7.654 | 0.05372 | Accept Null |
| `region_southwest` | 5.092 | 0.16519 | Accept Null |
| `bmi_category_Normal` | 4.264 | 0.23436 | Accept Null |
| `bmi_category_Overweight` | 4.202 | 0.24058 | Accept Null |
| `region_northwest` | 1.134 | 0.76882 | Accept Null |

Based on the notebook's `α = 0.05` decision rule, the features retained from this categorical analysis include:

- `is_smoker`
- `region_southeast`
- `is_female`

---

# 🎯 Final Feature Selection

After the preprocessing and statistical analysis, the notebook creates the final modeling dataframe:

```python
final_df = df_cleaned[
    [
        'age',
        'is_female',
        'bmi',
        'children',
        'is_smoker',
        'charges',
        'region_southeast'
    ]
]
```

### Final Features

```text
age
is_female
bmi
children
is_smoker
region_southeast
```

### Target

```text
charges
```

---

# 🤖 Linear Regression Model

The project uses **Linear Regression** to model insurance charges.

## Train/Test Split

The data is divided into training and testing sets using:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

This produces:

```text
Training samples: 1069
Testing samples:   268
```

---

## Model Training

The Linear Regression model is created and trained:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)
```

---

## Predictions

Predictions are generated using:

```python
y_pred = model.predict(X_test)
```

---

# 📏 Model Evaluation

The notebook evaluates the model using **R²** and **Adjusted R²**.

## R² Score

The resulting test R² is approximately:

```text
R² = 0.8054
```

This means the model explains approximately **80.54% of the variance in the test-set insurance charges**.

## Adjusted R²

The resulting Adjusted R² is approximately:

```text
Adjusted R² = 0.8009
```

The notebook contains a comment describing the result as "80.5% Accuracy." For a regression model, however, the reported metric is **R² rather than classification accuracy**, so this README uses the more precise interpretation of the result.

---

# 🔄 Complete Project Workflow

```text
Load insurance.csv
        │
        ▼
Understand Dataset
        │
        ▼
EDA
├── Shape
├── Head / Info / Describe
├── Missing Values
├── Numerical Distributions
├── Categorical Distributions
├── Outlier Inspection
└── Correlation Heatmap
        │
        ▼
Data Cleaning
├── Copy Dataset
├── Remove Duplicates
└── Check Missing Values
        │
        ▼
Categorical Encoding
├── Sex → is_female
├── Smoker → is_smoker
└── Region → One-Hot Encoding
        │
        ▼
Feature Engineering
└── BMI → BMI Categories
        │
        ▼
Feature Scaling
└── StandardScaler
        │
        ▼
Statistical Analysis
├── Pearson Correlation
└── Chi-Square Test
        │
        ▼
Feature Selection
        │
        ▼
Train/Test Split
        │
        ▼
Linear Regression
        │
        ▼
Predictions
        │
        ▼
R² + Adjusted R²
```

---

# 🔑 Key Findings

Based on the analysis performed in the notebook:

- The dataset contains **1,338 original observations** and **7 variables**.
- There are **no missing values** in the dataset.
- **1 duplicate record** is removed during preprocessing.
- `is_smoker` has the strongest Pearson correlation with `charges` among the selected features, with a correlation of approximately **0.7872**.
- `age` has a positive Pearson correlation of approximately **0.2983** with charges.
- `bmi` has a positive Pearson correlation of approximately **0.1962** with charges.
- The Chi-square analysis identifies `is_smoker`, `region_southeast`, and `is_female` as statistically significant at the notebook's **0.05 significance level**.
- The final Linear Regression model uses six input features.
- The model achieves a test **R² of approximately 0.8054**.
- The corresponding **Adjusted R² is approximately 0.8009**.

---

# 🚀 How to Run the Project

## 1. Clone the Repository

```bash
git clone <your-repository-url>
```

Replace `<your-repository-url>` with the URL of your GitHub repository.

## 2. Navigate to the Project Directory

```bash
cd Insurance-EDA
```

## 3. Install Dependencies

You can install the required libraries with:

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn jupyter
```

## 4. Start Jupyter Notebook

```bash
jupyter notebook
```

Then open:

```text
Untitled(1).ipynb
```

and run the notebook cells in order.

---

# 📦 Suggested requirements.txt

A `requirements.txt` file for this project can contain:

```text
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
jupyter
```

Install them with:

```bash
pip install -r requirements.txt
```

---

# ⚠️ Methodological Note

The current notebook fits `StandardScaler` on the full cleaned dataset before performing the train/test split:

```python
scaler.fit_transform(df_cleaned[cols])
```

For a production machine-learning pipeline, it is generally preferable to:

1. Split the data into training and testing sets first.
2. Fit the scaler only on the training data.
3. Transform both training and testing data using that fitted scaler.

This prevents information from the test set from influencing preprocessing.

The README documents the notebook as it currently exists and does not modify the project's workflow.

---

# 🔮 Future Improvements

Possible future improvements include:

- Rename the notebook to `Insurance_EDA.ipynb`.
- Create a dedicated `requirements.txt`.
- Build additional regression models such as:
  - Ridge Regression
  - Lasso Regression
  - Decision Tree Regression
  - Random Forest Regression
  - Gradient Boosting Regression
- Compare multiple models using consistent evaluation metrics.
- Add MAE and RMSE alongside R².
- Use cross-validation.
- Perform hyperparameter tuning.
- Build a proper preprocessing pipeline with `Pipeline` and `ColumnTransformer`.
- Prevent preprocessing leakage by fitting transformations only on training data.
- Create additional visualizations for important feature relationships.
- Deploy the final model through a Streamlit application.

---

# 📚 Learning Outcomes

This project demonstrates practical experience with:

- Loading datasets with Pandas
- Understanding data types and dataset structure
- Exploratory Data Analysis
- Numerical and categorical visualization
- Missing-value analysis
- Duplicate detection and removal
- Outlier inspection
- Correlation analysis
- Categorical encoding
- One-hot encoding
- Feature engineering
- Feature scaling
- Pearson correlation
- Chi-square testing
- Feature selection
- Train/test splitting
- Linear Regression
- R² evaluation
- Adjusted R² evaluation
- Building an end-to-end introductory data science workflow

---

# 👤 Author

**Fowad Ajmal**

---

## 📌 Project Summary

This project provides an end-to-end introduction to analyzing medical insurance costs using Python. It combines **EDA, preprocessing, statistical analysis, feature engineering, feature selection, and Linear Regression** to investigate the factors associated with insurance charges.

The analysis finds a strong statistical relationship between smoking status and insurance charges and produces a Linear Regression model with a test R² of approximately **0.8054**.

