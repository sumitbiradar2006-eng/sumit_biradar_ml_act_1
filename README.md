# sumit_biradar_ml_act_1
# CIE 2 – Activity 1: Multiple Linear Regression Worksheet

**Student Name:** Sumit Biradar
**Roll Number:** AD3124 
**Course:** Regression and Classification Algorithms (T.Y. B.Tech AI & DS, Semester V)  
**Department:** Artificial Intelligence & Data Science, Zeal College of Engineering and Research, Pune  
**Submission GitHub ID:** [https://github.com/complexiota](https://github.com/complexiota)

---

# 📱 Mobile Phone Price Prediction

A Machine Learning project using **Multiple Linear Regression** to predict mobile phone prices based on hardware specifications and brand metadata.

The pipeline features automated data cleaning, regular expression-based feature extraction, categorical encoding using Scikit-learn Pipelines, and model performance evaluation.

---

## 📌 Project Overview

This project processes unstructured metadata from modern mobile phone listings, such as multi-camera setup strings and formatted currency values, to build an end-to-end regression pipeline.

### Key Highlights

- **Data Preprocessing:** Uses regular expressions to extract numerical values from strings (RAM, Storage, Screen Size, Camera Megapixels, and Price).
- **Feature Engineering:** Extracts primary camera specifications from multi-lens configurations (e.g., `12 + 12 + 12`).
- **Scikit-learn Pipeline:** Integrates `ColumnTransformer`, `OneHotEncoder`, and `LinearRegression` for preprocessing and model training.
- **Model Evaluation:** Evaluates performance using R² Score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).
- **Data Visualization:** Uses Actual vs. Predicted visualizations to analyze model performance.

---

## 📊 Model Performance

The model was evaluated using an 80/20 train-test split, where 80% of the dataset was used for training and 20% for testing.

| Evaluation Metric | Score / Value |
|:---|---:|
| **R² Score** | **0.7895** |
| **Variance Explained** | **78.95%** |
| **Mean Absolute Error (MAE)** | **$89.88** |
| **Root Mean Squared Error (RMSE)** | **$138.69** |

---

## 📁 Dataset & Features

**Dataset:** Mobile Phone Price Dataset

**Data Source:** [MobilePhonePrice.csv – GitHub](https://github.com/Senka2112/VIS2023-datasets/blob/main/MobilePhonePrice.csv)

### Features Used

| Feature | Type | Description |
|:---|:---|:---|
| `Brand` | Categorical | Smartphone manufacturer name (One-Hot Encoded). |
| `Storage` | Numerical | Internal storage capacity in GB. |
| `RAM` | Numerical | Random Access Memory capacity in GB. |
| `Screen Size (inches)` | Numerical | Display diagonal size in inches. |
| `Primary_Camera_MP` | Numerical | Primary camera sensor resolution in megapixels. |
| `Battery Capacity (mAh)` | Numerical | Battery capacity in mAh. |

### Target Variable

| Target | Type | Description |
|:---|:---|:---|
| `Price ($)` | Numerical | Retail price of the mobile phone in USD. |

---

## 🛠️ Technology Stack

- **Programming Language:** Python 3.x
- **Data Manipulation:** Pandas, NumPy
- **Feature Extraction:** Python Regular Expressions (`re`)
- **Machine Learning:** Scikit-learn
- **Data Visualization:** Matplotlib, Seaborn

---

## ⚙️ Installation & Usage

### 1. Prerequisites

Ensure Python 3.x is installed on your system.

Install the required dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2. Dataset Setup

1. Download the dataset from the [GitHub repository](https://github.com/Senka2112/VIS2023-datasets/blob/main/MobilePhonePrice.csv).
2. Save the dataset as `Mobile phone price.csv`.
3. Place the CSV file in the root directory of the project.

### 3. Execution Pipeline

Run the Python script or execute the notebook cells sequentially.

The following code demonstrates the complete data preprocessing and model training pipeline:

```python
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# 1. Load Data
df = pd.read_csv('Mobile phone price.csv')
df.columns = df.columns.str.strip()

# 2. Data Cleaning & Feature Extraction
df['Storage'] = (
    df['Storage'].astype(str)
    .str.extract(r'(\d+)').astype(float)
)

df['RAM'] = (
    df['RAM'].astype(str)
    .str.extract(r'(\d+)').astype(float)
)

df['Screen Size (inches)'] = (
    df['Screen Size (inches)'].astype(str)
    .str.extract(r'([\d.]+)').astype(float)
)

df['Price ($)'] = (
    df['Price ($)'].astype(str)
    .str.replace('$', '', regex=False)
    .str.replace(',', '', regex=False)
    .astype(float)
)

# Extract primary camera megapixels
def get_primary_camera(val):
    nums = re.findall(r'[\d.]+', str(val))
    return float(nums[0]) if nums else np.nan

df['Primary_Camera_MP'] = (
    df['Camera (MP)'].apply(get_primary_camera)
)

# Remove rows containing missing required values
df_clean = df.dropna(
    subset=[
        'Storage',
        'RAM',
        'Screen Size (inches)',
        'Primary_Camera_MP',
        'Battery Capacity (mAh)',
        'Price ($)'
    ]
)

# 3. Features & Target
X = df_clean[
    [
        'Brand',
        'Storage',
        'RAM',
        'Screen Size (inches)',
        'Primary_Camera_MP',
        'Battery Capacity (mAh)'
    ]
]

y = df_clean['Price ($)']

# 4. Pipeline Setup
preprocessor = ColumnTransformer(
    transformers=[
        (
            'num',
            'passthrough',
            [
                'Storage',
                'RAM',
                'Screen Size (inches)',
                'Primary_Camera_MP',
                'Battery Capacity (mAh)'
            ]
        ),
        (
            'cat',
            OneHotEncoder(handle_unknown='ignore'),
            ['Brand']
        )
    ]
)

model = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ]
)

# 5. Model Training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model.fit(X_train, y_train)
```

---

## 🔮 Making Predictions

The trained model can predict the estimated price of a new mobile phone using its hardware specifications.

### Example: Samsung Smartphone

```python
sample_device = pd.DataFrame([{
    'Brand': 'Samsung',
    'Storage': 128.0,
    'RAM': 8.0,
    'Screen Size (inches)': 6.5,
    'Primary_Camera_MP': 48.0,
    'Battery Capacity (mAh)': 5000
}])

predicted_price = model.predict(sample_device)

print(f"Estimated Price: ${predicted_price[0]:.2f}")
```

**Example Output:**

```text
Estimated Price: $572.24
```

---

## 📈 Conclusion

This project demonstrates the application of Multiple Linear Regression for predicting mobile phone prices based on hardware specifications and brand information.

By combining data cleaning, regular expression-based feature extraction, categorical encoding, and Scikit-learn Pipelines, the project establishes a structured machine learning workflow.

The model achieved an **R² score of 0.7895**, explaining approximately **78.95% of the variance** in mobile phone prices on the test dataset.

---

## 📜 License

This project is developed for educational purposes as part of the Regression and Classification Algorithms coursework.  


---
