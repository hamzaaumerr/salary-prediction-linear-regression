# Salary Prediction using Linear Regression

A Machine Learning project that predicts employee salaries based on years of professional experience using **Linear Regression** and Python's Scikit-learn library.

## Project Overview

This project implements a complete beginner-level supervised Machine Learning workflow. It uses a real world salary dataset to train a Linear Regression model, evaluate its performance, analyze feature coefficients and predict salaries for employees with different levels of experience.

## Dataset

- **Dataset Name:** Salary Data
- **Source:** [Salary Data – GitHub](https://raw.githubusercontent.com/unimi-dse/959f8daa/master/data/Salary_Data.csv)
- **Input Feature:** Years of Experience
- **Target Variable:** Salary
- **Data Format:** CSV

The dataset is loaded directly from its GitHub URL using Pandas.

## Technologies & Libraries

- **Python** – Programming language
- **NumPy** – Numerical computations
- **Pandas** – Dataset loading and manipulation
- **Scikit-learn** – Machine Learning, preprocessing, model training and evaluation

## Machine Learning Workflow

The project follows these steps:

1. **Data Loading:** Load the salary dataset using Pandas.
2. **Data Inspection:** Examine dataset dimensions and sample records.
3. **Data Preparation:** Rename columns and separate features (X) and target (y).
4. **Train-Test Split:** Divide the dataset into 80% training and 20% testing data.
5. **Feature Scaling:** Apply StandardScaler to standardize input features.
6. **Model Training:** Train a Linear Regression model using the training data.
7. **Model Evaluation:** Evaluate training and testing performance.
8. **Cross-Validation:** Perform 5-fold cross-validation to assess model performance.
9. **Feature Analysis:** Examine regression coefficients to understand feature influence.
10. **Salary Prediction:** Predict salaries for new employee profiles.

## Model Evaluation

The model uses the following evaluation metrics:

| Metric | Purpose |
|--------|---------|
| Train R² Score | Measures model fit on training data |
| Test R² Score | Measures model performance on unseen test data |
| RMSE | Measures prediction error in the target variable's units |
| Cross-Validation R² | Evaluates performance across multiple training folds |

The actual evaluation results are displayed when the Python script is executed.

## Example Predictions

The model generates salary predictions for employees with the following experience levels:

| Employee | Years of Experience |
|----------|--------------------:|
| Employee 1 | 2 years |
| Employee 2 | 5 years |
| Employee 3 | 10 years |

Predicted salaries are displayed in the terminal after running the program.


## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/hamzaaumerr/salary-prediction-linear-regression
cd salary-prediction-linear-regression
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Project

```bash
python salary_prediction.py
```