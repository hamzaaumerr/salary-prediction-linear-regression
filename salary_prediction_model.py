import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def linear_regression_pipeline(X, y, feature_names=None):

    # 1. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # 4. Evaluate
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    print("=== Model Evaluation ===")
    print(f"Train R2: {r2_score(y_train, y_pred_train):.4f}")
    print(f"Test R2:  {r2_score(y_test, y_pred_test):.4f}")
    print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")

    # 5. Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"\nCross-validation R2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # 6. Feature importance
    if feature_names:
        print("\n=== Feature Importance ===")
        importance = pd.DataFrame({
            'feature': feature_names,
            'coefficient': model.coef_
        }).sort_values('coefficient', key=abs, ascending=False)
        print(importance.to_string(index=False))

    return model, scaler


# 1. Load the real dataset
url = "https://raw.githubusercontent.com/unimi-dse/959f8daa/master/data/Salary_Data.csv"
df = pd.read_csv(url)

# 2. Inspect and prepare the data
print("Dataset shape:", df.shape)
print(df.head())

df = df.rename(columns={"YearsExperience": "years_exp", "Salary": "salary"})

X = df[["years_exp"]]
y = df["salary"]

# 3. Run the full workflow
model, scaler = linear_regression_pipeline(X, y, feature_names=list(X.columns))

# 4. Predict salaries for new employees
profiles = pd.DataFrame({
    "years_exp": [2, 5, 10]
})

profiles_scaled = scaler.transform(profiles)
predictions = model.predict(profiles_scaled)

print("\n========== SALARY PREDICTIONS ==========")
for i, salary_prediction in enumerate(predictions):
    print(f"\nEmployee {i + 1}")
    print(f"Experience: {profiles.iloc[i]['years_exp']} years")
    print(f"Predicted Salary: ${salary_prediction:,.2f}")