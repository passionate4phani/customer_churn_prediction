import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_and_save_model(data_path:str, model_path: str):
    """Train model (RandomForest, LogisticRegression, XGBoost) and save best model."""

    # Load processed data
    df = pd.read_csv(data_path)
    X = df.drop("Churn", axis=1)
    y = df["Churn"]


    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline with placeholder classifier
    pipe = Pipeline([
        ("clf", RandomForestClassifier())
    ])

    # Parameter grid for multiple models
    param_grid = [
        {
            "clf":[RandomForestClassifier(random_state=42)],
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_split": [2, 5],
        },
        {
            "clf":[LogisticRegression(max_iter=500, solver="liblinear")],
            "clf__C":[0.01, 0.1, 1, 10],
            "clf__penalty":["l1", "l2"]
        },
        {
            "clf": [XGBClassifier(eval_metric="logloss", random_state=42)],
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [3, 5, 7],
            "clf__learning_rate": [0.01, 0.1, 0.2],
        }
    ]

    # Grid search across models
    grid_search = GridSearchCV(
        pipe,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_
    print(f"Best Model: {best_model}")
    print(f"Best Params: {grid_search.best_params_}")
    print(f"Best CV Score: {grid_search.best_score_:.4f}")

    # Save best model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)
    print(f"Best model saved at {model_path}")

if __name__=="__main__":
    processed_path = "../data/processed/processed_churn.csv"
    model_path = "../models/best_model.pkl"
    train_and_save_model(processed_path, model_path)
