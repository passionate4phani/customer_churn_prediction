import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)
from sklearn.model_selection import learning_curve, train_test_split

def evaluate_model(data_path: str, model_path:str, report_dir:str = "../reports"):
    """Load model and evaluate performance."""

    # Create folders
    plots_dir = os.path.join(report_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Load processed data
    df = pd.read_csv(data_path)
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Split data into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Load trained model
    model = joblib.load(model_path)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    # Metrics
    metrics = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "F1 Score":  f1_score(y_test, y_pred),
        "ROC-AUC":   roc_auc_score(y_test, y_prob)
    }

    print("Model Evaluation Results (on Test Set):")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Save metrics to file
    with open(os.path.join(report_dir, "metrics.txt"), "w") as f:
        f.write("Model Evaluation Results (on Test Set)\n")
        for k, v in metrics.items():
            f.write(f"{k}:{v:.4f}\n")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    cm_path = os.path.join(plots_dir, "confution_metrix.png")
    plt.savefig(cm_path)
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="black")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    roc_path = os.path.join(plots_dir, "roc_curve.png")
    plt.savefig(roc_path)
    plt.close()

    # Learning Curve
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=5, scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.figure(figsize=(6, 5))
    plt.plot(train_sizes, train_mean, "o-", label="Training Score")
    plt.plot(train_sizes, val_mean, "o-", label="Cross-Validation Score")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend()
    plt.tight_layout()
    lc_path = os.path.join(plots_dir, "learning_curve.png")
    plt.savefig(lc_path)
    plt.close()

    print(f"\n Metrics saved to : {os.path.join(report_dir, 'metrics.text')}")
    print(f" Plots saved in : {plots_dir}")

if __name__=="__main__":
    processed_path = "../data/processed/processed_churn.csv"
    model_path = "../models/best_model.pkl"
    evaluate_model(processed_path, model_path)
