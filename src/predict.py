import joblib
import pandas as pd
from typing import Union

def load_artifacts(model_path: str, preprocessor_path:str):
    """Load trained model and preprocessing pipeline."""
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    for name, transformer, cols in preprocessor.transformers_:
        print(f"Transformer: {name}")
        print(f"  Columns: {cols}")

    return model, preprocessor

def predict_single(input_data: dict, model, preprocessor):
    """
    Predict churn for a single customer.
    input_data: dict where keys = feature names, values = feature values
    """

    df = pd.DataFrame([input_data])
    
    # Ensure numeric columns are correctly typed
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    X_processed = preprocessor.transform(df)
    prediction = model.predict(X_processed)[0]
    probability = model.predict_proba(X_processed)[0][1]
    return prediction, probability

def predict_batch(data: Union[str, pd.DataFrame], model, preprocessor):
    """
    Predict churn for multiple customers.
    data: can be a CSV file path or a pandas DataFrame
    """

    # Load data if a file path is provided
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise ValueError("Input must be a CSV file path or a pandas DataFrame")
    
    X_processed = preprocessor.transform(df)
    predictions = model.predict(X_processed)
    probabilities = model.predict_proba(X_processed)[:,1]

    return predictions, probabilities

if __name__=="__main__":
    model_path = "../models/best_model.pkl"
    preprocessor_path = "../models/preprocessor.pkl"

    # Load artifacts
    model, preprocessor = load_artifacts(model_path, preprocessor_path)

    # Example Single prediction
    sample_input = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 845.5
    }

    pred, prob = predict_single(sample_input, model, preprocessor)
    print(f"Prediction: {'Churn' if pred else 'No Churn'}, Probability: {prob:.2f}")


    # Example batch prediction (CSV)
    '''
    results = predict_batch("../data/raw/sample_batch.csv", model, preprocessor)
    print(results.head())
    '''
