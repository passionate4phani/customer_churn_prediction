import pandas as pd
import os
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_data(file_path: str) -> pd.DataFrame:
    """Load raw churn dataset."""
    return pd.read_csv(file_path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw churn dataset (remove duplicates, fix dtypes, etc..)"""

    # Drop customerID if exists
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    # Convert TotalCharges to numeric (coerce errors)
    if "TotalCharges" in df.columns:
        df['TotalCharges'] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df['TotalCharges'].median())

    # Encode target column Churn (No=0, Yes=1)
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({"No": 0, "Yes": 1})

    # Remove duplicate rows if any
    df = df.drop_duplicates()

    return df

def preprocess_and_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Apply OneHotEncoding and Scaling, return full processed DataFrame"""

    # Seperate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Define preprocessing for categorical & numerical data
    categorical_transfomer = OneHotEncoder(handle_unknown='ignore')
    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transfomer, categorical_cols),
            ('num', numerical_transformer, numeric_cols)
        ]
    )

    # Fit & transform
    X_processed = preprocessor.fit_transform(X)

    # Covert processed features to Dataframe for saving to processed file
    processed_feature_names = (
        preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist() + numeric_cols
    )

    X_processed_df = pd.DataFrame(X_processed, columns=processed_feature_names)
    X_processed_df['Churn'] = y.values

    # Save preprocessor file
    joblib.dump(preprocessor, "../models/preprocessor.pkl")

    return X_processed_df

def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """Save processed dataset to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved at {output_path}")

if __name__ == "__main__":
    # Example usage
    raw_path = "../data/raw/Telco-Customer-Churn.csv"
    processed_path = "../data/processed/processed_churn.csv"
    
    
    df_raw = load_data(raw_path)
    df_clean = clean_data(df_raw)
    df_processed = preprocess_and_encode(df_clean)
    save_processed_data(df_processed, processed_path)