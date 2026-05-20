import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'Crop_recommendation.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

FEATURE_COLS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
TARGET_COL = 'label'


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def explore_data(df):
    print("=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nBasic statistics:\n{df[FEATURE_COLS].describe()}")
    print(f"\nCrop classes ({df[TARGET_COL].nunique()}): {sorted(df[TARGET_COL].unique())}")


def handle_missing_values(df):
    # Fill numeric columns with column mean
    for col in FEATURE_COLS:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())
    return df


def remove_outliers(df):
    # Clip values beyond 3 standard deviations for each feature column
    for col in FEATURE_COLS:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = df[col].clip(lower=mean - 3 * std, upper=mean + 3 * std)
    return df


def encode_labels(df):
    encoder = LabelEncoder()
    df['label_encoded'] = encoder.fit_transform(df[TARGET_COL])
    return df, encoder


def create_synthetic_yield(df):
    """
    The dataset has no yield column, so we engineer a synthetic one.
    Formula: yield ≈ 0.4*rainfall + 0.3*N + 0.2*temperature + 0.1*humidity + noise
    This is normalised to a realistic kg/hectare range (1000–8000).
    Documented in the report as synthetically derived for lab demonstration.
    """
    np.random.seed(42)
    noise = np.random.normal(0, 50, len(df))
    raw = (
        0.4 * df['rainfall'] +
        0.3 * df['N'] +
        0.2 * df['temperature'] +
        0.1 * df['humidity'] +
        noise
    )
    # Scale to a realistic agronomic range
    raw_min, raw_max = raw.min(), raw.max()
    df['yield_kg_per_ha'] = 1000 + (raw - raw_min) / (raw_max - raw_min) * 7000
    return df


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def run_preprocessing():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load
    df = load_data()
    explore_data(df)

    # 2. Clean
    df = handle_missing_values(df)
    df = remove_outliers(df)

    # 3. Synthetic yield column for Linear Regression
    df = create_synthetic_yield(df)

    # 4. Encode crop labels
    df, encoder = encode_labels(df)

    # 5. Prepare features and targets
    X = df[FEATURE_COLS].values
    y_class = df['label_encoded'].values      # for Decision Tree
    y_cluster = df[FEATURE_COLS].values       # for KMeans (unsupervised, no labels)
    y_yield = df['yield_kg_per_ha'].values    # for Linear Regression

    # 6. Train/test split (classification + regression share the same split)
    X_train, X_test, yc_train, yc_test, yy_train, yy_test = train_test_split(
        X, y_class, y_yield, test_size=0.2, random_state=42
    )

    # 7. Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # 8. Serialize scaler and encoder so the GUI can reuse them
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(encoder, os.path.join(MODELS_DIR, 'label_encoder.pkl'))

    print("\n[preprocessing] Scaler and LabelEncoder saved to models/")

    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'X_all': scaler.transform(X),         # full scaled dataset for KMeans
        'yc_train': yc_train,
        'yc_test': yc_test,
        'yy_train': yy_train,
        'yy_test': yy_test,
        'feature_names': FEATURE_COLS,
        'encoder': encoder,
        'scaler': scaler,
        'df': df,
    }


if __name__ == '__main__':
    data = run_preprocessing()
    print("\n[preprocessing] Done. Keys available:", list(data.keys()))
