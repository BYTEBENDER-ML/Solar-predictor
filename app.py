from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import warnings
import streamlit as st
import joblib

warnings.filterwarnings("ignore")

# ------------------------
# FEATURES
# ------------------------
FEATURES = [
    'temperature', 'irradiance', 'humidity', 'panel_age',
    'maintenance_count', 'soiling_ratio', 'voltage', 'current',
    'module_temperature', 'cloud_coverage', 'wind_speed', 'pressure',
    'string_id', 'error_code', 'installation_type'
]

# ------------------------
# DATA PREPROCESSING
# ------------------------
def preprocess_data(df, is_test=False, scaler=None, label_encoders=None, imputers=None):
    df = df.copy()

    # Separate target (efficiency)
    y = None
    if 'efficiency' in df.columns:
        y = df['efficiency'].copy()

    # Keep only relevant features (ensure columns exist)
    df = df.reindex(columns=FEATURES)

    # Identify categorical / numerical
    categorical_cols = ['string_id', 'error_code', 'installation_type']
    numerical_cols = [col for col in df.columns if col not in categorical_cols]

    # Convert numeric columns to numeric (coerce errors -> NaN)
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle categorical columns as strings
    for col in categorical_cols:
        df[col] = df[col].astype(str).fillna('missing')

    # Impute numerical missing values
    if not is_test:
        imputers = {}
        for col in numerical_cols:
            median = df[col].median()
            if np.isnan(median):
                median = 0.0
            imputers[col] = median
            df[col] = df[col].fillna(median)
    else:
        # use provided imputers
        if imputers is None:
            raise ValueError("imputers required for is_test=True")
        for col in numerical_cols:
            df[col] = df[col].fillna(imputers.get(col, 0.0))

    # Encode categorical columns using category lists (deterministic mapping)
    if not is_test:
        label_encoders = {}
        for col in categorical_cols:
            cats = list(pd.Series(df[col].fillna('missing')).astype(str).unique())
            label_encoders[col] = cats
            # map to integer codes, unseen handled later
            mapping = {v: i for i, v in enumerate(cats)}
            df[col] = df[col].map(mapping).fillna(len(cats)).astype(int)
    else:
        if label_encoders is None:
            raise ValueError("label_encoders required for is_test=True")
        for col in categorical_cols:
            cats = label_encoders.get(col, [])
            mapping = {v: i for i, v in enumerate(cats)}
            # unseen -> code = len(cats)
            df[col] = df[col].map(lambda x: mapping.get(str(x), len(cats))).astype(int)

    # Scale features
    if not is_test:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    else:
        if scaler is None:
            raise ValueError("scaler required for is_test=True")
        X = pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)

    # For training, fill NaNs in y using training median
    if y is not None:
        if not is_test:
            y = pd.to_numeric(y, errors='coerce')
            if y.isna().all():
                # fallback to zero if entirely missing
                y = y.fillna(0.0)
            else:
                y = y.fillna(y.median())
        else:
            # For test, keep numeric and don't fill here (we'll drop NaN targets before training/eval)
            y = pd.to_numeric(y, errors='coerce')

    return X, y, scaler, label_encoders, imputers

# ------------------------
# MODEL TRAINING & EVALUATION
# ------------------------
def train_and_evaluate(df):
    # Validate target presence / sufficiency before any split/preprocess
    if 'efficiency' not in df.columns:
        raise ValueError("Missing target column 'efficiency' in the uploaded file.")

    non_null_targets = df['efficiency'].replace('', np.nan).dropna()
    n_targets = non_null_targets.shape[0]
    if n_targets < 2:
        raise ValueError(f"Not enough non-missing 'efficiency' values: found {n_targets}. Provide at least 2 rows with targets to train.")

    # split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # preprocess train (fit)
    X_train, y_train, scaler, label_encoders, imputers = preprocess_data(train_df, is_test=False)

    # If preprocessing produced no training rows (all targets got removed), try fallback:
    if y_train is None or y_train.dropna().shape[0] == 0 or X_train.shape[0] == 0:
        # Attempt to use all available rows with non-null target for training (no split)
        df_nonnull = df.loc[df['efficiency'].replace('', np.nan).notna()].copy()
        if df_nonnull.shape[0] < 2:
            raise ValueError("After preprocessing there are still fewer than 2 training rows with non-missing targets. Please provide more labeled data.")
        # re-preprocess using only labeled rows
        X_train, y_train, scaler, label_encoders, imputers = preprocess_data(df_nonnull, is_test=False)
        # create an empty test set to skip evaluation (or keep a tiny split)
        X_test = X_train.iloc[:1]  # placeholder
        y_test = y_train.iloc[:1]
    else:
        # preprocess test (use fitted)
        X_test, y_test, _, _, _ = preprocess_data(test_df, is_test=True, scaler=scaler,
                                                 label_encoders=label_encoders, imputers=imputers)

    # Drop rows where target is NaN (only if present in file)
    if y_train is not None:
        train_mask = ~y_train.isna()
        X_train, y_train = X_train.loc[train_mask], y_train.loc[train_mask]
    if y_test is not None:
        test_mask = ~y_test.isna()
        X_test, y_test = X_test.loc[test_mask], y_test.loc[test_mask]

    # Final safety checks
    if X_train.shape[0] == 0 or y_train.shape[0] == 0:
        raise ValueError("No training samples remaining after preprocessing. Check 'efficiency' values and preprocessing steps.")

    # Train model
    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Predict & metrics (if test set empty skip metrics)
    if X_test.shape[0] == 0 or y_test.shape[0] == 0:
        return {
            "model": model,
            "scaler": scaler,
            "label_encoders": label_encoders,
            "imputers": imputers,
            "mae": None,
            "rmse": None,
            "r2": None
        }

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    return {
        "model": model,
        "scaler": scaler,
        "label_encoders": label_encoders,
        "imputers": imputers,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }

# ------------------------
# STREAMLIT APP
# ------------------------
st.set_page_config(page_title="Solar Panel Efficiency Predictor", layout="wide")
st.title("🌞 Solar Panel Efficiency Predictor")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    try:
        results = train_and_evaluate(df)
    except Exception as e:
        st.error(f"Error during training: {e}")
    else:
        st.success("Model trained")
        st.write(f"MAE: {results['mae']:.4f}")
        st.write(f"RMSE: {results['rmse']:.4f}")
        st.write(f"R2: {results['r2']:.4f}")
        # Save model and artifacts
        joblib.dump(results['model'], "rf_model.joblib")
        joblib.dump({'scaler': results['scaler'], 'label_encoders': results['label_encoders'],
                     'imputers': results['imputers']}, "artifacts.joblib")
else:
    st.info("Upload a CSV to get started.")
