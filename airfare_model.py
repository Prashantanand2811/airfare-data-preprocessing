"""
Airfare Price Prediction Model
Author: Prashant Anand
Description: Train and evaluate ML models to predict airfare prices
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# LOAD CLEANED DATA (from airfare_pipeline.py)
# ─────────────────────────────────────────

def load_data():
    """Load the cleaned airfare dataset."""
    from airfare_pipeline import (
        generate_raw_airfare_data,
        clean_airfare_data,
        engineer_features
    )
    raw     = generate_raw_airfare_data(n=500)
    cleaned = clean_airfare_data(raw)
    df      = engineer_features(cleaned)
    return df


# ─────────────────────────────────────────
# FEATURE PREPARATION
# ─────────────────────────────────────────

def prepare_features(df):
    """Encode and select features for modeling."""
    df = df.copy()

    le = LabelEncoder()
    df['airline_enc']      = le.fit_transform(df['airline'].astype(str))
    df['origin_enc']       = le.fit_transform(df['origin'].astype(str))
    df['destination_enc']  = le.fit_transform(df['destination'].astype(str))
    df['cabin_enc']        = le.fit_transform(df['cabin_class'].astype(str))
    df['booking_window_enc'] = le.fit_transform(df['booking_window'].astype(str))

    features = [
        'airline_enc', 'origin_enc', 'destination_enc',
        'duration_hrs', 'stops', 'days_before_departure',
        'cabin_enc', 'cabin_rank', 'month', 'day_of_week',
        'is_weekend_flight', 'is_holiday_season', 'booking_window_enc'
    ]

    df = df.dropna(subset=features + ['price_usd'])
    X  = df[features]
    y  = df['price_usd']
    return X, y, features


# ─────────────────────────────────────────
# TRAIN & EVALUATE MODELS
# ─────────────────────────────────────────

def train_models(X, y):
    """Train multiple models and compare performance."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler  = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_test  = scaler.transform(X_test)

    models = {
        'Linear Regression':       LinearRegression(),
        'Ridge Regression':        Ridge(alpha=1.0),
        'Random Forest':           RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting':       GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    results = {}
    print(f"\n{'='*60}")
    print("MODEL TRAINING & EVALUATION")
    print(f"{'='*60}")
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"\n{'Model':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'CV R²':>10}")
    print(f"{'-'*60}")

    best_model = None
    best_r2    = -np.inf

    for name, model in models.items():
        # Use scaled data for linear models
        if 'Regression' in name:
            model.fit(Xs_train, y_train)
            preds = model.predict(Xs_test)
            cv    = cross_val_score(model, Xs_train, y_train, cv=5, scoring='r2')
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            cv    = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)
        cv_r2 = cv.mean()

        print(f"{name:<25} ${mae:>7.0f} ${rmse:>7.0f} {r2:>8.3f} {cv_r2:>10.3f}")

        results[name] = {
            'model': model, 'preds': preds,
            'mae': mae, 'rmse': rmse, 'r2': r2, 'cv_r2': cv_r2,
            'scaled': 'Regression' in name
        }

        if r2 > best_r2:
            best_r2    = r2
            best_model = name

    print(f"\n🏆  Best model: {best_model}  (R² = {best_r2:.3f})")
    return results, X_test, y_test, best_model


# ─────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────

def feature_importance(results, features):
    """Print feature importances from best tree model."""
    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE — Random Forest")
    print(f"{'='*60}")

    rf  = results['Random Forest']['model']
    imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

    for feat, score in imp.items():
        bar = '█' * int(score * 50)
        print(f"  {feat:<30} {score:.3f}  {bar}")


# ─────────────────────────────────────────
# PREDICT NEW FLIGHT
# ─────────────────────────────────────────

def predict_single(results, features):
    """Demo: predict price for a new flight."""
    print(f"\n{'='*60}")
    print("DEMO: PREDICT A NEW FLIGHT")
    print(f"{'='*60}")

    # Manually constructed sample (encoded)
    sample = pd.DataFrame([{
        'airline_enc': 2,            # Delta
        'origin_enc': 3,             # JFK
        'destination_enc': 2,        # LAX
        'duration_hrs': 5.5,
        'stops': 0,
        'days_before_departure': 30,
        'cabin_enc': 0,              # Economy
        'cabin_rank': 0,
        'month': 7,
        'day_of_week': 4,
        'is_weekend_flight': 0,
        'is_holiday_season': 1,
        'booking_window_enc': 2
    }])

    rf    = results['Random Forest']['model']
    price = rf.predict(sample[features])[0]

    print(f"  Route          : JFK → LAX")
    print(f"  Airline        : Delta")
    print(f"  Cabin          : Economy")
    print(f"  Stops          : 0 (direct)")
    print(f"  Days ahead     : 30")
    print(f"  Season         : Summer (holiday)")
    print(f"\n  💰 Predicted price: ${price:,.0f}")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == '__main__':
    print("╔══════════════════════════════════════════════╗")
    print("║   AIRFARE PRICE PREDICTION MODEL             ║")
    print("║   Author: Prashant Anand                     ║")
    print("╚══════════════════════════════════════════════╝")

    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    df             = load_data()
    X, y, features = prepare_features(df)
    results, X_test, y_test, best = train_models(X, y)
    feature_importance(results, features)
    predict_single(results, features)

    print("\n╔══════════════════════════════════════════════╗")
    print("║   MODELING COMPLETE ✓                        ║")
    print("╚══════════════════════════════════════════════╝")
