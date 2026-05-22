"""
Airfare Data Preprocessing & Analysis Pipeline
Author: Prashant Anand
Description: End-to-end pipeline for scraping, cleaning, and analyzing airfare data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# STEP 1: DATA GENERATION (simulates scraped data)
# ─────────────────────────────────────────

def generate_raw_airfare_data(n=500, seed=42):
    """Simulates raw scraped airfare data with real-world messiness."""
    random.seed(seed)
    np.random.seed(seed)

    airlines = ['Delta', 'United', 'American', 'Southwest', 'JetBlue',
                'Spirit', 'Frontier', 'Alaska', 'delta', 'UNITED', None]
    routes = [
        ('JFK', 'LAX'), ('ORD', 'MIA'), ('SFO', 'SEA'), ('BOS', 'DEN'),
        ('ATL', 'LAS'), ('NYC', 'LAX'), ('jfk', 'lax'), ('ORD', None)
    ]
    cabin_classes = ['Economy', 'Business', 'First', 'economy', 'BUSINESS', 'Premium Economy', None]

    data = []
    base_date = datetime(2024, 1, 1)

    for i in range(n):
        route = random.choice(routes)
        origin, dest = route

        # Inject noise: missing values, duplicates, wrong types
        price = round(random.uniform(80, 1800), 2) if random.random() > 0.05 else None
        duration = round(random.uniform(1.5, 14.0), 1) if random.random() > 0.04 else -999
        stops = random.choice([0, 1, 2, None])
        days_ahead = random.randint(1, 180)
        flight_date = (base_date + timedelta(days=days_ahead)).strftime('%Y-%m-%d') \
                      if random.random() > 0.03 else random.choice(['2024/03/15', '15-04-2024', 'invalid'])

        row = {
            'flight_id':    f'FL{i:04d}',
            'airline':      random.choice(airlines),
            'origin':       origin,
            'destination':  dest,
            'flight_date':  flight_date,
            'price_usd':    price,
            'duration_hrs': duration,
            'stops':        stops,
            'cabin_class':  random.choice(cabin_classes),
            'days_before_departure': days_ahead,
            'scraped_at':   datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        data.append(row)

    # inject duplicates
    data += random.sample(data, 30)
    df = pd.DataFrame(data)
    return df


# ─────────────────────────────────────────
# STEP 2: DATA CLEANING
# ─────────────────────────────────────────

def clean_airfare_data(df):
    """Full cleaning pipeline."""
    print(f"\n{'='*50}")
    print("STEP 2: DATA CLEANING")
    print(f"{'='*50}")
    print(f"Raw records      : {len(df)}")

    # 2a. Remove duplicates
    df = df.drop_duplicates(subset=[c for c in df.columns if c != 'scraped_at'])
    print(f"After dedup      : {len(df)}")

    # 2b. Standardize text columns
    df['airline'] = df['airline'].str.strip().str.title()
    df['origin']  = df['origin'].str.upper().str.strip()
    df['destination'] = df['destination'].str.upper().str.strip()
    df['cabin_class'] = df['cabin_class'].str.strip().str.title()

    # 2c. Fix cabin class labels
    cabin_map = {
        'Economy': 'Economy', 'Business': 'Business',
        'First': 'First Class', 'Premium Economy': 'Premium Economy'
    }
    df['cabin_class'] = df['cabin_class'].map(cabin_map)

    # 2d. Parse flight_date (multiple formats)
    def parse_date(val):
        for fmt in ('%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y'):
            try:
                return pd.to_datetime(val, format=fmt)
            except:
                pass
        return pd.NaT

    df['flight_date'] = df['flight_date'].apply(parse_date)

    # 2e. Fix invalid duration values
    df.loc[df['duration_hrs'] < 0, 'duration_hrs'] = np.nan

    # 2f. Drop rows missing critical fields
    before = len(df)
    df = df.dropna(subset=['price_usd', 'airline', 'origin', 'destination', 'flight_date'])
    print(f"After dropna     : {len(df)}  (dropped {before - len(df)} rows)")

    # 2g. Remove price outliers (IQR method)
    Q1, Q3 = df['price_usd'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df = df[(df['price_usd'] >= Q1 - 1.5*IQR) & (df['price_usd'] <= Q3 + 1.5*IQR)]
    print(f"After IQR filter : {len(df)}")

    # 2h. Fill remaining missing stops with mode
    df['stops'] = df['stops'].fillna(df['stops'].mode()[0]).astype(int)

    # 2i. Fill missing duration with median per route
    df['duration_hrs'] = df.groupby(['origin','destination'])['duration_hrs']\
                           .transform(lambda x: x.fillna(x.median()))
    df['duration_hrs'] = df['duration_hrs'].fillna(df['duration_hrs'].median())

    return df.reset_index(drop=True)


# ─────────────────────────────────────────
# STEP 3: FEATURE ENGINEERING
# ─────────────────────────────────────────

def engineer_features(df):
    """Create ML-ready features."""
    print(f"\n{'='*50}")
    print("STEP 3: FEATURE ENGINEERING")
    print(f"{'='*50}")

    df['route'] = df['origin'] + '-' + df['destination']
    df['month'] = df['flight_date'].dt.month
    df['day_of_week'] = df['flight_date'].dt.dayofweek
    df['is_weekend_flight'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_holiday_season'] = df['month'].isin([6, 7, 8, 11, 12]).astype(int)

    # Booking window buckets
    bins   = [0, 7, 14, 30, 60, 180]
    labels = ['Last-min', '1-2wk', '2-4wk', '1-2mo', '2mo+']
    df['booking_window'] = pd.cut(df['days_before_departure'], bins=bins, labels=labels)

    # Price per hour
    df['price_per_hour'] = (df['price_usd'] / df['duration_hrs']).round(2)

    # Log price for modeling
    df['log_price'] = np.log1p(df['price_usd'])

    # Encode cabin class
    cabin_order = {'Economy': 0, 'Premium Economy': 1, 'Business': 2, 'First Class': 3}
    df['cabin_rank'] = df['cabin_class'].map(cabin_order)

    print(f"Features added   : route, month, day_of_week, is_weekend_flight,")
    print(f"                   is_holiday_season, booking_window, price_per_hour,")
    print(f"                   log_price, cabin_rank")
    return df


# ─────────────────────────────────────────
# STEP 4: ANALYSIS
# ─────────────────────────────────────────

def analyze(df):
    """Key insights from cleaned data."""
    print(f"\n{'='*50}")
    print("STEP 4: ANALYSIS & INSIGHTS")
    print(f"{'='*50}")

    print("\n📊 Price Statistics:")
    print(df['price_usd'].describe().round(2).to_string())

    print("\n✈️  Avg Price by Airline (Top 5):")
    print(df.groupby('airline')['price_usd'].mean().sort_values(ascending=False)
            .head(5).round(2).to_string())

    print("\n🛑  Avg Price by Number of Stops:")
    print(df.groupby('stops')['price_usd'].mean().round(2).to_string())

    print("\n💺  Avg Price by Cabin Class:")
    print(df.groupby('cabin_class')['price_usd'].mean().sort_values(ascending=False)
            .round(2).to_string())

    print("\n📅  Avg Price by Booking Window:")
    print(df.groupby('booking_window', observed=True)['price_usd'].mean()
            .round(2).to_string())

    print("\n🔥  Top 5 Most Expensive Routes:")
    print(df.groupby('route')['price_usd'].mean().sort_values(ascending=False)
            .head(5).round(2).to_string())

    corr = df[['price_usd','duration_hrs','stops','days_before_departure']].corr()
    print("\n📈  Correlation with Price:")
    print(corr['price_usd'].drop('price_usd').round(3).to_string())

    return df


# ─────────────────────────────────────────
# STEP 5: EXPORT
# ─────────────────────────────────────────

def export(df):
    df.to_csv('/home/claude/airfare-project/airfare_cleaned.csv', index=False)
    print(f"\n✅  Cleaned dataset saved → airfare_cleaned.csv  ({len(df)} rows, {len(df.columns)} cols)")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == '__main__':
    print("╔══════════════════════════════════════════════╗")
    print("║   AIRFARE DATA PREPROCESSING PIPELINE        ║")
    print("║   Author: Prashant Anand                     ║")
    print("╚══════════════════════════════════════════════╝")

    print(f"\n{'='*50}")
    print("STEP 1: DATA SCRAPING (Simulated)")
    print(f"{'='*50}")
    raw = generate_raw_airfare_data(n=500)
    print(f"Records scraped  : {len(raw)}")
    print(f"Columns          : {list(raw.columns)}")
    print(f"Missing values   :\n{raw.isnull().sum().to_string()}")

    cleaned  = clean_airfare_data(raw)
    featured = engineer_features(cleaned)
    analyzed = analyze(featured)
    export(analyzed)

    print("\n╔══════════════════════════════════════════════╗")
    print("║   PIPELINE COMPLETE ✓                        ║")
    print("╚══════════════════════════════════════════════╝")
