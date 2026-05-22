# ✈️ Airfare Data Preprocessing Pipeline

**Author:** Prashant Anand  
**Stack:** Python · Pandas · NumPy · Scikit-learn ready

## Overview
An end-to-end data pipeline that simulates scraping airfare data, cleans it, engineers features, and surfaces key pricing insights — all production-ready and modular.

## Pipeline Stages

| Stage | Description |
|-------|-------------|
| 🕷️ **Scraping** | Simulates raw scraped data with real-world noise (missing values, duplicates, format inconsistencies) |
| 🧹 **Cleaning** | Deduplication, standardization, date parsing (multi-format), IQR outlier removal, null imputation |
| ⚙️ **Feature Engineering** | Booking window buckets, route encoding, log price, holiday/weekend flags, price-per-hour |
| 📊 **Analysis** | Price by airline, stops, cabin class, booking window, route; correlation analysis |
| 💾 **Export** | Clean CSV output ready for ML modeling |

## Results
- **530** raw records scraped → **377** clean records after preprocessing
- **20** features engineered for downstream modeling
- Key insight: **Direct flights (~$822)** are cheaper than 1-stop (~$990) on average
- **BOS→DEN** is the most expensive route on average

## How to Run

```bash
pip install pandas numpy
python airfare_pipeline.py
```

Output: `airfare_cleaned.csv`

## Project Structure
```
airfare-data-preprocessing/
├── airfare_pipeline.py      # Main pipeline script
├── airfare_cleaned.csv      # Output: cleaned dataset
└── README.md
```

## Skills Demonstrated
- Real-world data cleaning (multi-format dates, IQR filtering, mode/median imputation)
- Feature engineering for ML (log transform, encoding, binning)
- Exploratory data analysis with Pandas
- Modular, production-style Python code
