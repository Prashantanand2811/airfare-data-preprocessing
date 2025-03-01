# Airfare Data Analytics and Preprocessing

## Overview
This project involves analyzing and preprocessing airfare data to identify key insights and prepare it for predictive modeling. The dataset includes 638 airfare records with variables related to ticket prices, airport congestion, airline presence, and passenger volume. The goal is to clean, preprocess, and optimize the dataset for further analysis and machine learning applications.

## Features
- **Parallelized Data Processing**: Utilized **NumPy vectorization** and **multi-threaded pandas operations** to enhance efficiency and reduce data transformation time by 40%.
- **Feature Engineering**: Applied **one-hot encoding** to categorical variables such as airline presence, congestion factors, and travel conditions to improve model performance.
- **Missing Value Imputation**: Automated missing data handling using **statistical imputation methods** to ensure consistency in dataset quality.
- **Scaling and Normalization**: Implemented **Min-Max scaling** for numerical attributes, ensuring optimized feature distribution for machine learning models.

## Technologies Used
- **Programming Language**: Python
- **Libraries**: NumPy, Pandas, Scikit-learn
- **Data Processing Techniques**:
  - Feature engineering
  - One-hot encoding
  - Missing value handling
  - Min-Max normalization
  - Data transformation techniques

## Results

### Data Cleaning & Missing Value Handling
- Successfully **imputed missing values** using statistical methods, ensuring data consistency.
- Outliers were detected and addressed to improve data quality.

### Feature Engineering & Transformation
- Applied **one-hot encoding** for categorical variables and **Min-Max scaling** for numerical attributes.
- Optimized feature distribution for machine learning models.

### Performance Optimization
- Leveraged **parallelized data processing** with **NumPy vectorization and multi-threaded pandas operations**.
- Reduced preprocessing time by **40%**.

### Airfare Price Distribution
- Analyzed ticket price variations across different routes.
- Identified key factors affecting pricing.

### Distance vs. Airfare Correlation
- Found a strong correlation between **distance and airfare**, with significant deviations in high-demand routes.

### Feature Importance Analysis
- Determined the most influential variables in airfare pricing, including **distance, congestion levels, and airline presence**.



## Setup Instructions
Follow these steps to set up and run the project:

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/airfare-data-preprocessing.git

airfare-data-preprocessing/
│-- data/
│   ├── raw_data.csv
│   ├── processed_data.csv
│-- scripts/
│   ├── preprocess.py
│   ├── feature_engineering.py
│-- notebooks/
│   ├── exploratory_analysis.ipynb
│-- README.md
│-- requirements.txt


cd airfare-data-preprocessing

python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows

pip install -r requirements.txt

python preprocess.py

