# E-Commerce Data Preprocessing Pipeline

This is a Python script I put together to automate the boring parts of data cleaning and feature engineering for retail and e-commerce datasets. Instead of manually handling missing values and hunting down outliers every time, this script takes raw transaction data and automatically spits out a clean, ML-ready CSV.

### What it does

The pipeline (`project1_decode.py`) runs through four main steps:

1. **Smart Data Loading:** It tries to load `DATA.csv` by default. If the file is secretly an Excel file in disguise (or named differently), it automatically catches the error and tries a few fallback options to get the data loaded.


2. **Cleaning & Outlier Handling:** It fills missing categorical data (like null `CouponCode` entries) and caps extreme numerical outliers in columns like `Quantity` or `UnitPrice` using the IQR (Winsorization) method so they don't skew the model.


3. **Feature Engineering:** It creates a few new predictive columns based on the existing data, such as calculating the total cost and the ratio of specific items to the overall cart size.


4. **ML Prep & Dimensionality Reduction:** Finally, it preps the data for machine learning. It drops useless metadata (like `TrackingNumber` or `CustomerID`), one-hot encodes the categorical variables, and automatically drops highly correlated features (anything with |r| > 0.80) to prevent collinearity issues.



### How to use it

1. Make sure you have `pandas` and `numpy` installed.


2. Drop your raw dataset into the same folder and name it `DATA.csv`.


3. Run the script:
```bash
python project1_decode.py

```


4. The script will print its progress to the console and generate a `cleaned_DATA.csv` file that is completely ready for model training.
