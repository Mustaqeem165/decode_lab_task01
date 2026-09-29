import numpy as np
import pandas as pd

#  LOAD DATA (AUTO-HANDLES EXCEL / CSV)
def load_data():
    try:
        # First attempt: Read as regular CSV
        return pd.read_csv("DATA.csv")
    except (UnicodeDecodeError, Exception):
        try:
            # Second attempt: Read as Excel file named DATA.csv
            return pd.read_excel("DATA.csv")
        except Exception:
            # Fallback: Read original .xlsx name if present in directory
            return pd.read_excel("Dataset for Data Analytics.xlsx")

df = load_data()
print(f"Dataset successfully loaded. Shape: {df.shape}")

#  MODULE 1: MISSING VALUES & OUTLIERS
if "CouponCode" in df.columns:
    df["CouponCode"] = df["CouponCode"].fillna("NONE")

# Neutralize outliers using IQR capping (Winsorization)
num_cols = [c for c in ["Quantity", "UnitPrice", "ItemsInCart"] if c in df.columns]
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = np.clip(df[col], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

#  MODULE 2: VECTORIZED FEATURE ENGINEERING
if "Quantity" in df.columns and "UnitPrice" in df.columns:
    df["calculated_total"] = df["Quantity"] * df["UnitPrice"]

if "ItemsInCart" in df.columns and "UnitPrice" in df.columns:
    df["items_per_unit_price"] = df["ItemsInCart"] / (df["UnitPrice"] + 1e-5)

if "Quantity" in df.columns and "ItemsInCart" in df.columns:
    df["quantity_cart_ratio"] = df["Quantity"] / (df["ItemsInCart"] + 1e-5)

# Drop non-predictive metadata
metadata = ["OrderID", "CustomerID", "ShippingAddress", "TrackingNumber", "Date"]
df = df.drop(columns=[c for c in metadata if c in df.columns])

# One-Hot Encoding
cat_cols = [c for c in ["Product", "PaymentMethod", "OrderStatus", "CouponCode", "ReferralSource"] if c in df.columns]
df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=float)

# Collinearity check & removal (|r| > 0.80)
target = "TotalPrice"
if target in df.columns:
    feature_cols = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
    corr_matrix = df[feature_cols].corr().abs()
    target_corr = df[feature_cols].apply(lambda x: abs(x.corr(df[target])))
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = []
    for c1 in upper_tri.columns:
        for c2 in upper_tri.index:
            if upper_tri.loc[c2, c1] > 0.80:
                col_to_remove = c1 if target_corr[c1] < target_corr[c2] else c2
                if col_to_remove not in to_drop:
                    to_drop.append(col_to_remove)

    if to_drop:
        print(f"Dropping collinear features: {to_drop}")
        df = df.drop(columns=to_drop)

# 4. EXPORT FINAL DATASET
df.to_csv("cleaned_DATA.csv", index=False)
print(f"Pipeline executed successfully. Cleaned dataset saved to 'cleaned_DATA.csv'. Shape: {df.shape}")
print(df.head())