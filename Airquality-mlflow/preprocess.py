import pandas as pd
import numpy as np

# 1. Load data
df = pd.read_csv("airquality.csv")

# 2. Basic cleanup: strip column names, unify case
df.columns = df.columns.str.strip()

# 3. Parse dates and sort
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.sort_values(["City", "Date"]).reset_index(drop=True)

# 4. Inspect missing values (optional printouts)
print("Missing values per column:")
print(df.isna().sum())

# 5. Convert numeric columns from object to numeric (coerce errors to NaN)
numeric_cols = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene", "AQI"
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 6. Standardize categorical columns
if "City" in df.columns:
    df["City"] = df["City"].astype("category")

if "AQI_Bucket" in df.columns:
    df["AQI_Bucket"] = df["AQI_Bucket"].astype("category")

# 7. Handle missing values
#    a) Drop rows with no date or city
df = df.dropna(subset=["City", "Date"])

#    b) Impute numeric columns groupwise by City (using median)
for col in numeric_cols:
    if col in df.columns:
        df[col] = df.groupby("City")[col].transform(
            lambda x: x.fillna(x.median())
        )

#    c) If still NaNs remain (e.g., all-missing groups), fill with global median
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# 8. Remove obviously invalid values (example: negative pollutant values)
for col in numeric_cols:
    if col in df.columns:
        df.loc[df[col] < 0, col] = np.nan
        # Re-impute after removing negatives
        df[col] = df.groupby("City")[col].transform(
            lambda x: x.fillna(x.median())
        )
        df[col] = df[col].fillna(df[col].median())

# 9. Feature engineering examples

#    a) Day, month, year from Date
df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month
df["day"] = df["Date"].dt.day

#    b) Weekday (0=Monday, 6=Sunday)
df["weekday"] = df["Date"].dt.weekday

#    c) Simple pollution index (mean of main particulate measures where present)
present_pm_cols = [c for c in ["PM2.5", "PM10"] if c in df.columns]
if present_pm_cols:
    df["PM_mean"] = df[present_pm_cols].mean(axis=1)

# 10. Optional: remove duplicate rows
df = df.drop_duplicates()

# 11. Optional: filter unrealistic pollutant ranges
#     Example: cap extremely large values at a high percentile
for col in numeric_cols:
    if col in df.columns:
        upper = df[col].quantile(0.999)
        df[col] = np.clip(df[col], None, upper)

# 12. Save cleaned dataset
# df.to_csv("airquality_cleaned.csv", index=False)

# print("Preprocessing done. Cleaned file saved as airquality_cleaned.csv")
